"""
Attention-head gauge experiment (Section 7).

The QK^T factorization in an attention head has the same gauge symmetry as UV^T: for a per-head
orthogonal A_h, W_Q^h -> A_h W_Q^h and W_K^h -> A_h W_K^h leave the logits q_h . k_h unchanged for
every input. Two such initializations are the same function under any gauge-equivariant optimizer
(SGD, momentum, shared-scalar Adam, Muon), while Adam's per-coordinate second moment reads the
arbitrary basis and the two runs split in function space.

Protocol: mod-47 addition transformer (2 layers, 4 heads, d=64, no qk-norm so the gauge is exact),
full-batch, deterministic CPU, identical data order; an A=I twin is the float-noise control.
Metrics: relative logit drift, per-layer QK^T product drift, and validation accuracy of both copies.
"""
from __future__ import annotations
import math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.use_deterministic_algorithms(True)

P = 47
TRAIN_FRAC = 0.4
D_MODEL = 64
DEPTH = 2
HEADS = 4
HEAD_DIM = D_MODEL // HEADS
MLP_DIM = 256
STEPS = 1500
CHECKS = [1, 10, 50, 100, 300, 600, 1000, 1500]
DEVICE = "cpu"
SEED = 42
EPS = 1e-8


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g


class Attention(nn.Module):
    """Standard attention, NO qk-norm (so the per-head orthogonal gauge is exact)."""
    def __init__(self, dim=D_MODEL, heads=HEADS):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(b, t, c)
        return self.o_proj(y)


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(P + 1, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, 3, D_MODEL) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": RMSNorm(D_MODEL),
                "attn": Attention(),
                "norm2": RMSNorm(D_MODEL),
                "mlp": nn.Sequential(
                    nn.Linear(D_MODEL, MLP_DIM, bias=False),
                    nn.SiLU(),
                    nn.Linear(MLP_DIM, D_MODEL, bias=False),
                ),
            })
            for _ in range(DEPTH)
        ])
        self.norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, P, bias=False)

    def forward(self, x):
        h = self.embedding(x) + self.pos[:, :x.shape[1]]
        for layer in self.layers:
            h = h + layer["attn"](layer["norm1"](h))
            h = h + layer["mlp"](layer["norm2"](h))
        return self.head(self.norm(h[:, -1]))


def make_data(seed=SEED):
    g = torch.Generator().manual_seed(seed)
    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    c = (a + b) % P
    eq = torch.full_like(a, P)
    x = torch.stack([a, b, eq], dim=1)
    perm = torch.randperm(P * P, generator=g)
    n_train = int(TRAIN_FRAC * P * P)
    tr, va = perm[:n_train], perm[n_train:]
    return (x[tr], c[tr]), (x[va], c[va])


def apply_gauge(model, seed, identity=False):
    """Per-head orthogonal A_h on the ROW blocks of W_Q and W_K (same A_h for both).
    q_h = W_Q^h x -> A_h W_Q^h x ;  q_h . k_h invariant. Function unchanged for every input."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in model.layers:
            att = layer["attn"]
            for h in range(HEADS):
                sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
                if identity:
                    A = torch.eye(HEAD_DIM)
                else:
                    A, _ = torch.linalg.qr(torch.randn(HEAD_DIM, HEAD_DIM, generator=g))
                att.q_proj.weight[sl, :] = A @ att.q_proj.weight[sl, :]
                att.k_proj.weight[sl, :] = A @ att.k_proj.weight[sl, :]
    return model


def newton_schulz(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)
    transpose = G.shape[0] > G.shape[1]
    if transpose:
        X = X.T
    for _ in range(steps):
        AA = X @ X.T
        B = b * AA + c * (AA @ AA)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X


class Trainer:
    """One optimizer, hand-rolled so scalar-Adam/Muon share the exact harness."""
    def __init__(self, model, kind, lr):
        self.model, self.kind, self.lr = model, kind, lr
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.mom = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
        self.muon_mask = [p.dim() == 2 and p.shape[0] != P + 1 and p.shape[0] != P
                          for p in self.params]

    def step(self):
        self.t += 1
        b1, b2 = 0.9, 0.999
        with torch.no_grad():
            if self.kind == 'adam_p0rms':
                tot, cnt = 0.0, 0
                for i, p in enumerate(self.params):
                    G = p.grad
                    self.mom[i].mul_(b1).add_(G, alpha=1 - b1)
                    self.v[i].mul_(b2).addcmul_(G, G, value=1 - b2)
                    tot += (self.v[i] / (1 - b2 ** self.t)).sum().item()
                    cnt += self.v[i].numel()
                s = math.sqrt(tot / cnt) + EPS
                for i, p in enumerate(self.params):
                    p.add_(self.mom[i] / (1 - b1 ** self.t), alpha=-self.lr / s)
                return
            for i, p in enumerate(self.params):
                G = p.grad
                if self.kind == 'sgd':
                    self.mom[i].mul_(0.9).add_(G)
                    p.add_(self.mom[i], alpha=-self.lr)
                elif self.kind == 'adam':
                    self.mom[i].mul_(b1).add_(G, alpha=1 - b1)
                    self.v[i].mul_(b2).addcmul_(G, G, value=1 - b2)
                    mh = self.mom[i] / (1 - b1 ** self.t)
                    vh = self.v[i] / (1 - b2 ** self.t)
                    p.addcdiv_(mh, vh.sqrt() + EPS, value=-self.lr)
                elif self.kind == 'muon':
                    if self.muon_mask[i]:
                        self.mom[i].mul_(0.9).add_(G)
                        p.add_(newton_schulz(self.mom[i]), alpha=-self.lr)
                    else:
                        self.mom[i].mul_(b1).add_(G, alpha=1 - b1)
                        self.v[i].mul_(b2).addcmul_(G, G, value=1 - b2)
                        mh = self.mom[i] / (1 - b1 ** self.t)
                        vh = self.v[i] / (1 - b2 ** self.t)
                        p.addcdiv_(mh, vh.sqrt() + EPS, value=-1e-3)


def qk_products(model):
    outs = []
    with torch.no_grad():
        for layer in model.layers:
            att = layer["attn"]
            for h in range(HEADS):
                sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
                outs.append(att.q_proj.weight[sl, :].T @ att.k_proj.weight[sl, :])
    return outs


def drift(model1, model2, xva):
    with torch.no_grad():
        l1, l2 = model1(xva), model2(xva)
        num = (l1 - l2).norm().item()
        den = 0.5 * (l1.norm().item() + l2.norm().item())
        ms1, ms2 = qk_products(model1), qk_products(model2)
        qk = np.mean([ (a - b).norm().item() / (0.5 * (a.norm() + b.norm()).item() + 1e-12)
                       for a, b in zip(ms1, ms2) ])
    return num / (den + 1e-12), qk


def run_pair(kind, lr, gauge_seed, identity=False):
    torch.manual_seed(SEED)
    base = TinyTransformer()
    m1 = TinyTransformer(); m1.load_state_dict(base.state_dict())
    m2 = TinyTransformer(); m2.load_state_dict(base.state_dict())
    apply_gauge(m2, gauge_seed, identity=identity)
    (xtr, ytr), (xva, yva) = make_data()
    d0, qk0 = drift(m1, m2, xva)
    t1, t2 = Trainer(m1, kind, lr), Trainer(m2, kind, lr)
    curve = [(0, d0, qk0)]
    for step in range(1, STEPS + 1):
        for m, tr in ((m1, t1), (m2, t2)):
            for p in tr.params:
                if p.grad is not None:
                    p.grad.detach_(); p.grad.zero_()
            loss = F.cross_entropy(m(xtr), ytr)
            loss.backward()
            tr.step()
        if step in CHECKS:
            d, qk = drift(m1, m2, xva)
            curve.append((step, d, qk))
    with torch.no_grad():
        acc1 = (m1(xva).argmax(-1) == yva).float().mean().item()
        acc2 = (m2(xva).argmax(-1) == yva).float().mean().item()
        trl = F.cross_entropy(m1(xtr), ytr).item()
    return curve, acc1, acc2, trl


def main():
    t0 = time.time()
    print("=" * 112, flush=True)
    print(f"ATTENTION GAUGE | mod-{P} add | {DEPTH}L {HEADS}H d{D_MODEL} (no qk-norm) | full-batch, "
          f"deterministic CPU | {STEPS} steps", flush=True)
    print("Two GAUGE-EQUIVALENT inits (identical function). Equivariant optimizers must stay identical;", flush=True)
    print("Adam is predicted to SPLIT. 'drift' = rel logit dist on val; 'qk' = rel QK^T product dist.", flush=True)
    print("=" * 112, flush=True)
    jobs = [
        ('adam',        1e-3, False, 'per-coordinate  -> predicted SPLIT'),
        ('adam',        1e-3, True,  'A=I control     -> float floor'),
        ('sgd',         0.5,  False, 'equivariant     -> predicted identical'),
        ('adam_p0rms',  3e-3, False, 'shared scalar   -> predicted identical (the dial fixes Adam)'),
        ('muon',        0.02, False, 'msign equivariant-> predicted identical'),
    ]
    for kind, lr, ident, note in jobs:
        curve, a1, a2, trl = run_pair(kind, lr, gauge_seed=777, identity=ident)
        tag = f"{kind}{' [A=I ctrl]' if ident else ''}"
        print(f"\n{tag:>22} lr={lr:g} | {note}", flush=True)
        print(f"{'':>8}step:  " + "  ".join(f"{s:>7d}" for s, _, _ in curve), flush=True)
        print(f"{'':>8}logit: " + "  ".join(f"{d:>7.1e}" for _, d, _ in curve), flush=True)
        print(f"{'':>8}QK^T:  " + "  ".join(f"{q:>7.1e}" for _, _, q in curve), flush=True)
        print(f"{'':>8}final: val acc copy1={a1:.4f}  copy2={a2:.4f}  |Δacc|={abs(a1-a2):.4f}  "
              f"train loss={trl:.3f}", flush=True)
    print("\nREAD: if Adam's drift is orders of magnitude above its A=I control AND above SGD/Muon/scalar-", flush=True)
    print("Adam, then what Adam learns in attention depends on an invisible basis choice - the gauge-", flush=True)
    print("breaking mechanism operates in TRANSFORMERS, not just synthetic matrix problems.", flush=True)
    print(f"\n[done in {(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == '__main__':
    main()


def apply_noise(model, seed, scale=1e-7):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in model.layers:
            att = layer["attn"]
            att.q_proj.weight.add_(torch.randn(att.q_proj.weight.shape, generator=g) * scale)
            att.k_proj.weight.add_(torch.randn(att.k_proj.weight.shape, generator=g) * scale)
    return model


def run_pair_noise(kind, lr, noise_seed):
    torch.manual_seed(SEED)
    base = TinyTransformer()
    m1 = TinyTransformer(); m1.load_state_dict(base.state_dict())
    m2 = TinyTransformer(); m2.load_state_dict(base.state_dict())
    apply_noise(m2, noise_seed)
    (xtr, ytr), (xva, yva) = make_data()
    d0, qk0 = drift(m1, m2, xva)
    t1, t2 = Trainer(m1, kind, lr), Trainer(m2, kind, lr)
    curve = [(0, d0, qk0)]
    for step in range(1, STEPS + 1):
        for m, tr in ((m1, t1), (m2, t2)):
            for p in tr.params:
                if p.grad is not None:
                    p.grad.detach_(); p.grad.zero_()
            loss = F.cross_entropy(m(xtr), ytr)
            loss.backward()
            tr.step()
        if step in CHECKS:
            d, qk = drift(m1, m2, xva)
            curve.append((step, d, qk))
    with torch.no_grad():
        acc1 = (m1(xva).argmax(-1) == yva).float().mean().item()
        acc2 = (m2(xva).argmax(-1) == yva).float().mean().item()
    return curve, acc1, acc2


def main_noise():
    t0 = time.time()
    print("=" * 112, flush=True)
    print(f"ATTENTION GAUGE v2 - NOISE-TWIN CONTROL | same basis, copy2 q/k weights + 1e-7 gaussian", flush=True)
    print("Discriminates STRUCTURAL basis-dependence (gauge-twin >> noise-twin at step 1: Adam) from", flush=True)
    print("CHAOTIC float amplification (gauge-twin ~ noise-twin curves: suspected for Muon).", flush=True)
    print("=" * 112, flush=True)
    for kind, lr in [('adam', 1e-3), ('muon', 0.02), ('sgd', 0.5), ('adam_p0rms', 3e-3)]:
        curve, a1, a2 = run_pair_noise(kind, lr, noise_seed=888)
        print(f"\n{kind:>12} lr={lr:g}  [noise-twin]", flush=True)
        print(f"{'':>8}step:  " + "  ".join(f"{s:>7d}" for s, _, _ in curve), flush=True)
        print(f"{'':>8}logit: " + "  ".join(f"{d:>7.1e}" for _, d, _ in curve), flush=True)
        print(f"{'':>8}final: acc1={a1:.4f} acc2={a2:.4f} |Δacc|={abs(a1-a2):.4f}", flush=True)
    print(f"\n[done in {(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'noise':
    main_noise()
