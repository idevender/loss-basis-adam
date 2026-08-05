"""Attention gauge at scale, deterministic and stochastic twins (Appendix D.9).

--task mod : modular-addition transformer, full-batch, the twin protocol of Section 6.
--task text: char-level LM on real text, mini-batch, same batch schedule for both twins.

Determinism is enforced throughout, so the A=I twin reports drift of exactly 0.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Sink, pick_device, dtype_of, newton_schulz

torch.use_deterministic_algorithms(True)
EPS = 1e-8


def linit(out_dim, in_dim, gen):
    b = 1.0 / math.sqrt(in_dim)
    return (torch.rand(out_dim, in_dim, generator=gen) * 2 - 1) * b


class Model(nn.Module):
    """Pre-norm transformer, bias-free, no qk-norm (exact gauge), one-hot embeddings."""

    def __init__(self, vocab_in, vocab_out, n_ctx, depth, d, heads, mlp_dim, causal, gen):
        super().__init__()
        self.depth, self.d, self.heads, self.hd = depth, d, heads, d // heads
        self.causal, self.vocab_in = causal, vocab_in
        self.scale = self.hd ** -0.5
        P = nn.Parameter
        self.emb = P(torch.randn(vocab_in, d, generator=gen))
        self.pos = P(torch.randn(1, n_ctx, d, generator=gen) * 0.02)
        self.wq = nn.ParameterList([P(linit(d, d, gen)) for _ in range(depth)])
        self.wk = nn.ParameterList([P(linit(d, d, gen)) for _ in range(depth)])
        self.wv = nn.ParameterList([P(linit(d, d, gen)) for _ in range(depth)])
        self.wo = nn.ParameterList([P(linit(d, d, gen)) for _ in range(depth)])
        self.g1 = nn.ParameterList([P(torch.ones(d)) for _ in range(depth)])
        self.g2 = nn.ParameterList([P(torch.ones(d)) for _ in range(depth)])
        self.w1 = nn.ParameterList([P(linit(mlp_dim, d, gen)) for _ in range(depth)])
        self.w2 = nn.ParameterList([P(linit(d, mlp_dim, gen)) for _ in range(depth)])
        self.gf = P(torch.ones(d))
        self.head = P(linit(vocab_out, d, gen))

    def rms(self, x, g):
        return F.normalize(x, dim=-1) * (self.d ** 0.5) * g

    def forward(self, idx, last_only):
        B, T = idx.shape
        oh = F.one_hot(idx, self.vocab_in).to(self.emb.dtype)
        h = oh @ self.emb + self.pos[:, :T]
        mask = None
        if self.causal:
            mask = torch.full((T, T), float('-inf'), device=idx.device, dtype=self.emb.dtype)
            mask = torch.triu(mask, diagonal=1)
        for i in range(self.depth):
            a = self.rms(h, self.g1[i])
            q = (a @ self.wq[i].T).view(B, T, self.heads, self.hd).transpose(1, 2)
            k = (a @ self.wk[i].T).view(B, T, self.heads, self.hd).transpose(1, 2)
            v = (a @ self.wv[i].T).view(B, T, self.heads, self.hd).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                att = att + mask
            att = att.softmax(dim=-1)
            y = (att @ v).transpose(1, 2).reshape(B, T, self.d)
            h = h + y @ self.wo[i].T
            m = self.rms(h, self.g2[i])
            h = h + F.silu(m @ self.w1[i].T) @ self.w2[i].T
        hn = self.rms(h, self.gf)
        if last_only:
            hn = hn[:, -1]
        return hn @ self.head.T

    def matrix_params(self):
        s = set()
        for pl in (self.wq, self.wk, self.wv, self.wo, self.w1, self.w2):
            for p in pl:
                s.add(id(p))
        return s


def apply_gauge(model, seed, identity=False):
    g = torch.Generator().manual_seed(seed)
    hd = model.hd
    with torch.no_grad():
        for i in range(model.depth):
            for h in range(model.heads):
                sl = slice(h * hd, (h + 1) * hd)
                if identity:
                    A = torch.eye(hd, dtype=torch.float64)
                else:
                    A, _ = torch.linalg.qr(torch.randn(hd, hd, generator=g, dtype=torch.float64))
                A = A.to(device=model.wq[i].device, dtype=model.wq[i].dtype)
                model.wq[i][sl, :] = A @ model.wq[i][sl, :]
                model.wk[i][sl, :] = A @ model.wk[i][sl, :]


def apply_noise(model, seed, scale):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for i in range(model.depth):
            for W in (model.wq[i], model.wk[i]):
                W.add_(torch.randn(W.shape, generator=g, dtype=torch.float64)
                       .to(device=W.device, dtype=W.dtype) * scale)


class Trainer:
    def __init__(self, model, kind, lr):
        self.kind, self.lr = kind, lr
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.is_mx = [id(p) in model.matrix_params() for p in self.params]
        self.mom = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2 = 0.9, 0.999
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
                if self.is_mx[i]:
                    self.mom[i].mul_(0.9).add_(G)
                    p.add_(newton_schulz(self.mom[i]), alpha=-self.lr)
                else:
                    self.mom[i].mul_(b1).add_(G, alpha=1 - b1)
                    self.v[i].mul_(b2).addcmul_(G, G, value=1 - b2)
                    mh = self.mom[i] / (1 - b1 ** self.t)
                    vh = self.v[i] / (1 - b2 ** self.t)
                    p.addcdiv_(mh, vh.sqrt() + EPS, value=-1e-3)
            else:
                raise ValueError(self.kind)


def qk_products(model):
    outs = []
    hd = model.hd
    with torch.no_grad():
        for i in range(model.depth):
            for h in range(model.heads):
                sl = slice(h * hd, (h + 1) * hd)
                outs.append(model.wq[i][sl, :].T @ model.wk[i][sl, :])
    return outs


def drift(m1, m2, xva, last_only):
    with torch.no_grad():
        l1, l2 = m1(xva, last_only), m2(xva, last_only)
        num = (l1 - l2).norm().item()
        den = 0.5 * (l1.norm().item() + l2.norm().item())
        qk = [((a - b).norm() / (0.5 * (a.norm() + b.norm()) + 1e-12)).item()
              for a, b in zip(qk_products(m1), qk_products(m2))]
    return num / (den + 1e-12), float(np.mean(qk)), float(np.max(qk)), qk


def mod_data(p_mod, frac, seed, device):
    g = torch.Generator().manual_seed(seed)
    a = torch.arange(p_mod).repeat_interleave(p_mod)
    b = torch.arange(p_mod).repeat(p_mod)
    c = (a + b) % p_mod
    eq = torch.full_like(a, p_mod)
    x = torch.stack([a, b, eq], dim=1)
    perm = torch.randperm(p_mod * p_mod, generator=g)
    ntr = int(frac * p_mod * p_mod)
    tr, va = perm[:ntr], perm[ntr:]
    return (x[tr].to(device), c[tr].to(device)), (x[va].to(device), c[va].to(device))


def text_data(path, ctx, batch, steps, seed, device, val_batch=64):
    raw = open(path, 'rb').read().decode('utf-8', errors='ignore')
    chars = sorted(set(raw))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in raw], dtype=torch.long)
    ntr = int(0.9 * len(data))
    train, val = data[:ntr], data[ntr:]
    g = torch.Generator().manual_seed(seed)
    ix = torch.randint(0, len(train) - ctx - 1, (steps, batch), generator=g)
    twins = train.unfold(0, ctx + 1, 1)
    vix = torch.randint(0, len(val) - ctx - 1, (val_batch,), generator=g)
    vwins = val.unfold(0, ctx + 1, 1)
    xva = vwins[vix][:, :-1].to(device)
    yva = vwins[vix][:, 1:].to(device)
    return len(chars), twins, ix, (xva, yva)


def ce_loss(logits, targets, vocab):
    logp = logits.log_softmax(dim=-1)
    oh = F.one_hot(targets, vocab).to(logits.dtype)
    return -(logp * oh).sum(dim=-1).mean()


def run_pair(args, kind, lr, twin, init_seed, twin_seed, noise_scale, device, dtype):
    torch.manual_seed(init_seed)
    gen = torch.Generator().manual_seed(init_seed)
    causal = args.task == 'text'
    if args.task == 'mod':
        vocab_in, vocab_out, n_ctx = args.p + 1, args.p, 3
    else:
        vocab_out = args._vocab
        vocab_in, n_ctx = vocab_out, args.ctx
    mk = lambda: Model(vocab_in, vocab_out, n_ctx, args.depth, args.dmodel, args.heads,
                       4 * args.dmodel, causal, torch.Generator().manual_seed(init_seed))
    base = mk()
    m1, m2 = mk(), mk()
    m1.load_state_dict(base.state_dict())
    m2.load_state_dict(base.state_dict())
    for m in (m1, m2):
        m.to(device=device, dtype=dtype)
    if twin == 'gauge':
        apply_gauge(m2, twin_seed)
    elif twin == 'identity':
        apply_gauge(m2, twin_seed, identity=True)
    elif twin == 'noise':
        apply_noise(m2, twin_seed, noise_scale)

    if args.task == 'mod':
        (xtr, ytr), (xva, yva) = mod_data(args.p, 0.4, args.data_seed, device)
        last_only = True
    else:
        vocab, twins, ix, (xva, yva) = args._textdata
        last_only = False

    checks = sorted(set([c for c in (1, 10, 50, 100, 300, 600, 1000, 1500, 2000, 3000)
                         if c <= args.steps] + [args.steps]))
    d0 = drift(m1, m2, xva, last_only)
    curve = [(0, d0[0], d0[1], d0[2])]
    t1, t2 = Trainer(m1, kind, lr), Trainer(m2, kind, lr)
    tstart = time.time()
    for step in range(1, args.steps + 1):
        if args.task == 'mod':
            xb, yb = xtr, ytr
        else:
            w = twins[ix[step - 1]]
            xb, yb = w[:, :-1].to(device), w[:, 1:].to(device)
        for m, tr in ((m1, t1), (m2, t2)):
            for p in tr.params:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            logits = m(xb, last_only)
            loss = ce_loss(logits, yb, logits.shape[-1])
            loss.backward()
            tr.step()
        if not np.isfinite(loss.item()):
            break
        if step in checks:
            d = drift(m1, m2, xva, last_only)
            curve.append((step, d[0], d[1], d[2]))
    final = drift(m1, m2, xva, last_only)
    with torch.no_grad():
        if args.task == 'mod':
            a1 = (m1(xva, True).argmax(-1) == yva).float().mean().item()
            a2 = (m2(xva, True).argmax(-1) == yva).float().mean().item()
            v1 = v2 = float('nan')
        else:
            v1 = ce_loss(m1(xva, False), yva, vocab).item()
            v2 = ce_loss(m2(xva, False), yva, vocab).item()
            a1 = a2 = float('nan')
        trl = loss.item()
    return dict(curve=curve, per_head_final=final[3], acc1=a1, acc2=a2,
                val1=v1, val2=v2, trainloss=trl, secs=round(time.time() - tstart, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['mod', 'text'], required=True)
    ap.add_argument('--p', type=int, default=97)
    ap.add_argument('--data', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'text', 'input.txt'))
    ap.add_argument('--ctx', type=int, default=128)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--dmodel', type=int, default=128)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--kinds', default='adam,sgd,adam_p0rms,muon')
    ap.add_argument('--init-seeds', default='42,43')
    ap.add_argument('--gauge-seeds', default='777,888,999')
    ap.add_argument('--noise-scales', default='1e-7,1e-5')
    ap.add_argument('--data-seed', type=int, default=42)
    ap.add_argument('--lr-adam', type=float, default=1e-3)
    ap.add_argument('--lr-sgd', type=float, default=None)
    ap.add_argument('--lr-p0', type=float, default=3e-3)
    ap.add_argument('--lr-muon', type=float, default=0.02)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--dtype', default='float64')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    device = pick_device(args.device)
    dtype = dtype_of(args.dtype)
    lr_sgd = args.lr_sgd if args.lr_sgd is not None else (0.5 if args.task == 'mod' else 0.1)
    LR = dict(adam=args.lr_adam, sgd=lr_sgd, adam_p0rms=args.lr_p0, muon=args.lr_muon)
    iseeds = [int(s) for s in args.init_seeds.split(',')]
    gseeds = [int(s) for s in args.gauge_seeds.split(',')]
    nscales = [float(s) for s in args.noise_scales.split(',')]
    kinds = args.kinds.split(',')

    if args.task == 'text':
        vocab, twins, ix, vb = None, None, None, None
        v, twins, ix, vb = text_data(args.data, args.ctx, args.batch, args.steps,
                                     args.data_seed, device)
        args._vocab = v
        args._textdata = (v, twins, ix, vb)

    cfg = f"{args.task}|p{args.p if args.task=='mod' else 'txt'}|L{args.depth}|d{args.dmodel}|h{args.heads}|s{args.steps}"
    sink = Sink(args.out, dict(exp='attention_suite', cfg=cfg, dtype=args.dtype, lrs=LR))
    print(f"[attention_suite] {cfg} kinds={kinds} device={device}", flush=True)

    jobs = []
    for kind in kinds:
        gp = [(i, g) for i in iseeds for g in gseeds]
        gp = gp[:6] if kind == 'adam' else gp[:2]
        for i, g in gp:
            jobs.append((kind, 'gauge', i, g, 0.0))
        ns = nscales if kind == 'adam' else nscales[:1]
        for s in ns:
            jobs.append((kind, 'noise', iseeds[0], 888, s))
        if kind == 'adam':
            jobs.append((kind, 'identity', iseeds[0], 0, 0.0))

    for kind, twin, iseed, tseed, scale in jobs:
        key = f'{cfg}|{kind}|{twin}|i{iseed}|t{tseed}|ns{scale:g}'
        if sink.has(key):
            continue
        r = run_pair(args, kind, LR[kind], twin, iseed, tseed, scale, device, dtype)
        sink.add(key, task=args.task, cfg=cfg, opt=kind, twin=twin, init_seed=iseed,
                 twin_seed=tseed, noise_scale=scale, lr=LR[kind], **r)
        c = r['curve']
        s1 = next((x[1] for x in c if x[0] == 1), float('nan'))
        print(f"  {key}: step1={s1:.2e} final={c[-1][1]:.2e} "
              f"acc=({r['acc1']:.3f},{r['acc2']:.3f}) val=({r['val1']:.3f},{r['val2']:.3f}) "
              f"train={r['trainloss']:.4f} {r['secs']}s", flush=True)
        if twin == 'identity':
            ok = c[-1][1] == 0.0
            print(f"  DETERMINISM-{'OK' if ok else 'FAIL'} ({cfg})", flush=True)
    print("[attention_suite] DONE", flush=True)


if __name__ == '__main__':
    main()
