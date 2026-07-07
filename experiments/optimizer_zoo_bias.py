"""
Optimizer-zoo geometry map (Section 5).

Runs nine optimizers on the factored matrix-sensing task and measures whether each preserves or
destroys gradient descent's low-rank implicit bias. The factored loss L(UV^T) is invariant under
(U, V) -> (UQ, VQ) for orthogonal Q. Optimizers whose update commutes with this action (GD,
momentum, shared-scalar Adam, Muon, Shampoo) follow gauge-covariant trajectories and preserve the
bias; coordinate-wise methods (Adam, RMSProp, signum, Lion, Adafactor) do not and are predicted to
destroy it.

Testbed: wd=0 sensing ladder (40x40, rank 3, k=40, m = 2 x dof), small init, run to interpolation,
3 seeds. Reports recovery error, effective rank, and the balancedness invariant ||U^T U - V^T V||_F.
"""
from __future__ import annotations
import math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from restoration_probe import make_problem, factors, stats, N, R_STAR, K, M, DOF, INIT

MAX_STEPS = 20000
TRAIN_TOL = 1e-7
EPS = 1e-8
SEEDS = [42, 123, 456]


def loss_of(U, V, A, y):
    W = (U @ V.T).reshape(-1)
    return ((A @ W - y) ** 2).mean()


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


def matrix_pow(Mx, p, eps=1e-6):
    Mx = 0.5 * (Mx + Mx.T)
    vals, vecs = torch.linalg.eigh(Mx)
    vals = torch.clamp(vals, min=eps)
    return (vecs * vals.pow(p)) @ vecs.T


def balancedness(U, V):
    with torch.no_grad():
        return (U.T @ U - V.T @ V).norm().item()


DECAYED = {'muon', 'signum', 'lion'}


def run(kind, seed, lr):
    Xs, A, y = make_problem(seed)
    U, V = factors(seed)
    params = [U, V]
    b1, b2 = 0.9, 0.999
    mom = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    Lp = [torch.zeros(p.shape[0], p.shape[0]) for p in params]
    Rp = [torch.zeros(p.shape[1], p.shape[1]) for p in params]
    Linv = [None, None]; Rinv = [None, None]
    Rrow = [torch.zeros(p.shape[0]) for p in params]
    Ccol = [torch.zeros(p.shape[1]) for p in params]
    train = 1.0
    for step in range(1, MAX_STEPS + 1):
        for p in params:
            if p.grad is not None:
                p.grad.detach_(); p.grad.zero_()
        loss = loss_of(U, V, A, y)
        train = loss.item()
        if not np.isfinite(train) or train > 1e8:
            return dict(rec=float('nan'), nuc=float('nan'), er=float('nan'),
                        train=float('nan'), bal=float('nan'), steps=step)
        loss.backward()
        lr_t = lr * 0.5 * (1 + math.cos(math.pi * step / MAX_STEPS)) if kind in DECAYED else lr
        with torch.no_grad():
            if kind == 'adam_p0rms':
                tot, cnt = 0.0, 0
                for i, p in enumerate(params):
                    G = p.grad
                    mom[i].mul_(b1).add_(G, alpha=1 - b1)
                    v[i].mul_(b2).addcmul_(G, G, value=1 - b2)
                    tot += (v[i] / (1 - b2 ** step)).sum().item()
                    cnt += v[i].numel()
                s = math.sqrt(tot / cnt) + EPS
                for i, p in enumerate(params):
                    mh = mom[i] / (1 - b1 ** step)
                    p.add_(mh, alpha=-lr_t / s)
            else:
                for i, p in enumerate(params):
                    G = p.grad
                    if kind == 'gd':
                        p.add_(G, alpha=-lr_t)
                    elif kind == 'adam':
                        mom[i].mul_(b1).add_(G, alpha=1 - b1)
                        v[i].mul_(b2).addcmul_(G, G, value=1 - b2)
                        mh = mom[i] / (1 - b1 ** step); vh = v[i] / (1 - b2 ** step)
                        p.addcdiv_(mh, vh.sqrt() + EPS, value=-lr_t)
                    elif kind == 'rmsprop':
                        v[i].mul_(0.99).addcmul_(G, G, value=0.01)
                        p.addcdiv_(G, v[i].sqrt() + EPS, value=-lr_t)
                    elif kind == 'signum':
                        mom[i].mul_(b1).add_(G, alpha=1 - b1)
                        p.add_(torch.sign(mom[i]), alpha=-lr_t)
                    elif kind == 'lion':
                        upd = torch.sign(b1 * mom[i] + (1 - b1) * G)
                        mom[i].mul_(0.99).add_(G, alpha=0.01)
                        p.add_(upd, alpha=-lr_t)
                    elif kind == 'adafactor':
                        g2 = G * G
                        Rrow[i].mul_(b2).add_(g2.mean(dim=1), alpha=1 - b2)
                        Ccol[i].mul_(b2).add_(g2.mean(dim=0), alpha=1 - b2)
                        vhat = torch.outer(Rrow[i], Ccol[i]) / (Rrow[i].mean() + 1e-30)
                        p.addcdiv_(G, vhat.sqrt() + EPS, value=-lr_t)
                    elif kind == 'muon':
                        mom[i].mul_(b1).add_(G)
                        p.add_(newton_schulz(mom[i]), alpha=-lr_t)
                    elif kind == 'shampoo':
                        Lp[i] += G @ G.T; Rp[i] += G.T @ G
                        if (step - 1) % 20 == 0 or Linv[i] is None:
                            Linv[i] = matrix_pow(Lp[i] + 1.0 * torch.eye(G.shape[0]), -0.25)
                            Rinv[i] = matrix_pow(Rp[i] + 1.0 * torch.eye(G.shape[1]), -0.25)
                        p.add_(Linv[i] @ G @ Rinv[i], alpha=-lr_t)
        if step % 200 == 0 and train < TRAIN_TOL:
            break
    rec, nuc, er = stats(U, V, Xs)
    return dict(rec=rec, nuc=nuc, er=er, train=train, bal=balancedness(U, V), steps=step)


def best(kind, lrs):
    """Pick lr that interpolates (train<1e-4) with lowest recovery; fall back to lowest train."""
    best_row = None
    for lr in lrs:
        per = [run(kind, s, lr) for s in SEEDS]
        rec = np.mean([p['rec'] for p in per]); tr = np.mean([p['train'] for p in per])
        er = np.mean([p['er'] for p in per]); nuc = np.mean([p['nuc'] for p in per])
        bal = np.mean([p['bal'] for p in per])
        if not (np.isfinite(rec) and np.isfinite(tr)):
            continue
        score = rec if tr < 1e-4 else rec + 10 + tr
        if best_row is None or score < best_row['score']:
            best_row = dict(score=score, rec=rec, tr=tr, er=er, nuc=nuc, bal=bal, lr=lr)
    if best_row is None:
        best_row = dict(score=float('nan'), rec=float('nan'), tr=float('nan'),
                        er=float('nan'), nuc=float('nan'), bal=float('nan'), lr=float('nan'))
    return best_row


def main():
    t0 = time.time()
    print("=" * 110, flush=True)
    print(f"OPTIMIZER-ZOO GEOMETRY MAP | sensing {N}x{N} r*{R_STAR} k{K} m={M} ({M/DOF:.1f}x dof) "
          f"init={INIT} wd=0 | {len(SEEDS)} seeds | decayed lr for {sorted(DECAYED)}", flush=True)
    print("Prediction: EQUIVARIANT (gd, adam_p0rms, muon, shampoo) preserve the low-rank bias;", flush=True)
    print("COORDINATE-WISE (adam, rmsprop, signum, lion, adafactor) destroy it. bal = |U^TU-V^TV|_F.", flush=True)
    print("=" * 110, flush=True)
    print(f"{'method':>12} | {'equivariant?':>12} | {'recovery':>9} | {'train':>9} | {'eff_rank':>8} | "
          f"{'bal':>9} | {'lr':>6} | predicted", flush=True)
    print("-" * 110, flush=True)
    configs = [
        ('gd',         'YES',  [0.01, 0.03, 0.1, 0.3],           'preserve (anchor)'),
        ('adam',       'no',   [1e-3, 1e-2, 3e-2],               'destroy (anchor)'),
        ('adam_p0rms', 'YES',  [1e-3, 3e-3, 1e-2, 3e-2],         'preserve (scalar = time-rescaled flow)'),
        ('rmsprop',    'no',   [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],   'destroy (per-coord)'),
        ('signum',     'no',   [1e-4, 3e-4, 1e-3, 3e-3],         'destroy HARD (l_inf extreme)'),
        ('lion',       'no',   [1e-4, 3e-4, 1e-3, 3e-3],         'destroy HARD (l_inf extreme)'),
        ('adafactor',  'no',   [3e-4, 1e-3, 3e-3, 1e-2],         'destroy (factored diagonal)'),
        ('muon',       'YES',  [3e-3, 1e-2, 3e-2, 0.1],          'preserve (msign equivariant) + NOW interpolates'),
        ('shampoo',    'YES',  [3e-2, 0.1, 0.3, 1.0],            'preserve (L,R transform covariantly)'),
    ]
    res = {}
    for kind, eq, lrs, note in configs:
        r = best(kind, lrs)
        res[kind] = r
        print(f"{kind:>12} | {eq:>12} | {r['rec']:>9.4f} | {r['tr']:>9.1e} | {r['er']:>8.2f} | "
              f"{r['bal']:>9.2e} | {r['lr']:>6g} | {note}", flush=True)
    print("-" * 110, flush=True)
    gd, ad = res['gd']['er'], res['adam']['er']
    mid = (gd + ad) / 2 if np.isfinite(gd) and np.isfinite(ad) else 9.0
    print(f"\nsplit at eff_rank {mid:.2f} (GD {gd:.2f} / Adam {ad:.2f}):", flush=True)
    hits, total = 0, 0
    for kind, eq, _, _ in configs:
        e = res[kind]['er']
        if not np.isfinite(e):
            print(f"  {kind:>12}: DIVERGED at all lrs", flush=True)
            continue
        got = 'preserve' if e < mid else 'destroy'
        want = 'preserve' if eq == 'YES' else 'destroy'
        ok = 'HIT ' if got == want else 'MISS'
        hits += got == want; total += 1
        print(f"  {kind:>12}: er={e:5.2f} bal={res[kind]['bal']:.2e} -> {got:8s} [{ok}]", flush=True)
    print(f"\nPREDICTION SCORE: {hits}/{total} - equivariance classifies the zoo" if hits == total
          else f"\nPREDICTION SCORE: {hits}/{total} - examine the misses", flush=True)
    print(f"[done in {(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == '__main__':
    main()
