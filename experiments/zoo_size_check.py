"""
Problem-size check for the zoo ordering.

Verifies that the zoo-map ordering (Muon ~ exact < GD < ... << Adam) survives a second size and
rank: 60x60, rank 5, k=60, m = 2 x dof, wd=0, small init, run to interpolation, 3 seeds. The full
size x rank x seed ladder is Appendix C8 (GPU).
"""
from __future__ import annotations
import math, time
import numpy as np
import torch

N, R_STAR, K = 60, 5, 60
DOF = R_STAR * (2 * N - R_STAR)
M = int(2.0 * DOF)
INIT = 1e-3
MAX_STEPS = 20000
TRAIN_TOL = 1e-7
EPS = 1e-8
SEEDS = [42, 123, 456]
DECAYED = {'muon'}


def make_problem(seed):
    g = torch.Generator().manual_seed(seed)
    Us = torch.randn(N, R_STAR, generator=g) / np.sqrt(R_STAR)
    Vs = torch.randn(N, R_STAR, generator=g) / np.sqrt(R_STAR)
    Xs = Us @ Vs.T
    A = torch.randn(M, N * N, generator=g)
    y = A @ Xs.reshape(-1)
    return Xs, A, y


def factors(seed):
    g = torch.Generator().manual_seed(seed + 7)
    U = torch.randn(N, K, generator=g) * INIT
    V = torch.randn(N, K, generator=g) * INIT
    U.requires_grad_(True); V.requires_grad_(True)
    return U, V


def stats(U, V, Xs):
    with torch.no_grad():
        W = U @ V.T
        rec = (W - Xs).norm().item() / Xs.norm().item()
        sv = torch.linalg.svdvals(W)
        p = sv / (sv.sum() + 1e-12)
        er = torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()
    return rec, er


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


def run(kind, seed, lr):
    Xs, A, y = make_problem(seed)
    U, V = factors(seed)
    params = [U, V]
    b1, b2 = 0.9, 0.999
    mom = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    train = 1.0
    for step in range(1, MAX_STEPS + 1):
        for p in params:
            if p.grad is not None:
                p.grad.detach_(); p.grad.zero_()
        W = (U @ V.T).reshape(-1)
        loss = ((A @ W - y) ** 2).mean()
        train = loss.item()
        if not np.isfinite(train) or train > 1e8:
            return dict(rec=float('nan'), er=float('nan'), train=float('nan'))
        loss.backward()
        lr_t = lr * 0.5 * (1 + math.cos(math.pi * step / MAX_STEPS)) if kind in DECAYED else lr
        with torch.no_grad():
            for i, p in enumerate(params):
                G = p.grad
                if kind == 'gd':
                    p.add_(G, alpha=-lr_t)
                elif kind == 'adam':
                    mom[i].mul_(b1).add_(G, alpha=1 - b1)
                    v[i].mul_(b2).addcmul_(G, G, value=1 - b2)
                    mh = mom[i] / (1 - b1 ** step); vh = v[i] / (1 - b2 ** step)
                    p.addcdiv_(mh, vh.sqrt() + EPS, value=-lr_t)
                elif kind == 'muon':
                    mom[i].mul_(b1).add_(G)
                    p.add_(newton_schulz(mom[i]), alpha=-lr_t)
        if step % 200 == 0 and train < TRAIN_TOL:
            break
    rec, er = stats(U, V, Xs)
    return dict(rec=rec, er=er, train=train)


def best(kind, lrs):
    best_row = None
    for lr in lrs:
        per = [run(kind, s, lr) for s in SEEDS]
        rec = np.mean([p['rec'] for p in per]); tr = np.mean([p['train'] for p in per])
        er = np.mean([p['er'] for p in per])
        if not (np.isfinite(rec) and np.isfinite(tr)):
            continue
        score = rec if tr < TRAIN_TOL else rec + 10 + tr
        if best_row is None or score < best_row['score']:
            best_row = dict(score=score, rec=rec, tr=tr, er=er, lr=lr)
    return best_row or dict(rec=float('nan'), tr=float('nan'), er=float('nan'), lr=float('nan'))


def main():
    t0 = time.time()
    print("=" * 96, flush=True)
    print(f"ZOO SIZE CHECK | sensing {N}x{N} r*{R_STAR} k{K} m={M} (2x dof) init={INIT} wd=0 | "
          f"{len(SEEDS)} seeds", flush=True)
    print("Does the zoo ordering (Muon ~exact < GD << Adam) survive a second size/rank?", flush=True)
    print("=" * 96, flush=True)
    for kind, lrs in [('gd', [0.01, 0.03, 0.1]), ('adam', [1e-3, 1e-2, 3e-2]),
                      ('muon', [0.03, 0.1, 0.3])]:
        r = best(kind, lrs)
        print(f"{kind:>8} | rec {r['rec']:>8.4f} | train {r['tr']:>9.1e} | er {r['er']:>6.2f} | "
              f"lr {r['lr']:g}", flush=True)
    print(f"\n[done in {(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == '__main__':
    main()
