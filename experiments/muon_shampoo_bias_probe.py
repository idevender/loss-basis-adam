"""Structured-optimizer bias probe: Muon and Shampoo (Section 5).

Whether the orthogonalized and full-matrix preconditioners keep the bias Adam loses. Anchors: GD
and flow reach effective rank ~4, Adam ~14.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from restoration_probe import make_problem, factors, stats, N, R_STAR, K, M, DOF, INIT

MAX_STEPS = 20000
TRAIN_TOL = 1e-7
EPS = 1e-8


def loss_of(U, V, A, y):
    W = (U @ V.T).reshape(-1)
    return ((A @ W - y) ** 2).mean()


def newton_schulz(G, steps=5, eps=1e-7):
    """Muon's quintic orthogonalization: returns ~ U V^T of G's SVD (singular values -> 1)."""
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


def run(kind, seed, lr, beta=0.9):
    Xs, A, y = make_problem(seed)
    U, V = factors(seed)
    params = [U, V]
    bufs = [torch.zeros_like(p) for p in params]
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    L = [torch.zeros(p.shape[0], p.shape[0]) for p in params]
    R = [torch.zeros(p.shape[1], p.shape[1]) for p in params]
    Linv = [None, None]; Rinv = [None, None]
    b1, b2 = 0.9, 0.999
    train = 1.0
    for step in range(1, MAX_STEPS + 1):
        for p in params:
            if p.grad is not None:
                p.grad.detach_(); p.grad.zero_()
        loss = loss_of(U, V, A, y)
        train = loss.item()
        if not np.isfinite(train) or train > 1e8:
            return dict(rec=float('nan'), nuc=float('nan'), er=float('nan'), train=float('nan'), steps=step)
        loss.backward()
        with torch.no_grad():
            for i, p in enumerate(params):
                G = p.grad
                if kind == 'gd':
                    p.add_(G, alpha=-lr)
                elif kind == 'adam':
                    m[i].mul_(b1).add_(G, alpha=1 - b1)
                    v[i].mul_(b2).addcmul_(G, G, value=1 - b2)
                    mh = m[i] / (1 - b1 ** step); vh = v[i] / (1 - b2 ** step)
                    p.addcdiv_(mh, vh.sqrt() + EPS, value=-lr)
                elif kind == 'muon':
                    bufs[i].mul_(beta).add_(G)
                    p.add_(newton_schulz(bufs[i]), alpha=-lr)
                elif kind == 'shampoo':
                    L[i] += G @ G.T; R[i] += G.T @ G
                    if (step - 1) % 20 == 0 or Linv[i] is None:
                        Linv[i] = matrix_pow(L[i] + 1.0 * torch.eye(G.shape[0]), -0.25)
                        Rinv[i] = matrix_pow(R[i] + 1.0 * torch.eye(G.shape[1]), -0.25)
                    p.add_(Linv[i] @ G @ Rinv[i], alpha=-lr)
        if step % 200 == 0 and train < TRAIN_TOL:
            break
    rec, nuc, er = stats(U, V, Xs)
    return dict(rec=rec, nuc=nuc, er=er, train=train, steps=step)


def best(kind, lrs, seeds):
    """Pick lr that interpolates (train < TRAIN_TOL) with lowest recovery; eff_rank there."""
    best_row = None
    for lr in lrs:
        per = [run(kind, s, lr) for s in seeds]
        rec = np.mean([p['rec'] for p in per]); tr = np.mean([p['train'] for p in per])
        er = np.mean([p['er'] for p in per]); nuc = np.mean([p['nuc'] for p in per])
        if not (np.isfinite(rec) and np.isfinite(tr)):
            continue
        score = rec if tr < TRAIN_TOL else rec + 10
        if best_row is None or score < best_row['score']:
            best_row = dict(score=score, rec=rec, tr=tr, er=er, nuc=nuc, lr=lr)
    if best_row is None:
        best_row = dict(score=float('nan'), rec=float('nan'), tr=float('nan'),
                        er=float('nan'), nuc=float('nan'), lr=float('nan'))
    return best_row


def main():
    t0 = time.time()
    seeds = [42, 123, 456]
    print("=" * 92, flush=True)
    print(f"MUON/SHAMPOO BIAS PROBE | sensing {N}x{N} r*{R_STAR} k{K} m={M} ({M/DOF:.1f}x dof) init={INIT} "
          f"wd=0 | {len(seeds)} seeds", flush=True)
    print("Do structured optimizers preserve GD-like low effective rank (~4) or resemble Adam (~14)?", flush=True)
    print("=" * 92, flush=True)
    print(f"{'method':>10} | {'recovery':>9} | {'train':>9} | {'nuc':>7} | {'eff_rank':>8} | {'lr':>6} | read", flush=True)
    print("-" * 92, flush=True)
    configs = [
        ('gd',      [0.01, 0.03, 0.1, 0.3],     'GD/flow reference'),
        ('adam',    [1e-3, 1e-2, 3e-2],         'coordinate-wise baseline'),
        ('muon',    [1e-3, 3e-3, 1e-2, 3e-2],   'orthogonalized matrix update'),
        ('shampoo', [3e-2, 0.1, 0.3, 1.0],      'full-matrix preconditioner'),
    ]
    res = {}
    for kind, lrs, note in configs:
        r = best(kind, lrs, seeds)
        res[kind] = r
        print(f"{kind:>10} | {r['rec']:>9.4f} | {r['tr']:>9.1e} | {r['nuc']:>7.2f} | {r['er']:>8.2f} | "
              f"{r['lr']:>6g} | {note}", flush=True)
    print("-" * 92, flush=True)
    gd, ad = res['gd']['er'], res['adam']['er']
    mid = 9.0 if not (np.isfinite(gd) and np.isfinite(ad)) else (gd + ad) / 2
    def verdict(name):
        e = res[name]['er']
        return f"{name}: er={e:.2f} -> {'closer to GD' if e < mid else 'closer to Adam'}"
    print(f"\nGD eff_rank={gd:.2f}   Adam eff_rank={ad:.2f}   midpoint={mid:.2f}", flush=True)
    print("  " + verdict('muon'), flush=True)
    print("  " + verdict('shampoo'), flush=True)
    print("\nThis probe compares effective-rank outcomes under the shared experimental protocol.", flush=True)
    print(f"\n[done in {(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == '__main__':
    main()
