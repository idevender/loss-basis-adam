"""
Spectral-tail phase diagram (Section 8).

Sweeps the spectral-tail mass of the target to reconcile Muon's two regimes: on exact low-rank
sensing Muon recovers nearly exactly, while on targets with a spectral tail it pumps the tail and
generalizes worst. The target is X = sqrt(1-tau^2) X3 + tau E_perp, where X3 is a rank-3 planted
matrix and E_perp is a random full-rank matrix projected orthogonal to X3 and scaled to ||X3||_F, so
tau^2 is the fraction of target energy in the tail. m = 2 x dof(3) Gaussian measurements of the full
X; wd=0, run to interpolation, small init.

As tau grows, Muon's recovery deteriorates fastest and crosses GD at a phase boundary tau*, while GD
keeps fitting head-first and Adam stays worst everywhere.
"""
from __future__ import annotations
import math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from restoration_probe import N, R_STAR, K, DOF, INIT, stats

M = int(2.0 * DOF)
MAX_STEPS = 20000
TRAIN_TOL = 1e-7
EPS = 1e-8
SEEDS = [42, 123, 456]
TAUS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5]
DECAYED = {'muon'}
GRIDS = dict(gd=[0.01, 0.03], adam=[3e-3, 0.01], muon=[0.03, 0.1], shampoo=[0.03, 0.1])


def make_tail_problem(seed, tau):
    g = torch.Generator().manual_seed(seed)
    Us = torch.randn(N, R_STAR, generator=g) / np.sqrt(R_STAR)
    Vs = torch.randn(N, R_STAR, generator=g) / np.sqrt(R_STAR)
    X3 = Us @ Vs.T
    E = torch.randn(N, N, generator=g)
    Qu, _ = torch.linalg.qr(Us); Qv, _ = torch.linalg.qr(Vs)
    E = E - Qu @ (Qu.T @ E)
    E = E - (E @ Qv) @ Qv.T
    E = E / (E.norm() + 1e-12) * X3.norm()
    X = math.sqrt(1 - tau ** 2) * X3 + tau * E
    A = torch.randn(M, N * N, generator=g)
    y = A @ X.reshape(-1)
    return X, A, y


def factors(seed):
    g = torch.Generator().manual_seed(seed + 7)
    U = torch.randn(N, K, generator=g) * INIT
    V = torch.randn(N, K, generator=g) * INIT
    U.requires_grad_(True); V.requires_grad_(True)
    return U, V


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


def run(kind, seed, tau, lr):
    X, A, y = make_tail_problem(seed, tau)
    U, V = factors(seed)
    params = [U, V]
    b1, b2 = 0.9, 0.999
    mom = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    Lp = [torch.zeros(p.shape[0], p.shape[0]) for p in params]
    Rp = [torch.zeros(p.shape[1], p.shape[1]) for p in params]
    Linv = [None, None]; Rinv = [None, None]
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
                elif kind == 'shampoo':
                    Lp[i] += G @ G.T; Rp[i] += G.T @ G
                    if (step - 1) % 20 == 0 or Linv[i] is None:
                        Linv[i] = matrix_pow(Lp[i] + 1.0 * torch.eye(G.shape[0]), -0.25)
                        Rinv[i] = matrix_pow(Rp[i] + 1.0 * torch.eye(G.shape[1]), -0.25)
                    p.add_(Linv[i] @ G @ Rinv[i], alpha=-lr_t)
        if step % 200 == 0 and train < TRAIN_TOL:
            break
    rec, _nuc, er = stats(U, V, X)
    return dict(rec=rec, er=er, train=train)


def best(kind, tau):
    best_row = None
    for lr in GRIDS[kind]:
        per = [run(kind, s, tau, lr) for s in SEEDS]
        rec = np.mean([p['rec'] for p in per]); tr = np.mean([p['train'] for p in per])
        er = np.mean([p['er'] for p in per])
        if not (np.isfinite(rec) and np.isfinite(tr)):
            continue
        score = rec if tr < 1e-4 else rec + 10 + tr
        if best_row is None or score < best_row['score']:
            best_row = dict(score=score, rec=rec, tr=tr, er=er, lr=lr)
    if best_row is None:
        best_row = dict(score=float('nan'), rec=float('nan'), tr=float('nan'), er=float('nan'), lr=float('nan'))
    return best_row


def main():
    t0 = time.time()
    kinds = ['gd', 'adam', 'muon', 'shampoo']
    print("=" * 112, flush=True)
    print(f"MUON PHASE DIAGRAM | sensing {N}x{N} head-rank {R_STAR} + orthogonal tail | m={M} (2x dof3) "
          f"init={INIT} wd=0 | {len(SEEDS)} seeds | tau^2 = tail energy fraction", flush=True)
    print("Prediction: Muon best at tau=0, deteriorates FASTEST with tau (spectral equalization pumps the", flush=True)
    print("tail), crossing GD at some tau* - the phase boundary for 'when to use Muon'. Adam bad throughout.", flush=True)
    print("=" * 112, flush=True)
    print(f"{'tau':>5} |" + "".join(f"  {k:>7}_rec" for k in kinds) + " |" +
          "".join(f"  {k:>6}_er" for k in kinds) + " | muon-vs-gd", flush=True)
    print("-" * 112, flush=True)
    table = {}
    for tau in TAUS:
        row = {k: best(k, tau) for k in kinds}
        table[tau] = row
        mg = (row['gd']['rec'] - row['muon']['rec'])
        verdict = "MUON wins" if mg > 0 else "GD wins"
        print(f"{tau:>5.2f} |" + "".join(f"  {row[k]['rec']:>11.4f}" for k in kinds) + " |" +
              "".join(f"  {row[k]['er']:>9.2f}" for k in kinds) + f" | {verdict}", flush=True)
    print("-" * 112, flush=True)
    prev = None
    tau_star = None
    for tau in TAUS:
        d = table[tau]['muon']['rec'] - table[tau]['gd']['rec']
        if prev is not None and prev < 0 <= d:
            tau_star = tau
        prev = d
    if tau_star is not None:
        print(f"\nPHASE BOUNDARY: Muon loses its edge to GD at tau ~ {tau_star:.2f} "
              f"(tail energy ~ {tau_star**2*100:.0f}%)", flush=True)
    else:
        print("\nNo clean crossing found in this tau range - inspect the table.", flush=True)
    print("train-loss check (all should interpolate):", flush=True)
    for tau in TAUS:
        print(f"  tau={tau:.2f}: " + "  ".join(f"{k}={table[tau][k]['tr']:.1e}" for k in kinds), flush=True)
    print(f"\n[done in {(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == '__main__':
    main()
