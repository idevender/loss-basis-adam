"""
Matrix-sensing restoration testbed (shared).

Defines the wd=0 matrix-sensing task and helpers used across the sensing-task experiments. In the
under-determined recovery regime many factorizations interpolate the measurements; only an optimizer
with a low-rank implicit bias recovers the true X*. Gradient flow on the factored loss reaches the
min-nuclear-norm solution and recovers; Adam's diagonal preconditioner damages that bias and
converges to a higher-rank interpolant.

Setup: X* = U* V*^T (n x n, rank r*), m random Gaussian measurements y_i = <A_i, X*>; overparam
factored model W = U V^T with k >= r* columns, small init, loss mean_i (<A_i, W> - y_i)^2, wd=0.
Every method is run to interpolation with final train loss reported and lr tuned per optimizer; the
metric is ground-truth recovery ||W - X*||_F / ||X*||_F, alongside nuclear norm and effective rank.
"""

import numpy as np
import torch
import torch.nn as nn
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowadam import FlowAdam

N, R_STAR, K = 40, 3, 40
DOF = R_STAR * (2 * N - R_STAR)
M = int(2.0 * DOF)
INIT = 1e-3
MAX_STEPS = 30000
TRAIN_TOL = 1e-7
SEEDS = [42, 123, 456]
PAPER = dict(switch_sensitivity=0.90, curvature_sensitivity=0.1, ode_t_scale=0.5)


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
    U = nn.Parameter(torch.randn(N, K, generator=g) * INIT)
    V = nn.Parameter(torch.randn(N, K, generator=g) * INIT)
    return U, V


def stats(U, V, Xs):
    with torch.no_grad():
        W = U @ V.T
        rec = (W - Xs).norm().item() / Xs.norm().item()
        sv = torch.linalg.svdvals(W)
        nuc = sv.sum().item()
        p = sv / (sv.sum() + 1e-12)
        er = torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()
    return rec, nuc, er


def run(kind, seed, lr, wd=0.0):
    Xs, A, y = make_problem(seed)
    U, V = factors(seed)
    params = [U, V]

    def loss_fn():
        W = (U @ V.T).reshape(-1)
        return ((A @ W - y) ** 2).mean()

    if kind == 'flow':
        opt = FlowAdam(params, lr=lr, ode_method='euler', **PAPER)
        for step in range(MAX_STEPS):
            def closure():
                opt.zero_grad()
                l = loss_fn()
                if wd > 0:
                    l = l + wd * (U.pow(2).sum() + V.pow(2).sum())
                l.backward()
                return l
            opt.step(closure)
            if step % 200 == 0 and loss_fn().item() < TRAIN_TOL:
                break
    else:
        if kind == 'gd':
            opt = torch.optim.SGD(params, lr=lr)
        else:
            opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        for step in range(MAX_STEPS):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()
            if step % 200 == 0 and loss_fn().item() < TRAIN_TOL:
                break

    train = loss_fn().item()
    rec, nuc, er = stats(U, V, Xs)
    return dict(train=train, rec=rec, nuc=nuc, er=er)


def best(kind, lrs, wds=(0.0,)):
    """tune lr (and wd) by RECOVERY error among configs that actually interpolated."""
    best_row, best_cfg = None, None
    rows = {}
    for lr in lrs:
        for wd in wds:
            per = [run(kind, s, lr, wd) for s in SEEDS]
            rec = np.mean([p['rec'] for p in per])
            tr = np.mean([p['train'] for p in per])
            rows[(lr, wd)] = (rec, tr, per)
            interp = tr < 1e-4
            score = rec if interp else rec + 10
            if best_row is None or score < best_row[0]:
                best_row, best_cfg = (score, rec, tr, per), (lr, wd)
    return best_row, best_cfg, rows


def main():
    t0 = time.time()
    print("=" * 100, flush=True)
    print(f"RESTORATION PROBE | matrix sensing {N}x{N} r*{R_STAR} k{K} | dof~{DOF} m={M} ({M/DOF:.1f}x dof) | "
          f"init{INIT} | {len(SEEDS)} seeds | wd=0 unless noted", flush=True)
    print("Thesis: GD recovers, Adam(wd0) fails, FlowAdam restores. Train must interpolate (else under-training).", flush=True)
    print("=" * 100, flush=True)
    print(f"{'method':>14} | {'recovery_err':>12} | {'train_loss':>11} | {'nuc_norm':>9} | {'eff_rank':>8} | cfg", flush=True)
    print("-" * 100, flush=True)

    configs = [
        ('gd',          'GD (flow ref)',  [0.1, 0.3, 1.0, 3.0], (0.0,)),
        ('adam',        'Adam wd=0',      [1e-3, 1e-2, 3e-2],   (0.0,)),
        ('adam',        'Adam wd=TUNED',  [1e-3, 1e-2, 3e-2],   (1e-4, 1e-3, 1e-2)),
        ('flow',        'FlowAdam wd=0',  [1e-3, 1e-2, 3e-2],   (0.0,)),
    ]
    summary = {}
    for kind, label, lrs, wds in configs:
        (score, rec, tr, per), cfg, rows = best(kind, lrs, wds)
        recs = [p['rec'] for p in per]
        print(f"{label:>14} | {rec:>12.4f} | {tr:>11.2e} | "
              f"{np.mean([p['nuc'] for p in per]):>9.3f} | {np.mean([p['er'] for p in per]):>8.2f} | "
              f"lr={cfg[0]:g} wd={cfg[1]:g}  perseed_rec={np.round(recs,4)}", flush=True)
        summary[label] = rec

    print("-" * 100, flush=True)
    gd, a0, aw, fl = summary['GD (flow ref)'], summary['Adam wd=0'], summary['Adam wd=TUNED'], summary['FlowAdam wd=0']
    print(f"\nREAD:", flush=True)
    print(f"  Adam(wd0) damages bias?   recovery {a0:.4f} vs GD {gd:.4f}   ->  {'YES Adam worse' if a0 > 1.5*gd else 'no clear damage'}", flush=True)
    print(f"  FlowAdam restores?        recovery {fl:.4f} vs GD {gd:.4f}   ->  {'YES flow ~ GD' if fl < 1.5*gd else 'NO flow != GD'}", flush=True)
    print(f"  FlowAdam vs Adam(wd0):    {(a0-fl)/a0*100:+.1f}%   (the mechanism win)", flush=True)
    print(f"  FlowAdam vs Adam(TUNED-wd): {(aw-fl)/aw*100:+.1f}%   (the practical bar - must beat tuned wd, not just wd0)", flush=True)
    print(f"\n[done in {(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == '__main__':
    main()
