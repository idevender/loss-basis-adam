"""Earlier hyperspectral matched-train-loss protocol.

Records the full (train, test, effective rank) trajectory and compares methods at matched
train-loss levels, so held-out differences reflect the interpolant rather than early stopping.
Also measures Muon's spectral-tail behaviour on real data, where effective rank inflates even
though it recovers exactly on exact-low-rank synthetic targets, the regime dependence Section 8
formalizes.

This fixed-lr, two-seed version is kept for protocol comparison. hyperspectral_wilson_v3.py is
the canonical CPU reproduction.
"""
from __future__ import annotations
import math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hyperspectral_completion import load_matrix, make_split, eff_rank

K_MODEL = 48
INIT = 1e-2
MAX_STEPS = 30000
SEEDS = [42, 123]
DENSITIES = [0.15, 0.25]
LEVELS = [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
EPS = 1e-8
LRS = dict(gd=10.0, adam=3e-3, adam_p0rms=1e-2, muon=0.03)


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


def run_traj(kind, x, mask_tr, mask_te, seed, lr):
    g = torch.Generator().manual_seed(seed + 1000)
    U = torch.randn(x.shape[0], K_MODEL, generator=g) * INIT
    V = torch.randn(x.shape[1], K_MODEL, generator=g) * INIT
    U.requires_grad_(True); V.requires_grad_(True)
    params = [U, V]
    n_tr, n_te = mask_tr.sum(), mask_te.sum()
    b1, b2 = 0.9, 0.999
    mom = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    hits = {}
    pending = sorted(LEVELS, reverse=True)
    for step in range(1, MAX_STEPS + 1):
        for p in params:
            if p.grad is not None:
                p.grad.detach_(); p.grad.zero_()
        W = U @ V.T
        loss = (((W - x) * mask_tr) ** 2).sum() / n_tr
        train = loss.item()
        if not np.isfinite(train) or train > 1e6:
            break
        while pending and train <= pending[0]:
            lvl = pending.pop(0)
            with torch.no_grad():
                test = torch.sqrt((((W - x) * mask_te) ** 2).sum() / n_te).item()
                hits[lvl] = (test, eff_rank(W), step)
        if not pending:
            break
        loss.backward()
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
                    p.add_(mom[i] / (1 - b1 ** step), alpha=-lr / s)
            else:
                for i, p in enumerate(params):
                    G = p.grad
                    if kind == 'gd':
                        p.add_(G, alpha=-lr)
                    elif kind == 'adam':
                        mom[i].mul_(b1).add_(G, alpha=1 - b1)
                        v[i].mul_(b2).addcmul_(G, G, value=1 - b2)
                        mh = mom[i] / (1 - b1 ** step); vh = v[i] / (1 - b2 ** step)
                        p.addcdiv_(mh, vh.sqrt() + EPS, value=-lr)
                    elif kind == 'muon':
                        mom[i].mul_(b1).add_(G)
                        p.add_(newton_schulz(mom[i]), alpha=-lr)
    return hits


def main():
    t0 = time.time()
    print("LEGACY FIXED-LEARNING-RATE PROTOCOL: use hyperspectral_wilson_v3.py for paper reproduction.",
          flush=True)
    x = load_matrix(seed=0)
    dof24 = 24 * (x.shape[0] + x.shape[1] - 24)
    print("=" * 118, flush=True)
    print(f"HYPERSPECTRAL WILSON v2 - MATCHED TRAIN LOSS | Indian Pines {tuple(x.shape)} | rank {K_MODEL} "
          f"init {INIT} wd=0 | {len(SEEDS)} seeds | fixed lrs {LRS}", flush=True)
    print("Compare held-out RMSE + eff_rank at matched train loss -> pure interpolant-selection, "
          "no early-stopping confound.", flush=True)
    print("=" * 118, flush=True)
    for dens in DENSITIES:
        m_over_dof = dens * x.numel() / dof24
        print(f"\ndensity {dens:.2f}  (m/dof24 = {m_over_dof:.2f})", flush=True)
        header = f"{'method':>11} |" + "".join(f"   train<={lvl:<8.0e}" for lvl in LEVELS)
        print(header, flush=True)
        print("-" * len(header), flush=True)
        for kind in ['gd', 'adam', 'adam_p0rms', 'muon']:
            allhits = []
            for s in SEEDS:
                sp = make_split(x, dens, s)
                mask_tr = torch.zeros_like(x, dtype=torch.bool)
                mask_tr[sp['tr_r'], sp['tr_c']] = True
                allhits.append(run_traj(kind, x, mask_tr.float(), (~mask_tr).float(), s, LRS[kind]))
            cells_t, cells_r = [], []
            for lvl in LEVELS:
                vals = [h[lvl] for h in allhits if lvl in h]
                if vals:
                    cells_t.append(f"{np.mean([v[0] for v in vals]):>13.5f}")
                    cells_r.append(f"{np.mean([v[1] for v in vals]):>10.1f}rk  ")
                else:
                    cells_t.append(f"{'--':>13}")
                    cells_r.append(f"{'--':>13}")
            print(f"{kind:>11} |" + "".join(cells_t) + "   <- test RMSE", flush=True)
            print(f"{'':>11} |" + "".join(cells_r) + "   <- eff_rank", flush=True)
        print("-" * len(header), flush=True)
    print("\nREAD: at each matched train level, does Adam sit above GD in test with inflated rank?", flush=True)
    print("And does Muon's rank blow past everyone as it descends (tail equalization on real spectra)?", flush=True)
    print(f"[done in {(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == '__main__':
    main()
