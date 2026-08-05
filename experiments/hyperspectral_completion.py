"""Earlier hyperspectral completion loader and CPU smoke test.

Loads the Indian Pines cube (rows = pixels, columns = bands), removes band means, masks entries
and runs dense low-rank-residual completion. Supplies the load_matrix and make_split helpers the
matched-loss demos reuse. Arms: Adam with optimizer weight decay, Adam with the same L2 in the
loss, softened Adam without flow, FlowAdam, and FlowAdam with the softened preconditioner.

Selects on held-out RMSE at density 0.50, so it is a loader and smoke test rather than a source
of paper results. hyperspectral_wilson_v3.py is the canonical CPU reproduction.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowadam import FlowAdam


MAT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "hyperspectral", "Indian_pines_corrected.mat")
ROWS = 2000
RANK = 24
DENSITY = 0.50
STEPS = 300
SEEDS = [42, 123]
LR = 1e-2
WDS = [0.0, 1e-6, 1e-5]
FLOW = dict(switch_sensitivity=0.90, curvature_sensitivity=0.1, ode_t_scale=0.5)


class MF(nn.Module):
    def __init__(self, n, d, rank, init=0.03):
        super().__init__()
        self.U = nn.Parameter(torch.randn(n, rank) * init)
        self.V = nn.Parameter(torch.randn(d, rank) * init)

    def forward(self, rows, cols):
        return (self.U[rows] * self.V[cols]).sum(dim=1)

    def full(self):
        return self.U @ self.V.T


def load_matrix(seed=0):
    mat = loadmat(MAT_PATH)
    key = "indian_pines_corrected"
    cube = torch.tensor(mat[key], dtype=torch.float32)
    x = cube.reshape(-1, cube.shape[-1])
    x = x / (x.max() + 1e-12)
    x = x[x.norm(dim=1) > 1e-8]
    g = torch.Generator().manual_seed(seed + 2026)
    idx = torch.randperm(x.shape[0], generator=g)[:ROWS]
    x = x[idx].contiguous()
    x = x - x.mean(dim=0, keepdim=True)
    return x


def spectral_tail(x, ranks=(8, 16, 24, 48)):
    s = torch.linalg.svdvals(x)
    fro = torch.sqrt((s * s).sum())
    return [(r, (torch.sqrt((s[r:] * s[r:]).sum()) / (fro + 1e-12)).item()) for r in ranks if r < len(s)]


def make_split(x, density, seed):
    g = torch.Generator().manual_seed(seed + 7)
    mask = torch.rand(x.shape, generator=g) < density
    tr_r, tr_c = torch.where(mask)
    te_r, te_c = torch.where(~mask)
    return dict(tr_r=tr_r, tr_c=tr_c, tr_y=x[tr_r, tr_c],
                te_r=te_r, te_c=te_c, te_y=x[te_r, te_c])


def eff_rank(w):
    s = torch.linalg.svdvals(w)
    p = s / (s.sum() + 1e-12)
    return torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()


def train_one(x, split, seed, method, wd, p_pow=1.0, conservative=False):
    torch.manual_seed(seed + 1000)
    m = MF(x.shape[0], x.shape[1], RANK)
    tr_r, tr_c, tr_y = split["tr_r"], split["tr_c"], split["tr_y"]
    te_r, te_c, te_y = split["te_r"], split["te_c"], split["te_y"]

    if method == "adam_optwd":
        opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=wd)
        use_loss_l2 = False
    elif method == "adam_lossL2":
        opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=0.0)
        use_loss_l2 = True
    elif method == "adam_p":
        opt = FlowAdam(m.parameters(), lr=LR, ode_method="euler",
                       switch_sensitivity=1e-9, curvature_sensitivity=1e9,
                       precond_power=p_pow)
        use_loss_l2 = True
    else:
        flow_kw = dict(switch_sensitivity=0.5, curvature_sensitivity=2.0) if conservative else FLOW
        opt = FlowAdam(m.parameters(), lr=LR, ode_method="euler",
                       clip_mode="globalnorm", clip_norm_c=10.0,
                       precond_power=p_pow, **flow_kw)
        use_loss_l2 = True

    for _ in range(STEPS):
        if method.startswith("adam_") and method != "adam_p":
            opt.zero_grad()
            loss = ((m(tr_r, tr_c) - tr_y) ** 2).mean()
            if use_loss_l2 and wd > 0:
                loss = loss + wd * (m.U.pow(2).sum() + m.V.pow(2).sum())
            loss.backward()
            opt.step()
        else:
            def closure():
                opt.zero_grad()
                loss_inner = ((m(tr_r, tr_c) - tr_y) ** 2).mean()
                if use_loss_l2 and wd > 0:
                    loss_inner = loss_inner + wd * (m.U.pow(2).sum() + m.V.pow(2).sum())
                loss_inner.backward()
                return loss_inner
            opt.step(closure)

    with torch.no_grad():
        train = torch.sqrt(((m(tr_r, tr_c) - tr_y) ** 2).mean()).item()
        test = torch.sqrt(((m(te_r, te_c) - te_y) ** 2).mean()).item()
        er = eff_rank(m.full())
        ode = opt.get_ode_count() if isinstance(opt, FlowAdam) else 0
    return train, test, er, ode


def best(x, method, cfgs):
    best_row = None
    for cfg in cfgs:
        wd, p_pow, conservative = cfg
        rows = []
        for seed in SEEDS:
            split = make_split(x, DENSITY, seed)
            rows.append(train_one(x, split, seed, method, wd, p_pow, conservative))
        tests = np.array([r[1] for r in rows])
        if best_row is None or tests.mean() < best_row["tests"].mean():
            best_row = dict(tests=tests, trains=np.array([r[0] for r in rows]),
                            ers=np.array([r[2] for r in rows]), odes=np.array([r[3] for r in rows]),
                            cfg=cfg)
    return best_row


def main():
    t0 = time.time()
    print("LEGACY EXPLORATORY SMOKE TEST: configuration selection below uses held-out RMSE; "
          "do not use this script for paper numbers.", flush=True)
    x = load_matrix(SEEDS[0])
    zero = torch.sqrt((x * x).mean()).item()
    tails = ", ".join([f"r{r}:{v:.3f}" for r, v in spectral_tail(x)])
    print("=" * 118, flush=True)
    print(f"HYPERSPECTRAL COMPLETION | Indian Pines pixels x bands {tuple(x.shape)} | rank={RANK} density={DENSITY} "
          f"| zeroRMSE={zero:.4f} | SVD tail {tails}", flush=True)
    print("=" * 118, flush=True)
    grids = {
        "adam_optwd": [(wd, 1.0, False) for wd in WDS],
        "adam_lossL2": [(wd, 1.0, False) for wd in WDS],
        "adam_p": [(wd, 0.0, False) for wd in [0.0, 1e-6, 1e-5]],
        "flow": [(wd, 1.0, False) for wd in [0.0, 1e-6, 1e-5]],
        "flow_p": [(wd, 0.0, False) for wd in [0.0, 1e-6, 1e-5]],
        "flow_p_cons": [(wd, 0.0, True) for wd in [0.0, 1e-6, 1e-5]],
    }
    print(f"{'method':>14} | {'test RMSE':>16} | {'train':>8} | {'effR':>6} | {'vs optwd':>8} | "
          f"{'vs same':>8} | {'nW':>4} | cfg/trigger", flush=True)
    print("-" * 118, flush=True)
    results = {}
    for method in grids:
        results[method] = best(x, method, grids[method])
        r = results[method]
        print(f"  finished {method}: test={r['tests'].mean():.5f} cfg={r['cfg']} ode={r['odes'].mean():.0f}",
              flush=True)

    base = results["adam_optwd"]["tests"]
    same = results["adam_lossL2"]["tests"]
    for method, r in results.items():
        te = r["tests"]
        vs_base = (base.mean() - te.mean()) / base.mean() * 100
        vs_same = (same.mean() - te.mean()) / same.mean() * 100
        nW = int((te < base).sum())
        print(f"{method:>14} | {te.mean():.5f} +- {te.std():.5f} | {r['trains'].mean():>8.5f} | "
              f"{r['ers'].mean():>6.1f} | {vs_base:>+7.1f}% | {vs_same:>+7.1f}% | "
              f"{nW}/{len(SEEDS)} | {r['cfg']} ode={r['odes'].mean():.0f}", flush=True)
    print(f"[done in {(time.time() - t0) / 60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
