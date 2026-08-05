"""FlowAdam-p: softening the preconditioner while keeping the flow (Section 10).

Pairs the flow with the softened denominator so the injected GD direction survives the following
Adam step. Reports recovery, effective rank and steps to interpolation.
"""

import numpy as np
import torch
import torch.nn as nn
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowadam import FlowAdam
from restoration_probe import make_problem, factors, stats, N, R_STAR, K, M, DOF, INIT, SEEDS

MAX_STEPS = 30000
TRAIN_TOL = 1e-7
PAPER = dict(switch_sensitivity=0.90, curvature_sensitivity=0.1, ode_t_scale=0.5)


def run(kind, seed, lr, wd=0.0, p_pow=1.0, clip_mode='globalnorm', c=10.0):
    Xs, A, y = make_problem(seed)
    U, V = factors(seed)
    params = [U, V]

    def base_loss():
        W = (U @ V.T).reshape(-1)
        return ((A @ W - y) ** 2).mean()

    steps = MAX_STEPS
    if kind == 'adam':
        opt = FlowAdam(params, lr=lr, ode_method='euler', switch_sensitivity=1e-9,
                       curvature_sensitivity=1e9, precond_power=p_pow, **{k: v for k, v in PAPER.items()
                       if k not in ('switch_sensitivity', 'curvature_sensitivity')})
        for step in range(MAX_STEPS):
            def closure():
                opt.zero_grad(); l = base_loss()
                if wd > 0: l = l + wd * (U.pow(2).sum() + V.pow(2).sum())
                l.backward(); return l
            opt.step(closure)
            if step % 100 == 0 and base_loss().item() < TRAIN_TOL:
                steps = step; break
    else:
        opt = FlowAdam(params, lr=lr, ode_method='euler', clip_mode=clip_mode, clip_norm_c=c,
                       precond_power=p_pow, **PAPER)
        for step in range(MAX_STEPS):
            def closure():
                opt.zero_grad(); l = base_loss()
                if wd > 0: l = l + wd * (U.pow(2).sum() + V.pow(2).sum())
                l.backward(); return l
            opt.step(closure)
            if step % 100 == 0 and base_loss().item() < TRAIN_TOL:
                steps = step; break

    tr = base_loss().item()
    rec, nuc, er = stats(U, V, Xs)
    return dict(train=tr, rec=rec, nuc=nuc, er=er, steps=steps,
                diverged=(not np.isfinite(tr)) or tr > 1e6)


def best(kind, lrs, wds=(0.0,), p_pow=1.0, clip_mode='globalnorm', c=10.0):
    best_row, best_cfg = None, None
    for lr in lrs:
        for wd in wds:
            per = [run(kind, s, lr, wd, p_pow, clip_mode, c) for s in SEEDS]
            if any(x['diverged'] for x in per):
                continue
            rec = np.mean([x['rec'] for x in per]); tr = np.mean([x['train'] for x in per])
            score = rec if tr < TRAIN_TOL else rec + 10
            if best_row is None or score < best_row[0]:
                best_row, best_cfg = (score, rec, tr, per), (lr, wd)
    if best_row is None:
        return None, None
    return best_row, best_cfg


def line(label, row, cfg):
    if row is None:
        print(f"{label:>30} | {'DIVERGED/none':>9}", flush=True)
        return None
    _, rec, tr, per = row
    er = np.mean([x['er'] for x in per]); st = int(np.mean([x['steps'] for x in per]))
    print(f"{label:>30} | {rec:>9.4f} | {tr:>9.2e} | {er:>8.2f} | {st:>6} | "
          f"cfg={cfg} perseed={np.round([x['rec'] for x in per],3)}", flush=True)
    return (rec, er, st)


def main():
    t0 = time.time()
    print("=" * 112, flush=True)
    print(f"FLOWADAM UPGRADE | sensing {N}x{N} r*{R_STAR} k{K} m={M} ({M/DOF:.1f}x dof) | {len(SEEDS)} seeds | wd=0 | "
          f"run to interp", flush=True)
    print("Q: does FLOW + softened preconditioner (FlowAdam-p) beat shipped FlowAdam AND Adam-p, at FEWER steps?",
          flush=True)
    print("=" * 112, flush=True)
    print(f"{'method':>30} | {'recovery':>9} | {'train':>9} | {'eff_rank':>8} | {'steps':>6} | cfg", flush=True)
    print("-" * 112, flush=True)

    LRS = [1e-3, 3e-3, 1e-2, 3e-2]
    frontier = {}
    frontier['Adam (p=1)'] = line('Adam (p=1, no flow)', *best('adam', LRS, p_pow=1.0))
    frontier['Adam-p=0 (no flow)'] = line('Adam-p=0 (softened, no flow)', *best('adam', LRS, p_pow=0.0))
    frontier['FlowAdam shipped (pc,p1)'] = line('FlowAdam shipped (percoord,p1)',
                                                *best('flow', LRS, p_pow=1.0, clip_mode='percoord', c=1.0))
    frontier['FlowAdam GN (p=1)'] = line('FlowAdam GN c10 (p=1)', *best('flow', LRS, p_pow=1.0, c=10.0))
    for p in [0.5, 0.25, 0.0]:
        frontier[f'FlowAdam-p={p:g} (GN)'] = line(f'FlowAdam-p={p:g} (GN c10) UPGRADE',
                                                  *best('flow', LRS, p_pow=p, c=10.0))
    line('Adam + TUNED wd (ceiling)', *best('adam', LRS, wds=(1e-4, 1e-3, 1e-2), p_pow=1.0))

    print("-" * 112, flush=True)
    print("\nSPEED x BIAS FRONTIER (recovery, steps) -- lower-left = better:", flush=True)
    for k, v in frontier.items():
        if v is None: continue
        rec, er, st = v
        print(f"   {k:>28}: recovery {rec:.4f}  rank {er:.1f}  steps {st}", flush=True)
    fa_p0 = frontier.get('FlowAdam-p=0 (GN)'); adam_p0 = frontier.get('Adam-p=0 (no flow)')
    if fa_p0 and adam_p0:
        print(f"\n  FlowAdam-p=0 vs Adam-p=0 (no-flow):  recovery {fa_p0[0]:.4f} vs {adam_p0[0]:.4f} "
              f"({(adam_p0[0]-fa_p0[0])/adam_p0[0]*100:+.1f}%)  | steps {fa_p0[2]} vs {adam_p0[2]} "
              f"({'FLOW FASTER' if fa_p0[2] < adam_p0[2] else 'flow not faster'})", flush=True)
    print(f"\n[done in {(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == '__main__':
    main()
