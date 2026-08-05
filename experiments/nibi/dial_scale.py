"""Anisotropy dial and FlowAdam-p at scale (Appendix D.9).

Adam-p under the RMS convention at p in {1, .75, .5, .25, 0} on the zoo task, n=40 with 10 seeds
and n=128 with 5. With --flowadam it also runs the FlowAdam-p rows (global-norm clip c=10, Euler
flow, precond_scalar='rms') on GPU. lr is tuned per (p, n) from a small grid by the
collector-side rule.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

from common import (Sink, pick_device, dtype_of, run_sensing, make_sensing, factors,
                    stats, TRAIN_TOL)

SEEDS_ALL = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
PAPER = dict(switch_sensitivity=0.90, curvature_sensitivity=0.1, ode_t_scale=0.5)


def run_flowadam(seed, n, r, mdof, lr, max_steps, p_pow, device, dtype, init=1e-3):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flowadam import FlowAdam
    X, A, y = make_sensing(seed, n, r, mdof, device=device, dtype=dtype)
    U, V = factors(seed, n, n, init, device=device, dtype=dtype)
    params = [U, V]
    opt = FlowAdam(params, lr=lr, ode_method='euler', clip_mode='globalnorm',
                   clip_norm_c=10.0, precond_power=p_pow, precond_scalar='rms', **PAPER)

    def loss_fn():
        W = (U @ V.T).reshape(-1)
        return ((A @ W - y) ** 2).mean()

    t0 = time.time()
    train = float('inf')
    step = 0
    for step in range(1, max_steps + 1):
        def closure():
            opt.zero_grad()
            l = loss_fn()
            l.backward()
            return l
        opt.step(closure)
        if step % 200 == 0:
            train = loss_fn().item()
            if not np.isfinite(train) or train > 1e8:
                return dict(rec=float('nan'), nuc=float('nan'), er=float('nan'),
                            bal=float('nan'), train=train, steps=step,
                            secs=time.time() - t0, status='diverged')
            if train < TRAIN_TOL:
                break
    out = stats(U, V, X)
    out.update(train=train, steps=step, secs=round(time.time() - t0, 2),
               status='interp' if train < TRAIN_TOL else ('loose' if train < 1e-4 else 'n/i'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--rank', type=int, default=3)
    ap.add_argument('--mdof', type=float, default=2.0)
    ap.add_argument('--ps', default='1.0,0.75,0.5,0.25,0.0')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--max-steps', type=int, default=30000)
    ap.add_argument('--flowadam', action='store_true')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--dtype', default='float64')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = dtype_of(args.dtype)
    ps = [float(p) for p in args.ps.split(',')]
    seeds = SEEDS_ALL[:args.seeds]
    scale = math.sqrt(args.n / 40.0)
    lrs = [round(b * scale, 12) for b in (3e-4, 1e-3, 3e-3)]

    sink = Sink(args.out, dict(exp='dial_scale', n=args.n, ps=ps, seeds=seeds, lrs=lrs,
                               max_steps=args.max_steps, flowadam=args.flowadam))
    print(f"[dial_scale] n={args.n} ps={ps} lrs={lrs} seeds={len(seeds)} device={device}", flush=True)
    for p in ps:
        for lr in lrs:
            for seed in seeds:
                key = f'n{args.n}|dial|p{p:g}|lr{lr:g}|s{seed}'
                if sink.has(key):
                    continue
                res = run_sensing('adam_p', seed, args.n, args.rank, args.mdof, lr,
                                  args.max_steps, p_pow=p, device=device, dtype=dtype)
                sink.add(key, opt='adam_p', p=p, lr=lr, seed=seed, n=args.n, **res)
                print(f"  {key}: rec={res['rec']:.4f} er={res['er']:.2f} "
                      f"train={res['train']:.1e} steps={res['steps']} [{res['status']}]", flush=True)
    if args.flowadam:
        for p in (0.0,):
            for lr in [round(b * scale, 12) for b in (1e-3, 3e-3)]:
                for seed in seeds:
                    key = f'n{args.n}|flowp|p{p:g}|lr{lr:g}|s{seed}'
                    if sink.has(key):
                        continue
                    res = run_flowadam(seed, args.n, args.rank, args.mdof, lr,
                                       args.max_steps, p, device, dtype)
                    sink.add(key, opt='flowadam_p', p=p, lr=lr, seed=seed, n=args.n, **res)
                    print(f"  {key}: rec={res['rec']:.4f} er={res['er']:.2f} "
                          f"train={res['train']:.1e} [{res['status']}]", flush=True)
    print("[dial_scale] DONE", flush=True)


if __name__ == '__main__':
    main()
