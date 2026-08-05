"""Post-ladder diagnostics (Appendix D.9).

A) signum horizon-dependence: n in {40, 128}, cosine horizon in {20k, 40k, 80k}, plus a
   constant-lr control.
B) budget arm: GD / adam_p0rms / adam at n in {128, 256}, budget 150k, testing whether the
   equivariant rows' degradation with n is truncation at the 40k cap.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import (Sink, pick_device, dtype_of, make_sensing, factors, Stepper,
                    stats, DECAYED_PROTOCOL)
import math


def run_traj(kind, seed, n, r, mdof, lr, max_steps, horizon, schedule, device, dtype,
             dump_every=500, tol=1e-7):
    X, A, y = make_sensing(seed, n, r, mdof, device=device, dtype=dtype)
    U, V = factors(seed, n, n, 1e-3, device=device, dtype=dtype)
    params = [U, V]
    stp = Stepper(kind, params)
    decayed = schedule == 'cosine'
    traj = []
    train = float('inf')
    t0 = time.time()
    for step in range(1, max_steps + 1):
        for p in params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        W = (U @ V.T).reshape(-1)
        loss = ((A @ W - y) ** 2).mean()
        train = loss.item()
        if not np.isfinite(train) or train > 1e8:
            break
        loss.backward()
        lr_t = lr * 0.5 * (1 + math.cos(math.pi * step / horizon)) if decayed else lr
        stp.step(lr_t, step)
        if step % dump_every == 0 or step == 1:
            st = stats(U, V, X)
            traj.append([step, round(train, 10), round(st['rec'], 5),
                         round(st['er'], 3), round(st['bal'], 4)])
        if step % 200 == 0 and train < tol:
            break
    st = stats(U, V, X)
    return dict(**st, train=train, steps=step, secs=round(time.time() - t0, 1), traj=traj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--dtype', default='float64')
    args = ap.parse_args()
    device = pick_device(args.device)
    dtype = dtype_of(args.dtype)
    sink = Sink(args.out, dict(exp='diag_signum_budget'))

    for n in (40, 128):
        f = math.sqrt(n / 40.0)
        lr = 3e-3 * f
        for horizon in (20000, 40000, 80000):
            for seed in (42, 123, 456):
                key = f'sig|n{n}|H{horizon}|cos|s{seed}'
                if sink.has(key):
                    continue
                r = run_traj('signum', seed, n, 3, 2.0, lr, horizon, horizon, 'cosine',
                             device, dtype)
                sink.add(key, opt='signum', n=n, horizon=horizon, schedule='cosine',
                         lr=lr, seed=seed, **r)
                print(f"{key}: rec={r['rec']:.4f} er={r['er']:.2f} train={r['train']:.1e} "
                      f"steps={r['steps']}", flush=True)
        for seed in (42,):
            key = f'sig|n{n}|const|s{seed}'
            if not sink.has(key):
                r = run_traj('signum', seed, n, 3, 2.0, lr, 40000, 40000, 'const',
                             device, dtype)
                sink.add(key, opt='signum', n=n, horizon=0, schedule='const', lr=lr,
                         seed=seed, **r)
                print(f"{key}: rec={r['rec']:.4f} train={r['train']:.1e}", flush=True)

    sel = {(128, 'gd'): 0.0003125, (128, 'adam_p0rms'): 0.000178885, (128, 'adam'): 0.000178885,
           (256, 'gd'): 0.00015625, (256, 'adam_p0rms'): 0.000252982, (256, 'adam'): 0.000252982}
    for n in (128, 256):
        for kind in ('gd', 'adam_p0rms', 'adam'):
            lr = sel[(n, kind)] / 3.0
            for seed in (42, 123, 456):
                key = f'budget|n{n}|{kind}|s{seed}'
                if sink.has(key):
                    continue
                r = run_traj(kind, seed, n, 3, 2.0, lr, 150000, 150000, 'const',
                             device, dtype, dump_every=2000)
                sink.add(key, opt=kind, n=n, lr=lr, seed=seed, budget=150000, **r)
                print(f"{key}: rec={r['rec']:.4f} er={r['er']:.2f} train={r['train']:.1e} "
                      f"steps={r['steps']}", flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
