"""Spectral-tail phase diagram at scale (Appendix D.9).

Refines Section 8's diagram with extra points near the crossing, an n=128 arm and a scalar-Adam
row. collect.py applies the selection rule per (method, tau).
"""
from __future__ import annotations

import argparse
import math

from common import Sink, pick_device, dtype_of, run_sensing

TAUS_DEFAULT = "0,0.05,0.1,0.15,0.2,0.25,0.3,0.325,0.35,0.375,0.4,0.45,0.5,0.6"
BASE_GRIDS = {
    'gd':          [0.01, 0.03],
    'adam':        [3e-3, 0.01],
    'muon':        [0.03, 0.1],
    'shampoo':     [0.03, 0.1],
    'adam_p0rms':  [1e-3, 3e-3],
}
SEEDS_ALL = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]


def scale_factor(kind, n):
    if kind == 'gd':
        return 40.0 / n
    if kind in ('muon', 'shampoo'):
        return math.sqrt(40.0 / n)
    return math.sqrt(n / 40.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--rank', type=int, default=3)
    ap.add_argument('--mdof', type=float, default=2.0)
    ap.add_argument('--taus', default=TAUS_DEFAULT)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--opts', default='gd,adam,muon,shampoo,adam_p0rms')
    ap.add_argument('--max-steps', type=int, default=20000)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--dtype', default='float64')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = dtype_of(args.dtype)
    taus = [float(t) for t in args.taus.split(',')]
    seeds = SEEDS_ALL[:args.seeds]
    opts = args.opts.split(',')
    grids = {k: [round(lr * scale_factor(k, args.n), 12) for lr in BASE_GRIDS[k]] for k in opts}

    sink = Sink(args.out, dict(exp='phase_fine', n=args.n, rank=args.rank, mdof=args.mdof,
                               taus=taus, seeds=seeds, grids=grids,
                               max_steps=args.max_steps, dtype=args.dtype))
    print(f"[phase_fine] n={args.n} taus={taus} seeds={len(seeds)} device={device}", flush=True)
    for tau in taus:
        for kind in opts:
            for lr in grids[kind]:
                for seed in seeds:
                    key = f'n{args.n}|tau{tau:g}|{kind}|lr{lr:g}|s{seed}'
                    if sink.has(key):
                        continue
                    res = run_sensing(kind, seed, args.n, args.rank, args.mdof, lr,
                                      args.max_steps, tau=tau, device=device, dtype=dtype)
                    sink.add(key, opt=kind, tau=tau, lr=lr, seed=seed, n=args.n, **res)
                    print(f"  {key}: rec={res['rec']:.4f} er={res['er']:.2f} "
                          f"train={res['train']:.1e} [{res['status']}]", flush=True)
    print("[phase_fine] DONE", flush=True)


if __name__ == '__main__':
    main()
