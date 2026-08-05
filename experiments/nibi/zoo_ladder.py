"""Optimizer-zoo replication ladder (Appendix D.9).

Runs the zoo across problem sizes, ranks and m/dof, 10 seeds per cell. Per (n, rank, m/dof) and
optimizer: phase 1 sweeps the lr grid over 3 selection seeds under the paper schedule, recording
the full recovery-vs-lr curves; phase 2 runs 7 extension seeds at the selected lr for the
10-seed table; phase 3 (--cosine-all) re-runs every seed under the uniform cosine schedule, the
Appendix D.1 control.

Per-size lr grids scale by family: GD ~ (40/n); Muon, Shampoo, ScaledGD ~ sqrt(40/n); Adam and
the sign family ~ sqrt(n/40). At n=40 every grid reduces to the CPU paper grid.
"""
from __future__ import annotations

import argparse
import math

import torch

from common import (ZOO9, Sink, pick_device, dtype_of, run_sensing, load_cells,
                    select_lr)

BASE_GRIDS = {
    'gd':          [1e-3, 3e-3, 1e-2, 3e-2, 0.1],
    'adam':        [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    'adam_p0rms':  [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    'rmsprop':     [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    'signum':      [1e-4, 3e-4, 1e-3, 3e-3],
    'lion':        [1e-4, 3e-4, 1e-3, 3e-3],
    'adafactor':   [3e-4, 1e-3, 3e-3, 1e-2],
    'muon':        [3e-3, 1e-2, 3e-2, 0.1, 0.3],
    'shampoo':     [1e-2, 3e-2, 0.1, 0.3],
    'scaledgd':    [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
}

SEL_SEEDS = [42, 123, 456]
EXT_SEEDS = [789, 1011, 1213, 1415, 1617, 1819, 2021]


def scale_factor(kind, n):
    if kind == 'gd':
        return 40.0 / n
    if kind in ('muon', 'shampoo', 'scaledgd'):
        return math.sqrt(40.0 / n)
    return math.sqrt(n / 40.0)


def grid_for(kind, n):
    f = scale_factor(kind, n)
    return [round(lr * f, 12) for lr in BASE_GRIDS[kind]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--rank', type=int, default=3)
    ap.add_argument('--mdof', type=float, default=2.0)
    ap.add_argument('--init', type=float, default=1e-3)
    ap.add_argument('--max-steps', type=int, default=40000)
    ap.add_argument('--opts', default=','.join(ZOO9))
    ap.add_argument('--seeds', type=int, default=10, help='total seeds (3 selection + rest)')
    ap.add_argument('--cosine-all', action='store_true')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--dtype', default='float64')
    ap.add_argument('--arm', default='ladder', help='tag: ladder | rank | mdof')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = dtype_of(args.dtype)
    opts = args.opts.split(',')
    n_ext = max(0, args.seeds - len(SEL_SEEDS))
    ext = EXT_SEEDS[:n_ext]

    meta = dict(exp='zoo_ladder', arm=args.arm, n=args.n, rank=args.rank,
                mdof=args.mdof, init=args.init, max_steps=args.max_steps,
                dtype=args.dtype, grids={k: grid_for(k, args.n) for k in opts})
    sink = Sink(args.out, meta)
    print(f"[zoo_ladder] n={args.n} r={args.rank} mdof={args.mdof} opts={opts} "
          f"device={device} dtype={args.dtype}", flush=True)

    def cell(kind, lr, seed, schedule):
        key = f'{args.arm}|n{args.n}|r{args.rank}|m{args.mdof}|{kind}|lr{lr:g}|s{seed}|{schedule}'
        if sink.has(key):
            return
        res = run_sensing(kind, seed, args.n, args.rank, args.mdof, lr,
                          args.max_steps, init=args.init, schedule=schedule,
                          device=device, dtype=dtype)
        sink.add(key, phase='grid' if schedule == 'protocol' and seed in SEL_SEEDS else 'ext',
                 opt=kind, lr=lr, seed=seed, schedule=schedule, n=args.n,
                 rank=args.rank, mdof=args.mdof, **res)
        print(f"  {key}: rec={res['rec']:.4f} er={res['er']:.2f} train={res['train']:.1e} "
              f"steps={res['steps']} [{res['status']}] {res['secs']}s", flush=True)

    for kind in opts:
        grid = grid_for(kind, args.n)
        for lr in grid:
            for seed in SEL_SEEDS:
                cell(kind, lr, seed, 'protocol')
        cells = [c for c in load_cells(args.out)
                 if c.get('opt') == kind and c.get('schedule') == 'protocol'
                 and c.get('seed') in SEL_SEEDS and c.get('n') == args.n
                 and c.get('rank') == args.rank and abs(c.get('mdof', -1) - args.mdof) < 1e-9]
        lr_star = select_lr(cells)
        if lr_star is None:
            print(f"  !! {kind}: no usable lr on grid", flush=True)
            continue
        edge = 'EDGE' if lr_star in (min(grid), max(grid)) else 'ok'
        sink.add(f'{args.arm}|n{args.n}|r{args.rank}|m{args.mdof}|{kind}|__select__',
                 kind_rec='select', opt=kind, lr=lr_star, edge=edge, n=args.n,
                 rank=args.rank, mdof=args.mdof)
        print(f"  -> {kind}: selected lr={lr_star:g} ({edge})", flush=True)
        for seed in ext:
            cell(kind, lr_star, seed, 'protocol')
        if args.cosine_all:
            for seed in SEL_SEEDS + ext:
                cell(kind, lr_star, seed, 'cosine-all')
    print("[zoo_ladder] DONE", flush=True)


if __name__ == '__main__':
    main()
