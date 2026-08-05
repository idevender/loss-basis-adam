"""Pavia University hyperspectral completion (Section 9, Appendix D.9).

Runs the matched-train-loss protocol on GPU for Pavia University and, in the same harness, Indian
Pines. Selection rules and the tables are applied by collect.py.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
from scipy.io import loadmat

from common import Sink, pick_device, dtype_of, newton_schulz

EPS = 1e-8
LEVELS = [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
GRIDS = dict(gd=[0.3, 1.0, 3.0, 10.0, 30.0],
             adam=[1e-3, 3e-3, 1e-2],
             adam_p0rms=[3e-3, 1e-2, 3e-2],
             muon=[0.01, 0.03, 0.1])
KEYS = dict(paviau='paviaU', indianpines='indian_pines_corrected')


def load_matrix(path, dskey, rows, seed=0):
    mat = loadmat(path)
    cube = torch.tensor(np.asarray(mat[KEYS[dskey]]), dtype=torch.float32)
    x = cube.reshape(-1, cube.shape[-1])
    x = x / (x.max() + 1e-12)
    x = x[x.norm(dim=1) > 1e-8]
    g = torch.Generator().manual_seed(seed + 2026)
    idx = torch.randperm(x.shape[0], generator=g)[:rows]
    x = x[idx].contiguous()
    x = x - x.mean(dim=0, keepdim=True)
    return x


def make_split(x, density, seed):
    g = torch.Generator().manual_seed(seed + 7)
    mask = torch.rand(x.shape, generator=g) < density
    return mask


def eff_rank(w):
    s = torch.linalg.svdvals(w)
    p = s / (s.sum() + 1e-12)
    return torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()


def run_traj(kind, x, mask_tr, mask_te, seed, lr, k_model, init, max_steps):
    g = torch.Generator().manual_seed(seed + 1000)
    device, dtype = x.device, x.dtype
    U = (torch.randn(x.shape[0], k_model, generator=g) * init).to(device, dtype).requires_grad_(True)
    V = (torch.randn(x.shape[1], k_model, generator=g) * init).to(device, dtype).requires_grad_(True)
    params = [U, V]
    n_tr, n_te = mask_tr.sum(), mask_te.sum()
    b1, b2 = 0.9, 0.999
    mom = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    hits = {}
    pending = sorted(LEVELS, reverse=True)
    for step in range(1, max_steps + 1):
        for p in params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
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
                        mh = mom[i] / (1 - b1 ** step)
                        vh = v[i] / (1 - b2 ** step)
                        p.addcdiv_(mh, vh.sqrt() + EPS, value=-lr)
                    elif kind == 'muon':
                        mom[i].mul_(b1).add_(G)
                        p.add_(newton_schulz(mom[i]), alpha=-lr)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['paviau', 'indianpines'], required=True)
    ap.add_argument('--mat', required=True)
    ap.add_argument('--rows', type=int, default=2000)
    ap.add_argument('--k', type=int, default=48)
    ap.add_argument('--init', type=float, default=1e-2)
    ap.add_argument('--densities', required=True, help='csv, e.g. 0.15,0.25')
    ap.add_argument('--seeds', default='42,123,456,789')
    ap.add_argument('--kinds', default='gd,adam,adam_p0rms,muon')
    ap.add_argument('--max-steps', type=int, default=30000)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--dtype', default='float64')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = dtype_of(args.dtype)
    x = load_matrix(args.mat, args.dataset, args.rows).to(device=device, dtype=dtype)
    dof24 = 24 * (x.shape[0] + x.shape[1] - 24)
    dens = [float(d) for d in args.densities.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    kinds = args.kinds.split(',')
    sink = Sink(args.out, dict(exp='pavia', dataset=args.dataset, shape=list(x.shape),
                               k=args.k, init=args.init, densities=dens,
                               mdof24=[round(d * x.numel() / dof24, 3) for d in dens],
                               levels=LEVELS, grids=GRIDS, max_steps=args.max_steps))
    print(f"[pavia] {args.dataset} {tuple(x.shape)} densities={dens} "
          f"m/dof24={[round(d * x.numel() / dof24, 2) for d in dens]} device={device}", flush=True)

    for d in dens:
        for kind in kinds:
            for lr in GRIDS[kind]:
                for seed in seeds:
                    key = f'{args.dataset}|d{d:g}|{kind}|lr{lr:g}|s{seed}'
                    if sink.has(key):
                        continue
                    t0 = time.time()
                    mask = make_split(x, d, seed).to(device)
                    hits = run_traj(kind, x, mask.to(dtype), (~mask).to(dtype), seed, lr,
                                    args.k, args.init, args.max_steps)
                    row = {f'{lvl:g}': list(hits[lvl]) for lvl in hits}
                    sink.add(key, dataset=args.dataset, dens=d, opt=kind, lr=lr, seed=seed,
                             hits=row, secs=round(time.time() - t0, 1))
                    deepest = min(hits.keys()) if hits else None
                    print(f"  {key}: levels={len(hits)} deepest={deepest} "
                          f"({round(time.time() - t0, 1)}s)", flush=True)
    print("[pavia] DONE", flush=True)


if __name__ == '__main__':
    main()
