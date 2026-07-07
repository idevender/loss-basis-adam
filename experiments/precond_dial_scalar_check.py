"""
Anisotropy dial, scalar-convention check (Section 6).

The dial Adam-p uses denom_i = s_i^p * sbar^(1-p) with s_i = sqrt(vhat_i) and sbar a shared scalar.
Two conventions for sbar:
  'rms'     = sqrt(mean(vhat)): a function of ||G||_F only, hence exactly gauge-invariant; the p=0
              endpoint is exactly the zoo's equivariant scalar-Adam.
  'geomean' = exp(mean(log s_i)): the legacy convention, invariant only in the flow limit.
The paper uses 'rms'. This re-derives the full dial table under both conventions to show the
mechanism is insensitive to the choice.

Protocol: wd=0, run to interpolation (train < 1e-7), lr tuned per (p, scalar) by best recovery among
interpolating rates, 3 paired seeds, sensing 40x40 rank 3, m = 2 x dof.
"""

import numpy as np
import torch
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from restoration_probe import make_problem, factors, stats, N, R_STAR, K, M, DOF, INIT, SEEDS

MAX_STEPS = 30000
TRAIN_TOL = 1e-7


def run_adam_p(seed, lr, p, scalar='rms', betas=(0.9, 0.999), eps=1e-8):
    Xs, A, y = make_problem(seed)
    U, V = factors(seed)
    params = [U, V]
    state = [{'m': torch.zeros_like(q), 'v': torch.zeros_like(q)} for q in params]
    b1, b2 = betas

    def base_loss():
        W = (U @ V.T).reshape(-1)
        return ((A @ W - y) ** 2).mean()

    steps_to_interp = MAX_STEPS
    for step in range(1, MAX_STEPS + 1):
        for q in params:
            q.grad = None
        base_loss().backward()
        with torch.no_grad():
            for q, st in zip(params, state):
                g = q.grad
                st['m'].mul_(b1).add_(g, alpha=1 - b1)
                st['v'].mul_(b2).addcmul_(g, g, value=1 - b2)
            mhat_list, vhat_list = [], []
            for st in state:
                mhat_list.append(st['m'] / (1 - b1 ** step))
                vhat_list.append(st['v'] / (1 - b2 ** step))
            if p >= 1.0:
                for q, mh, vh in zip(params, mhat_list, vhat_list):
                    q.add_(mh / (vh.sqrt() + eps), alpha=-lr)
            else:
                all_vh = torch.cat([vh.reshape(-1) for vh in vhat_list])
                if scalar == 'rms':
                    sbar = all_vh.mean().sqrt()
                else:
                    sbar = torch.exp(torch.log(all_vh.sqrt() + eps).mean())
                for q, mh, vh in zip(params, mhat_list, vhat_list):
                    s = vh.sqrt()
                    denom = (s + eps).pow(p) * (sbar + eps).pow(1 - p)
                    q.add_(mh / (denom + eps), alpha=-lr)
        if step % 200 == 0:
            l = base_loss().item()
            if not np.isfinite(l) or l > 1e6:
                break
            if l < TRAIN_TOL:
                steps_to_interp = step
                break

    train = base_loss().item()
    rec, nuc, er = stats(U, V, Xs)
    return dict(train=train, rec=rec, nuc=nuc, er=er, steps=steps_to_interp,
                diverged=(not np.isfinite(train)) or train > 1e6)


def best(p, scalar, lrs):
    best_row, best_cfg = None, None
    for lr in lrs:
        per = [run_adam_p(s, lr, p, scalar) for s in SEEDS]
        if any(x['diverged'] for x in per):
            continue
        rec = np.mean([x['rec'] for x in per])
        tr = np.mean([x['train'] for x in per])
        score = rec if tr < 1e-4 else rec + 10
        if best_row is None or score < best_row[0]:
            best_row, best_cfg = (score, rec, tr, per), lr
    return best_row, best_cfg


def main():
    t0 = time.time()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    print("=" * 112, flush=True)
    print(f"PRECOND DIAL x SCALAR CONVENTION | sensing {N}x{N} r*{R_STAR} k{K} m={M} "
          f"({M/DOF:.1f}x dof) init{INIT} | {len(SEEDS)} seeds | wd=0 | interpolation bar {TRAIN_TOL}",
          flush=True)
    print("=" * 112, flush=True)
    print(f"{'config':>26} | {'recovery':>9} | {'sd':>7} | {'train':>9} | {'erank':>6} | "
          f"{'steps':>6} | cfg", flush=True)
    print("-" * 112, flush=True)

    lrs = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    results = {}
    row, cfg = best(1.0, 'rms', lrs)
    for scalar in ('rms', 'geomean'):
        results[(scalar, 1.0)] = (row, cfg)
    _, rec, tr, per = row
    print(f"{'Adam p=1 (both)':>26} | {rec:>9.4f} | {np.std([x['rec'] for x in per]):>7.4f} | "
          f"{tr:>9.2e} | {np.mean([x['er'] for x in per]):>6.2f} | "
          f"{int(np.mean([x['steps'] for x in per])):>6} | lr={cfg:g} "
          f"perseed={np.round([x['rec'] for x in per], 4)}", flush=True)

    for scalar in ('rms', 'geomean'):
        for p in [0.75, 0.5, 0.25, 0.0]:
            row, cfg = best(p, scalar, lrs)
            if row is None:
                print(f"{scalar + ' p=' + format(p, 'g'):>26} | DIVERGED at all lr", flush=True)
                continue
            results[(scalar, p)] = (row, cfg)
            _, rec, tr, per = row
            print(f"{scalar + ' p=' + format(p, 'g'):>26} | {rec:>9.4f} | "
                  f"{np.std([x['rec'] for x in per]):>7.4f} | {tr:>9.2e} | "
                  f"{np.mean([x['er'] for x in per]):>6.2f} | "
                  f"{int(np.mean([x['steps'] for x in per])):>6} | lr={cfg:g} "
                  f"perseed={np.round([x['rec'] for x in per], 4)}", flush=True)

    print("-" * 112, flush=True)
    print("\nPAPER TABLE (rms convention, exactly equivariant at p=0):", flush=True)
    for p in [1.0, 0.75, 0.5, 0.25, 0.0]:
        if ('rms', p) in results:
            row, cfg = results[('rms', p)]
            _, rec, tr, per = row
            print(f"  p={p:<5g} rec {rec:.4f}  erank {np.mean([x['er'] for x in per]):.1f}  "
                  f"steps {int(np.mean([x['steps'] for x in per]))}", flush=True)
    print("\nROBUSTNESS (geomean convention, legacy):", flush=True)
    for p in [1.0, 0.75, 0.5, 0.25, 0.0]:
        if ('geomean', p) in results:
            row, cfg = results[('geomean', p)]
            _, rec, tr, per = row
            print(f"  p={p:<5g} rec {rec:.4f}  erank {np.mean([x['er'] for x in per]):.1f}  "
                  f"steps {int(np.mean([x['steps'] for x in per]))}", flush=True)
    print(f"\n[done in {(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == '__main__':
    main()
