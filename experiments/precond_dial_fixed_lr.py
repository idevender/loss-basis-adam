"""Anisotropy dial, fixed-learning-rate arm (Section 7).

precond_dial_scalar_check.py re-selects the lr at each p, so its monotonicity is a statement
about a best-achievable envelope. This arm drops the selection: the sweep p in {1, .75, .5,
.25, 0} is re-run at every lr of the same grid, with only p varying inside a row.

Protocol as in the RMS dial: wd=0, interpolation bar train < 1e-7 checked every 200 steps,
30k-step budget, 3 paired seeds, sensing 40x40 rank 3, m = 2 x dof, rms scalar. The update rule
is imported from precond_dial_scalar_check.run_adam_p.
"""

import numpy as np
import torch
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from restoration_probe import N, R_STAR, K, M, DOF, INIT, SEEDS
from precond_dial_scalar_check import run_adam_p, MAX_STEPS, TRAIN_TOL

LRS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
PS = [1.0, 0.75, 0.5, 0.25, 0.0]


def main():
    t0 = time.time()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    print("=" * 112, flush=True)
    print(f"PRECOND DIAL, FIXED-LR ARM (no per-p selection) | sensing {N}x{N} r*{R_STAR} k{K} "
          f"m={M} ({M/DOF:.1f}x dof) init{INIT} | {len(SEEDS)} seeds | wd=0 | rms scalar | "
          f"bar {TRAIN_TOL}", flush=True)
    print("=" * 112, flush=True)
    table = {}
    for lr in LRS:
        for p in PS:
            per = [run_adam_p(s, lr, p, 'rms') for s in SEEDS]
            div = any(x['diverged'] for x in per)
            interp = (not div) and all(x['steps'] < MAX_STEPS for x in per)
            rec = float('nan') if div else np.mean([x['rec'] for x in per])
            sd = float('nan') if div else np.std([x['rec'] for x in per])
            er = float('nan') if div else np.mean([x['er'] for x in per])
            tr = float('nan') if div else np.mean([x['train'] for x in per])
            st = 0 if div else int(np.mean([x['steps'] for x in per]))
            table[(lr, p)] = dict(rec=rec, sd=sd, er=er, tr=tr, steps=st,
                                  div=div, interp=interp)
            tag = 'DIVERGED' if div else ('' if interp else 'no-interp')
            print(f"lr={lr:<7g} p={p:<5g} | rec {rec:>7.4f} sd {sd:.4f} | erank {er:>6.2f} | "
                  f"train {tr:9.2e} | steps {st:>6} | {tag}", flush=True)

    print("-" * 112, flush=True)
    print("\nFIXED-LR MATRIX (recovery; * = interpolates, d = diverged, n = misses bar):",
          flush=True)
    print(f"{'lr':>8} | " + " | ".join(f"p={p:<11g}" for p in PS), flush=True)
    for lr in LRS:
        cells = []
        for p in PS:
            c = table[(lr, p)]
            mark = 'd' if c['div'] else ('*' if c['interp'] else 'n')
            cells.append('   --        ' if c['div'] else f"{c['rec']:.4f}{mark} ({c['er']:.1f})")
        print(f"{lr:>8g} | " + " | ".join(cells), flush=True)

    print("\nVERDICT per fixed lr (monotone restoration in p over interpolating cells):",
          flush=True)
    for lr in LRS:
        cells = [(p, table[(lr, p)]) for p in PS]
        ok = [(p, c) for p, c in cells if c['interp']]
        if len(ok) < len(PS):
            missing = [p for p, c in cells if not c['interp']]
            print(f"  lr={lr:<7g}: sweep incomplete (non-interpolating p: {missing})", flush=True)
            continue
        recs = [c['rec'] for _, c in ok]
        ers = [c['er'] for _, c in ok]
        mono_rec = all(recs[i] >= recs[i + 1] for i in range(len(recs) - 1))
        mono_er = all(ers[i] >= ers[i + 1] for i in range(len(ers) - 1))
        print(f"  lr={lr:<7g}: full sweep interpolates | recovery monotone: {mono_rec} "
              f"({recs[0]:.3f} -> {recs[-1]:.3f}) | erank monotone: {mono_er} "
              f"({ers[0]:.1f} -> {ers[-1]:.1f})", flush=True)
    print(f"\n[done in {(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == '__main__':
    main()
