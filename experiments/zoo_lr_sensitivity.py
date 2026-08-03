"""
Learning-rate sensitivity of the zoo split (Section 5.1, Appendix C2).

Prints recovery, effective rank, and train loss at every learning rate on a dense grid for the five
main methods. Tests the selection-rule-free statement: in the gradient-flow limit (smallest
interpolating lr) the equivariant methods recover while Adam does not, at any lr.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import optimizer_zoo_bias as zoo

SEEDS = zoo.SEEDS
GRIDS = {
    'gd':         [3e-3, 0.01, 0.03, 0.1],
    'adam':       [3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    'adam_p0rms': [3e-4, 1e-3, 3e-3, 1e-2],
    'muon':       [3e-3, 1e-2, 3e-2, 0.1],
    'shampoo':    [1e-2, 3e-2, 0.1, 0.3],
}
DECAY = {'muon'}


def main():
    t0 = time.time()
    print("=" * 100, flush=True)
    print(f"ZOO LR-SENSITIVITY | sensing {zoo.N}x{zoo.N} r*{zoo.R_STAR} m={zoo.M} init={zoo.INIT} wd=0 | "
          f"{len(SEEDS)} seeds | full lr curves (muon decayed as in zoo)", flush=True)
    print("Read: does Adam destroy at EVERY lr? Do equivariant methods recover at their SMALLEST "
          "interpolating lr (flow limit)?", flush=True)
    print("=" * 100, flush=True)
    old_dec = zoo.DECAYED
    for kind, lrs in GRIDS.items():
        zoo.DECAYED = DECAY & {kind}
        print(f"\n{kind} {'(decay)' if kind in DECAY else '(const)'}:", flush=True)
        min_interp = None
        for lr in lrs:
            per = [zoo.run(kind, s, lr) for s in SEEDS]
            rec = np.mean([p['rec'] for p in per]); tr = np.mean([p['train'] for p in per])
            er = np.mean([p['er'] for p in per]); st = np.mean([p['steps'] for p in per])
            flag = ''
            if np.isfinite(tr) and tr < zoo.TRAIN_TOL and min_interp is None:
                min_interp = (lr, rec, er)
                flag = '   <- SMALLEST INTERPOLATING LR (flow-limit row)'
            print(f"   lr {lr:<7g}: rec {rec:>7.4f}  er {er:>6.2f}  train {tr:>9.1e}  "
                  f"steps {st:>7.0f}{flag}", flush=True)
        if min_interp is None:
            print("   (never interpolated on this grid)", flush=True)
    zoo.DECAYED = old_dec
    print(f"\n[done in {(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == '__main__':
    main()
