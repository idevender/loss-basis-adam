"""
Initialization-scale robustness of the zoo split (Appendix C3).

The low-rank implicit bias is an init-to-zero phenomenon, and the zoo ran at init=1e-3. This runs
the five main methods at init in {1e-3, 3e-3, 1e-2} under the same protocol. Absolute recovery
degrades as init grows, but the equivariant vs coordinate-wise split is expected to persist at each
init.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import optimizer_zoo_bias as zoo
import restoration_probe as rp

SEEDS = zoo.SEEDS
INITS = [1e-3, 3e-3, 1e-2]
GRIDS = {
    'gd':         [0.01, 0.03, 0.1],
    'adam':       [1e-3, 1e-2, 3e-2],
    'adam_p0rms': [1e-3, 3e-3, 1e-2],
    'muon':       [1e-2, 3e-2, 0.1],
    'shampoo':    [3e-2, 0.1, 0.3],
}
DECAY = {'muon'}


def best(kind, init):
    old_init, old_dec = rp.INIT, zoo.DECAYED
    rp.INIT = init
    zoo.DECAYED = DECAY & {kind}
    try:
        best_row = None
        for lr in GRIDS[kind]:
            per = [zoo.run(kind, s, lr) for s in SEEDS]
            rec = np.mean([p['rec'] for p in per]); tr = np.mean([p['train'] for p in per])
            er = np.mean([p['er'] for p in per])
            if not (np.isfinite(rec) and np.isfinite(tr)):
                continue
            score = rec if tr < zoo.TRAIN_TOL else rec + 10 + tr
            if best_row is None or score < best_row['score']:
                best_row = dict(score=score, rec=rec, tr=tr, er=er, lr=lr)
        return best_row or dict(rec=np.nan, tr=np.nan, er=np.nan, lr=np.nan)
    finally:
        rp.INIT, zoo.DECAYED = old_init, old_dec


def main():
    t0 = time.time()
    print("=" * 100, flush=True)
    print(f"ZOO INIT-SCALE ROBUSTNESS | sensing {zoo.N}x{zoo.N} r*{zoo.R_STAR} m={zoo.M} wd=0 | "
          f"{len(SEEDS)} seeds | INIT sweep {INITS} (zoo used 1e-3)", flush=True)
    print("Read: split (gd/p0rms/muon/shampoo low rec vs adam high rec) must persist at every init.", flush=True)
    print("=" * 100, flush=True)
    print(f"{'init':>8} |" + "".join(f" {k:>13}" for k in GRIDS), flush=True)
    print("-" * 100, flush=True)
    for init in INITS:
        row = {k: best(k, init) for k in GRIDS}
        print(f"{init:>8g} |" + "".join(f" {row[k]['rec']:>13.4f}" for k in GRIDS) + "   <- recovery", flush=True)
        print(f"{'':>8} |" + "".join(f" {row[k]['er']:>13.2f}" for k in GRIDS) + "   <- eff_rank", flush=True)
        print(f"{'':>8} |" + "".join(f" {row[k]['tr']:>13.1e}" for k in GRIDS) + "   <- train", flush=True)
        eq = [row[k]['rec'] for k in ('gd', 'adam_p0rms', 'muon', 'shampoo') if np.isfinite(row[k]['rec'])]
        ad = row['adam']['rec']
        ok = np.isfinite(ad) and eq and max(eq) < ad
        print(f"{'':>8} | split: max(equivariant rec)={max(eq) if eq else np.nan:.4f} vs adam={ad:.4f} -> "
              f"{'HOLDS' if ok else 'BROKEN - inspect'}", flush=True)
    print(f"\n[done in {(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == '__main__':
    main()
