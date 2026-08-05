"""Indian Pines at matched training loss, with train-only lr selection (Section 9, Appendix D.6).

Hardens the matched-loss protocol against two confounds. Learning rates are picked per method
and density by a train-only rule (the lr reaching the deepest matched train level, ties by
fewest steps, selection seeds {42, 123}), which removes test-selection leakage; and a GD
transparency table reports GD at every grid lr, so the effect is not an edge-of-stability
artifact. Seeds {456, 789} are then added at the selected lr for a 4-seed table. Reuses
hyperspectral_wilson_v2.run_traj; wd=0 throughout.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hyperspectral_wilson_v2 as v2
from hyperspectral_completion import load_matrix, make_split

LEVELS = v2.LEVELS
SEL_SEEDS = [42, 123]
EXTRA_SEEDS = [456, 789]
DENSITIES = [0.15, 0.25]
GRIDS = dict(gd=[1.0, 3.0, 10.0], adam=[1e-3, 3e-3, 1e-2],
             adam_p0rms=[3e-3, 1e-2, 3e-2], muon=[0.01, 0.03, 0.1])


def traj(kind, x, dens, seed, lr, cache={}):
    key = (kind, dens, seed, lr)
    if key not in cache:
        sp = make_split(x, dens, seed)
        mask_tr = torch.zeros_like(x, dtype=torch.bool)
        mask_tr[sp['tr_r'], sp['tr_c']] = True
        cache[key] = v2.run_traj(kind, x, mask_tr.float(), (~mask_tr).float(), seed, lr)
    return cache[key]


def depth_speed(hits_list):
    """(-#levels reached by all seeds, mean steps at the deepest common level) - train-only score."""
    common = [lvl for lvl in LEVELS if all(lvl in h for h in hits_list)]
    if not common:
        return (0, float('inf')), None
    deepest = common[-1]
    return (-len(common), float(np.mean([h[deepest][2] for h in hits_list]))), deepest


def cells(hits_list):
    out_t, out_r = [], []
    for lvl in LEVELS:
        vals = [h[lvl] for h in hits_list if lvl in h]
        n = len(vals)
        if vals and n == len(hits_list):
            out_t.append(f"{np.mean([v[0] for v in vals]):>9.5f}+-{np.std([v[0] for v in vals]):<7.5f}")
            out_r.append(f"{np.mean([v[1] for v in vals]):>9.1f}rk{'':<7}")
        elif vals:
            out_t.append(f"{np.mean([v[0] for v in vals]):>9.5f} ({n}s){'':<3}")
            out_r.append(f"{np.mean([v[1] for v in vals]):>9.1f}rk({n}s){'':<3}")
        else:
            out_t.append(f"{'--':>18}")
            out_r.append(f"{'--':>18}")
    return out_t, out_r


def main():
    t0 = time.time()
    x = load_matrix(seed=0)
    dof24 = 24 * (x.shape[0] + x.shape[1] - 24)
    print("=" * 130, flush=True)
    print(f"HYPERSPECTRAL WILSON v3 | Indian Pines {tuple(x.shape)} rank {v2.K_MODEL} init {v2.INIT} wd=0 | "
          f"train-speed lr selection (seeds {SEL_SEEDS}) + fresh seeds {EXTRA_SEEDS} | grids {GRIDS}", flush=True)
    print("=" * 130, flush=True)
    chosen = {}
    for dens in DENSITIES:
        print(f"\n########## density {dens:.2f}  (m/dof24 = {dens * x.numel() / dof24:.2f}) ##########", flush=True)
        print("\n-- phase A: train-only lr selection (deepest level, then fewest steps; seeds 42/123) --", flush=True)
        for kind in ('gd', 'adam', 'adam_p0rms', 'muon'):
            scored = []
            for lr in GRIDS[kind]:
                hl = [traj(kind, x, dens, s, lr) for s in SEL_SEEDS]
                score, deepest = depth_speed(hl)
                scored.append((score, lr, deepest))
                print(f"   {kind:>11} lr {lr:<6g}: deepest common level "
                      f"{deepest if deepest is not None else '--':>6}  steps {score[1]:>8.0f}  "
                      f"levels {-score[0]}", flush=True)
            scored.sort()
            chosen[(kind, dens)] = scored[0][1]
            print(f"   {kind:>11} SELECTED lr = {scored[0][1]:g}", flush=True)
        print("\n-- phase B: 4-seed matched-train-loss table at the train-selected lr --", flush=True)
        header = f"{'method':>11} lr      |" + "".join(f"   train<={lvl:<10.0e}" for lvl in LEVELS)
        print(header, flush=True)
        print("-" * len(header), flush=True)
        for kind in ('gd', 'adam', 'adam_p0rms', 'muon'):
            lr = chosen[(kind, dens)]
            hl = [traj(kind, x, dens, s, lr) for s in SEL_SEEDS + EXTRA_SEEDS]
            ct, cr = cells(hl)
            print(f"{kind:>11} {lr:<7g}|" + "".join(ct) + "  <- test RMSE", flush=True)
            print(f"{'':>19}|" + "".join(cr) + "  <- eff_rank", flush=True)
        for lvl in reversed(LEVELS):
            g = [traj('gd', x, dens, s, chosen[('gd', dens)]) for s in SEL_SEEDS + EXTRA_SEEDS]
            a = [traj('adam', x, dens, s, chosen[('adam', dens)]) for s in SEL_SEEDS + EXTRA_SEEDS]
            if all(lvl in h for h in g) and all(lvl in h for h in a):
                gm = np.mean([h[lvl][0] for h in g]); am = np.mean([h[lvl][0] for h in a])
                gw = sum(h1[lvl][0] < h2[lvl][0] for h1, h2 in zip(g, a))
                print(f"\n   SUMMARY d={dens}: at matched train<={lvl:g}: GD {gm:.5f} vs Adam {am:.5f} "
                      f"-> GD {'+' if am > gm else ''}{(am - gm) / am * 100:.1f}%  ({gw}/4 seeds)", flush=True)
                break

    print("\n########## C: GD lr-transparency at d=0.15 (EoS check; 4 seeds each lr) ##########", flush=True)
    dens = 0.15
    header = f"{'GD lr':>8} |" + "".join(f"   train<={lvl:<10.0e}" for lvl in LEVELS)
    print(header, flush=True)
    a = [traj('adam', x, dens, s, chosen[('adam', dens)]) for s in SEL_SEEDS + EXTRA_SEEDS]
    for lr in GRIDS['gd']:
        hl = [traj('gd', x, dens, s, lr) for s in SEL_SEEDS + EXTRA_SEEDS]
        ct, _ = cells(hl)
        print(f"{lr:>8g} |" + "".join(ct), flush=True)
        for lvl in reversed(LEVELS):
            if all(lvl in h for h in hl) and all(lvl in h for h in a):
                gm = np.mean([h[lvl][0] for h in hl]); am = np.mean([h[lvl][0] for h in a])
                print(f"{'':>8}   vs Adam at train<={lvl:g}: GD {'+' if am > gm else ''}"
                      f"{(am - gm) / am * 100:.1f}%", flush=True)
                break
    print(f"\n[done in {(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == '__main__':
    main()
