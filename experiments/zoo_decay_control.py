"""
Schedule-symmetrization control for the zoo split (Appendix C1).

In the zoo, the constant-update-norm methods (Muon, signum, Lion) use cosine decay to reach
interpolation while the others run at constant lr. This control runs every method under both
schedules (plus a full-horizon decay variant) and reports two lr-selection rules (lowest recovery
among interpolating lrs; fewest steps to interpolate). It checks that the equivariant vs
coordinate-wise split survives both schedules and both rules.

Reuses optimizer_zoo_bias.run via module-global override.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import optimizer_zoo_bias as zoo

SEEDS = zoo.SEEDS
# The interpolation bar the paper quotes. Held separately from zoo.TRAIN_TOL because the
# decay-full arm sets that global to -1 to disable early stopping; the bar itself never moves.
INTERP_BAR = zoo.TRAIN_TOL
GRIDS = {
    'gd':         [0.01, 0.03, 0.1, 0.3],
    'adam':       [1e-3, 1e-2, 3e-2],
    'adam_p0rms': [1e-3, 3e-3, 1e-2, 3e-2],
    'rmsprop':    [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    'signum':     [1e-4, 3e-4, 1e-3, 3e-3],
    'lion':       [1e-4, 3e-4, 1e-3, 3e-3],
    'adafactor':  [3e-4, 1e-3, 3e-3, 1e-2],
    'muon':       [3e-3, 1e-2, 3e-2, 0.1],
    'shampoo':    [3e-2, 0.1, 0.3, 1.0],
}
EQUIV = {'gd': 'YES', 'adam_p0rms': 'YES', 'muon': 'YES', 'shampoo': 'YES'}


def run(kind, seed, lr, decay, full_horizon=False, max_steps=20000):
    """optimizer_zoo_bias.run with schedule + horizon overridden via module globals."""
    old_dec, old_tol, old_max = zoo.DECAYED, zoo.TRAIN_TOL, zoo.MAX_STEPS
    zoo.DECAYED = {kind} if decay else set()
    zoo.TRAIN_TOL = -1.0 if full_horizon else old_tol
    zoo.MAX_STEPS = max_steps
    try:
        return zoo.run(kind, seed, lr)
    finally:
        zoo.DECAYED, zoo.TRAIN_TOL, zoo.MAX_STEPS = old_dec, old_tol, old_max


def sweep(kind, mode, max_steps=20000):
    decay = mode in ('decay', 'decay-full')
    full = mode == 'decay-full'
    rows = []
    for lr in GRIDS[kind]:
        per = [run(kind, s, lr, decay, full, max_steps) for s in SEEDS]
        agg = {k: float(np.mean([p[k] for p in per])) for k in ('rec', 'er', 'nuc', 'bal', 'train', 'steps')}
        agg['lr'] = lr
        agg['ok'] = np.isfinite(agg['rec']) and np.isfinite(agg['train'])
        rows.append(agg)
    good = [r for r in rows if r['ok']]
    interp = [r for r in good if r['train'] < INTERP_BAR]
    pool = interp if interp else good
    rec_sel = min(pool, key=lambda r: r['rec']) if pool else None
    spd_sel = (min(interp, key=lambda r: r['steps']) if interp
               else (min(good, key=lambda r: r['train']) if good else None))
    return rec_sel, spd_sel, rows


def fmt(r):
    if r is None:
        return "  DIVERGED at all lrs"
    return (f"rec {r['rec']:.4f}  er {r['er']:5.2f}  train {r['train']:.1e}  "
            f"bal {r['bal']:.2e}  steps {r['steps']:>6.0f}  lr {r['lr']:g}")


def split_score(results, mode, rule):
    """2-cluster check: equivariant methods below midpoint(gd, adam) in recovery?"""
    key = 0 if rule == 'rec' else 1
    sel = {k: results[(k, mode)][key] for k in GRIDS if (k, mode) in results}
    sel = {k: v for k, v in sel.items() if v is not None and np.isfinite(v['rec'])}
    if 'gd' not in sel or 'adam' not in sel:
        return
    mid = 0.5 * (sel['gd']['rec'] + sel['adam']['rec'])
    hits = 0
    for k, v in sel.items():
        want = 'preserve' if k in EQUIV else 'destroy'
        got = 'preserve' if v['rec'] < mid else 'destroy'
        hits += got == want
    print(f"    [{mode} / {rule}-sel] split at rec {mid:.3f}: {hits}/{len(sel)} "
          f"{'- class-consistent' if hits == len(sel) else '- inspect mismatches'}", flush=True)


def main():
    t0 = time.time()
    print("=" * 112, flush=True)
    print(f"ZOO DECAY CONTROL | sensing {zoo.N}x{zoo.N} r*{zoo.R_STAR} m={zoo.M} init={zoo.INIT} wd=0 | "
          f"{len(SEEDS)} seeds | modes: const / decay(early-stop) / decay-full(no early stop)", flush=True)
    print("Reads: (1) Adam+decay still destroys? (2) GD+decay-full -> Muon-exact? (3) split survives "
          "both schedules x both selection rules?", flush=True)
    print("=" * 112, flush=True)
    results = {}
    all_kinds = list(GRIDS)
    plan = [(k, 'const') for k in all_kinds] + [(k, 'decay') for k in all_kinds] + \
           [(k, 'decay-full') for k in ('gd', 'adam', 'adam_p0rms', 'muon', 'shampoo')]
    for kind, mode in plan:
        rec_sel, spd_sel, rows = sweep(kind, mode)
        results[(kind, mode)] = (rec_sel, spd_sel)
        eq = EQUIV.get(kind, 'no')
        print(f"\n{kind:>11} [{mode:>10}] (equivariant: {eq})", flush=True)
        print(f"     rec-sel: {fmt(rec_sel)}", flush=True)
        print(f"     spd-sel: {fmt(spd_sel)}", flush=True)
        if rec_sel is not None and rec_sel['train'] >= INTERP_BAR:
            print(f"     NOTE: best row did NOT interpolate (train >= {INTERP_BAR:g}) - floor, not bias.",
                  flush=True)
    print(f"\n{'rmsprop':>11} [ const-60k ] extended budget (was train 2.9e-4 at 20k):", flush=True)
    rec_sel, spd_sel, _ = sweep('rmsprop', 'const', max_steps=60000)
    results[('rmsprop', 'const-60k')] = (rec_sel, spd_sel)
    print(f"     rec-sel: {fmt(rec_sel)}", flush=True)
    print(f"     spd-sel: {fmt(spd_sel)}", flush=True)

    print("\n" + "=" * 112, flush=True)
    print("CLUSTER-SPLIT SCORES:", flush=True)
    for mode in ('const', 'decay'):
        for rule in ('rec', 'spd'):
            split_score(results, mode, rule)
    print("\nSUMMARY DELTAS:", flush=True)
    for a, b, msg in [
        (('adam', 'const'), ('adam', 'decay'), "Adam const -> decay"),
        (('adam', 'const'), ('adam', 'decay-full'), "Adam const -> decay-full"),
        (('gd', 'const'), ('gd', 'decay'), "GD   const -> decay"),
        (('gd', 'const'), ('gd', 'decay-full'), "GD   const -> decay-full"),
        (('muon', 'decay'), ('gd', 'decay-full'), "Muon(decay) vs GD(decay-full)"),
        (('muon', 'const'), ('muon', 'decay'), "Muon const(floor) -> decay"),
    ]:
        ra, rb = results.get(a, (None,))[0], results.get(b, (None,))[0]
        if ra and rb:
            print(f"  {msg:>32}: rec {ra['rec']:.4f} -> {rb['rec']:.4f}   er {ra['er']:.2f} -> {rb['er']:.2f}", flush=True)
    print(f"\n[done in {(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == '__main__':
    main()
