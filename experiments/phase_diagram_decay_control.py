"""Schedule control for the phase diagram (Appendix D.1).

In muon_phase_diagram.py only Muon gets cosine decay; GD, Adam and Shampoo run at constant lr.
This re-runs the tail sweep with decay given to every method as a second row, plus Muon without
decay, and checks whether the Muon-vs-GD crossing tau* moves and whether Adam stays worst at
every tau.

Reuses muon_phase_diagram.run via module-global override.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import muon_phase_diagram as mpd

SEEDS = mpd.SEEDS
TAUS = mpd.TAUS
GRIDS = {
    ('gd', False):      [0.01, 0.03],
    ('gd', True):       [0.01, 0.03, 0.1],
    ('adam', False):    [3e-3, 0.01],
    ('adam', True):     [3e-3, 0.01, 0.03],
    ('muon', True):     [0.03, 0.1],
    ('muon', False):    [0.03, 0.1],
    ('shampoo', False): [0.03, 0.1],
    ('shampoo', True):  [0.03, 0.1, 0.3],
}
CONFIGS = [('gd', False), ('gd', True), ('adam', False), ('adam', True),
           ('muon', True), ('muon', False), ('shampoo', False), ('shampoo', True)]


def run(kind, seed, tau, lr, decay):
    old = mpd.DECAYED
    mpd.DECAYED = {kind} if decay else set()
    try:
        return mpd.run(kind, seed, tau, lr)
    finally:
        mpd.DECAYED = old


def best(kind, tau, decay):
    best_row = None
    for lr in GRIDS[(kind, decay)]:
        per = [run(kind, s, tau, lr, decay) for s in SEEDS]
        rec = np.mean([p['rec'] for p in per]); tr = np.mean([p['train'] for p in per])
        er = np.mean([p['er'] for p in per])
        recs = [p['rec'] for p in per]
        if not (np.isfinite(rec) and np.isfinite(tr)):
            continue
        score = rec if tr < mpd.TRAIN_TOL else rec + 10 + tr
        if best_row is None or score < best_row['score']:
            best_row = dict(score=score, rec=rec, tr=tr, er=er, lr=lr,
                            rec_std=float(np.std(recs)))
    if best_row is None:
        best_row = dict(score=np.nan, rec=np.nan, tr=np.nan, er=np.nan, lr=np.nan, rec_std=np.nan)
    return best_row


def crossing(table, a, b):
    """first tau where rec(a) >= rec(b) after being below (a loses its edge to b)."""
    prev = None
    for tau in TAUS:
        d = table[tau][a]['rec'] - table[tau][b]['rec']
        if prev is not None and prev < 0 <= d:
            return tau
        prev = d
    return None


def main():
    t0 = time.time()
    labels = {(k, d): f"{k}{'+dec' if d else ''}" for k, d in CONFIGS}
    print("=" * 130, flush=True)
    print(f"PHASE-DIAGRAM DECAY CONTROL | sensing {mpd.N}x{mpd.N} head r*{mpd.R_STAR} + orth tail | "
          f"m={mpd.M} wd=0 init={mpd.INIT} | {len(SEEDS)} seeds | tau^2 = tail energy", flush=True)
    print("Every method now has a +dec (cosine) row. Reads: does tau* (muon-vs-gd) move under gd+dec? "
          "Adam still worst at every tau?", flush=True)
    print("=" * 130, flush=True)
    hdr = f"{'tau':>5} |" + "".join(f" {labels[c]:>12}" for c in CONFIGS)
    print(hdr + "   (recovery; +-std in small print below)", flush=True)
    print("-" * len(hdr), flush=True)
    table = {}
    for tau in TAUS:
        row = {c: best(c[0], tau, c[1]) for c in CONFIGS}
        table[tau] = row
        print(f"{tau:>5.2f} |" + "".join(f" {row[c]['rec']:>12.4f}" for c in CONFIGS), flush=True)
        print(f"{'':>5} |" + "".join(f" {'+-' + format(row[c]['rec_std'], '.3f'):>12}" for c in CONFIGS), flush=True)
        print(f"{'er':>5} |" + "".join(f" {row[c]['er']:>12.2f}" for c in CONFIGS), flush=True)
    print("-" * len(hdr), flush=True)
    print("\nCROSSINGS (tau where Muon(+dec) loses its edge):", flush=True)
    for gd_cfg in (('gd', False), ('gd', True)):
        ts = crossing(table, ('muon', True), gd_cfg)
        print(f"  muon+dec vs {labels[gd_cfg]:>7}: tau* = {ts if ts is not None else 'none in range'}"
              f"{f'  (tail energy ~{ts**2*100:.0f}%)' if ts is not None else ''}", flush=True)
    print("\nADAM TAX CHECK (is adam/adam+dec worst at every tau?):", flush=True)
    for tau in TAUS:
        row = table[tau]
        recs = {labels[c]: row[c]['rec'] for c in CONFIGS if np.isfinite(row[c]['rec'])}
        worst = max(recs, key=recs.get)
        adam_worst = worst in ('adam', 'adam+dec')
        print(f"  tau={tau:.2f}: worst = {worst:>9} ({recs[worst]:.4f})  "
              f"{'OK (adam family)' if adam_worst else '<-- NOT adam, inspect'}", flush=True)
    print("\ntrain-loss check (interpolation; muon w/o decay expected to floor):", flush=True)
    for tau in TAUS:
        print(f"  tau={tau:.2f}: " + "  ".join(f"{labels[c]}={table[tau][c]['tr']:.1e}" for c in CONFIGS), flush=True)
    print(f"\n[done in {(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == '__main__':
    main()
