"""
FlowAdam-p under the RMS scalar (Section 10 numbers).

Re-derives the FlowAdam-p rows of flowadam_upgrade.py under the gauge-invariant RMS scalar
convention (precond_scalar='rms'; see precond_dial_scalar_check.py), so the flow-beyond-the-dial
comparison is matched against the RMS dial-alone reference. Also re-runs the geomean p=0 row as a
bridge to the legacy numbers.
"""
import functools
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flowadam_upgrade as m
from flowadam import FlowAdam as _F

LRS = [1e-3, 3e-3, 1e-2, 3e-2]
DIAL_RMS_P0 = 0.2010


def show(tag, row, cfg):
    if row is None:
        print(f"{tag:>34} | DIVERGED/none", flush=True)
        return None
    _, rec, tr, per = row
    er = np.mean([x['er'] for x in per])
    st = int(np.mean([x['steps'] for x in per]))
    print(f"{tag:>34} | rec {rec:.4f} (sd {np.std([x['rec'] for x in per]):.4f}) | "
          f"train {tr:.2e} | erank {er:.2f} | steps {st} | lr={cfg[0]:g} "
          f"perseed={np.round([x['rec'] for x in per], 4)}", flush=True)
    return rec, er, st


def main():
    t0 = time.time()
    print("=" * 112, flush=True)
    print("FLOWADAM-p x SCALAR CONVENTION | same protocol as flowadam_upgrade.py "
          "(GN clip c=10, wd=0, interp bar 1e-7, 3 seeds)", flush=True)
    print("=" * 112, flush=True)

    out = {}
    for scalar in ('rms', 'geomean'):
        m.FlowAdam = functools.partial(_F, precond_scalar=scalar)
        for p in (0.5, 0.25, 0.0):
            row, cfg = m.best('flow', LRS, p_pow=p, c=10.0)
            out[(scalar, p)] = show(f"FlowAdam-p={p:g} (GN, {scalar})", row, cfg)

    print("-" * 112, flush=True)
    fa = out.get(('rms', 0.0))
    if fa:
        rec = fa[0]
        print(f"\nSUMMARY: FlowAdam-p=0 (rms) rec {rec:.4f} vs dial-alone (rms) {DIAL_RMS_P0:.4f} "
              f"-> flow adds {(DIAL_RMS_P0 - rec) / DIAL_RMS_P0 * 100:+.1f}%", flush=True)
    print(f"\n[done in {(time.time() - t0) / 60:.1f} min]", flush=True)


if __name__ == '__main__':
    main()
