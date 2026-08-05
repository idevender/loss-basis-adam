"""Extended-budget continuation of the one dial cell that never interpolated (Appendix D.9).

At n=40 the FlowAdam-p (p=0) row was selected at lr=1e-3, but every seed hit the 30k-step cap
above the 1e-7 bar. Same seeds, same lr, same settings, longer budget.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import Sink, pick_device, dtype_of, TRAIN_TOL
from dial_scale import run_flowadam, SEEDS_ALL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=300000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device, dtype = pick_device(a.device), dtype_of(a.dtype)
    seeds = SEEDS_ALL[:a.seeds]
    sink = Sink(a.out, dict(exp="dial_flowlong", n=a.n, lr=a.lr, seeds=seeds,
                            max_steps=a.max_steps, bar=TRAIN_TOL))
    print(f"[flowlong] n={a.n} lr={a.lr} seeds={len(seeds)} budget={a.max_steps} "
          f"bar={TRAIN_TOL:g} device={device}", flush=True)
    recs, oks = [], 0
    for seed in seeds:
        key = f"n{a.n}|flowlong|p0|lr{a.lr:g}|s{seed}|m{a.max_steps}"
        if sink.has(key):
            continue
        t0 = time.time()
        res = run_flowadam(seed, a.n, 3, 2.0, a.lr, a.max_steps, 0.0, device, dtype)
        sink.add(key, opt="flowadam_p", p=0.0, lr=a.lr, seed=seed, n=a.n, **res)
        recs.append(res["rec"]); oks += res["status"] == "interp"
        print(f"  s{seed}: rec={res['rec']:.4f} er={res['er']:.2f} train={res['train']:.2e} "
              f"steps={res['steps']} [{res['status']}] {time.time()-t0:.0f}s", flush=True)
    if recs:
        print(f"\n[flowlong] {oks}/{len(recs)} interpolated | rec mean {np.mean(recs):.4f} "
              f"+- {np.std(recs, ddof=1):.4f}", flush=True)
    print("[flowlong] DONE", flush=True)


if __name__ == "__main__":
    main()
