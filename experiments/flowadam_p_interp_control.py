"""
Interpolation control for the FlowAdam-p result (Section 10, Appendix C7).

The FlowAdam-p winner fits to a shallower train loss than the dial-alone baseline, which could be
implicit early-stopping regularization. This runs the selected FlowAdam-p=0 (rms, global-norm clip
c=10, lr=1e-3) configuration to a 120k-step budget, recording (train, recovery, effective-rank) at
checkpoints, so the claim can be made at the strict 1e-7 interpolation bar or its floor reported
honestly.
"""
import sys, os, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowadam import FlowAdam
from restoration_probe import make_problem, factors, stats, SEEDS

MAX_STEPS = 120000
PAPER = dict(switch_sensitivity=0.90, curvature_sensitivity=0.1, ode_t_scale=0.5)
CHECKPOINTS = (1e-4, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7)


def run(seed):
    Xs, A, y = make_problem(seed)
    U, V = factors(seed)
    params = [U, V]
    opt = FlowAdam(params, lr=1e-3, ode_method='euler', clip_mode='globalnorm', clip_norm_c=10.0,
                   precond_power=0.0, precond_scalar='rms', **PAPER)

    def base_loss():
        W = (U @ V.T).reshape(-1)
        return ((A @ W - y) ** 2).mean()

    hit = {}
    t0 = time.time()
    for step in range(MAX_STEPS):
        def closure():
            opt.zero_grad()
            l = base_loss()
            l.backward()
            return l
        opt.step(closure)
        if step % 200 == 0:
            l = base_loss().item()
            for lev in CHECKPOINTS:
                if lev not in hit and l < lev:
                    rec, nuc, er = stats(U, V, Xs)
                    hit[lev] = (step, rec, er)
                    print(f"  seed {seed}: train<{lev:.0e} at step {step}: rec {rec:.4f} er {er:.2f} "
                          f"[{(time.time()-t0)/60:.1f}m]", flush=True)
            if l < 1e-7:
                break
    l = base_loss().item()
    rec, nuc, er = stats(U, V, Xs)
    print(f"  seed {seed}: FINAL train {l:.2e} rec {rec:.4f} er {er:.2f} steps<= {step}", flush=True)
    return hit, l, rec, er


def main():
    print("FlowAdam-p=0 (rms, GN c10, lr 1e-3) extended-budget interpolation control "
          f"({MAX_STEPS} steps, bar 1e-7)", flush=True)
    finals = []
    for s in SEEDS:
        hit, l, rec, er = run(s)
        finals.append((l, rec, er))
    recs = [r for _, r, _ in finals]
    print(f"\nFINAL over {len(SEEDS)} seeds: rec {np.mean(recs):.4f} (sd {np.std(recs):.4f}) "
          f"train {[f'{l:.1e}' for l, _, _ in finals]} er {[round(e, 2) for _, _, e in finals]}",
          flush=True)
    print("Compare: 30k-budget main rec 0.1691 (train ~7e-6); dial-alone (rms) 0.2010 "
          "(train 3.5e-8).", flush=True)


if __name__ == '__main__':
    main()
