"""Attention gauge across seeds, gauge draws, and noise scales (Appendix D.5).

Repeats the attention twins over several init seeds and gauge draws, and adds noise-twin
controls at perturbation scales 1e-7 and 1e-5 to bound Adam's chaos sensitivity: for a
structural reading the gauge split has to sit orders of magnitude above the noise split.
Equivariant methods stay at the float floor throughout; Muon's gauge and noise twins track
each other.

Model and protocol are imported from attention_gauge.py; only the seeds vary.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attention_gauge as ag

STEPS = ag.STEPS
CHECKS = ag.CHECKS


def run_pair(kind, lr, init_seed, gauge_seed=None, noise_scale=None, noise_seed=888):
    """One twin pair: gauge twin if gauge_seed given, else noise twin at noise_scale."""
    torch.manual_seed(init_seed)
    base = ag.TinyTransformer()
    m1 = ag.TinyTransformer(); m1.load_state_dict(base.state_dict())
    m2 = ag.TinyTransformer(); m2.load_state_dict(base.state_dict())
    if gauge_seed is not None:
        ag.apply_gauge(m2, gauge_seed)
    else:
        ag.apply_noise(m2, noise_seed, scale=noise_scale)
    (xtr, ytr), (xva, yva) = ag.make_data()
    d0, qk0 = ag.drift(m1, m2, xva)
    t1, t2 = ag.Trainer(m1, kind, lr), ag.Trainer(m2, kind, lr)
    curve = [(0, d0, qk0)]
    for step in range(1, STEPS + 1):
        for m, tr in ((m1, t1), (m2, t2)):
            for p in tr.params:
                if p.grad is not None:
                    p.grad.detach_(); p.grad.zero_()
            loss = F.cross_entropy(m(xtr), ytr)
            loss.backward()
            tr.step()
        if step in CHECKS:
            d, qk = ag.drift(m1, m2, xva)
            curve.append((step, d, qk))
    with torch.no_grad():
        acc1 = (m1(xva).argmax(-1) == yva).float().mean().item()
        acc2 = (m2(xva).argmax(-1) == yva).float().mean().item()
    return curve, acc1, acc2


def show(tag, curve, a1, a2):
    steps = [s for s, _, _ in curve]
    ds = [d for _, d, _ in curve]
    print(f"  {tag:<38} step1 {ds[1]:>8.1e}  step100 {ds[steps.index(100)]:>8.1e}  "
          f"final {ds[-1]:>8.1e}  |dacc| {abs(a1-a2):.4f}", flush=True)
    return ds[1], ds[-1]


def main():
    t0 = time.time()
    print("=" * 112, flush=True)
    print(f"ATTENTION GAUGE MULTI-SEED | mod-{ag.P} | {ag.DEPTH}L {ag.HEADS}H d{ag.D_MODEL} | "
          f"{STEPS} steps full-batch deterministic CPU | drift = rel logit dist (val)", flush=True)
    print("=" * 112, flush=True)

    adam_g_s1, adam_g_fin = [], []
    print("\n[adam GAUGE-twin x (2 init seeds x 3 gauge draws)] - predicted: step-1 structural split, every pair", flush=True)
    for iseed in (42, 123):
        for gseed in (777, 999, 1234):
            curve, a1, a2 = run_pair('adam', 1e-3, iseed, gauge_seed=gseed)
            s1, fin = show(f"adam init{iseed} gauge{gseed}", curve, a1, a2)
            adam_g_s1.append(s1); adam_g_fin.append(fin)

    print("\n[adam NOISE-twin, scales 1e-7 / 1e-5] - bound on chaos sensitivity (same basis)", flush=True)
    adam_n_fin = {}
    for scale in (1e-7, 1e-5):
        curve, a1, a2 = run_pair('adam', 1e-3, 42, noise_scale=scale)
        _, fin = show(f"adam init42 noise {scale:g}", curve, a1, a2)
        adam_n_fin[scale] = fin

    print("\n[sgd + scalar-Adam GAUGE-twins] - equivariant: must stay at float floor", flush=True)
    eq_fin = []
    for kind, lr in (('sgd', 0.5), ('adam_p0rms', 3e-3)):
        for iseed, gseed in ((42, 777), (123, 999)):
            curve, a1, a2 = run_pair(kind, lr, iseed, gauge_seed=gseed)
            _, fin = show(f"{kind} init{iseed} gauge{gseed}", curve, a1, a2)
            eq_fin.append(fin)

    print("\n[muon GAUGE vs NOISE twins] - chaos check: the two curves should track each other", flush=True)
    for iseed, gseed in ((42, 777), (123, 999)):
        cg, a1, a2 = run_pair('muon', 0.02, iseed, gauge_seed=gseed)
        show(f"muon init{iseed} gauge{gseed}", cg, a1, a2)
        cn, b1, b2 = run_pair('muon', 0.02, iseed, noise_scale=1e-7)
        show(f"muon init{iseed} noise 1e-7", cn, b1, b2)
        idx = [i for i, (s, _, _) in enumerate(cg) if s >= 100]
        ratio = np.mean([abs(np.log10((cg[i][1] + 1e-12) / (cn[i][1] + 1e-12))) for i in idx])
        print(f"     muon init{iseed}: mean|log10 gauge/noise| past step100 = {ratio:.2f} "
              f"({'CHAOS confirmed (curves track)' if ratio < 1.0 else 'curves DIVERGE - inspect'})", flush=True)

    print("\n" + "=" * 112, flush=True)
    print("SUMMARY:", flush=True)
    print(f"  adam gauge step-1 split: min {min(adam_g_s1):.1e}  max {max(adam_g_s1):.1e}  (n=6)", flush=True)
    print(f"  adam gauge final drift : min {min(adam_g_fin):.1e}  max {max(adam_g_fin):.1e}", flush=True)
    print(f"  adam noise final drift : 1e-7 -> {adam_n_fin[1e-7]:.1e}   1e-5 -> {adam_n_fin[1e-5]:.1e}", flush=True)
    print(f"  structural ratio (min gauge final / max noise final): "
          f"{min(adam_g_fin) / max(adam_n_fin.values()):.0f}x", flush=True)
    print(f"  equivariant (sgd/p0rms) max final drift: {max(eq_fin):.1e}", flush=True)
    print(f"\n[done in {(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == '__main__':
    main()
