"""
Equivariance and balancedness probe (Section 6).

For W = U V^T the factorization has a gauge symmetry (U, V) -> (UQ, VQ), Q orthogonal, that leaves W
unchanged; gradient flow respects it, coordinate-wise Adam does not because its second-moment
buffers live in the factor coordinate system. Two runs are initialized with the same W0 but
different latent gauge, and the probe measures product drift ||W_base - W_gauge|| / ||W_base||, the
recovery and effective-rank gaps, and the balancedness invariant ||U^T U - V^T V||_F. Equivariant
methods give near-zero drift; Adam gives large drift and worse rank and recovery.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from restoration_probe import make_problem, factors, stats, N, R_STAR, K, M, DOF, INIT, SEEDS

MAX_STEPS = 20000
TRAIN_TOL = 1e-7
EPS = 1e-8


def loss_of(U, V, A, y):
    W = (U @ V.T).reshape(-1)
    return ((A @ W - y) ** 2).mean()


def orthogonal_q(seed):
    g = torch.Generator().manual_seed(seed + 999)
    Z = torch.randn(K, K, generator=g)
    Q, R = torch.linalg.qr(Z)
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1
    return Q * signs


def balancedness(U, V):
    with torch.no_grad():
        num = (U.T @ U - V.T @ V).norm()
        den = U.pow(2).sum() + V.pow(2).sum() + EPS
        return (num / den).item()


def run_gd(U0, V0, A, y, Xs, lr):
    U = nn.Parameter(U0.clone())
    V = nn.Parameter(V0.clone())
    for step in range(1, MAX_STEPS + 1):
        if U.grad is not None:
            U.grad.zero_()
            V.grad.zero_()
        loss = loss_of(U, V, A, y)
        loss.backward()
        with torch.no_grad():
            U.add_(U.grad, alpha=-lr)
            V.add_(V.grad, alpha=-lr)
        if step % 200 == 0 and loss.item() < TRAIN_TOL:
            break
    rec, nuc, er = stats(U, V, Xs)
    return dict(U=U.detach(), V=V.detach(), W=(U @ V.T).detach(),
                train=loss_of(U, V, A, y).item(), rec=rec, nuc=nuc, er=er,
                bal=balancedness(U, V), steps=step)


def run_adam_p(U0, V0, A, y, Xs, lr, p, scalar="geomean", betas=(0.9, 0.999)):
    U = nn.Parameter(U0.clone())
    V = nn.Parameter(V0.clone())
    params = [U, V]
    state = [{"m": torch.zeros_like(U), "v": torch.zeros_like(U)},
             {"m": torch.zeros_like(V), "v": torch.zeros_like(V)}]
    b1, b2 = betas
    loss = None

    for step in range(1, MAX_STEPS + 1):
        for q in params:
            if q.grad is not None:
                q.grad.zero_()
        loss = loss_of(U, V, A, y)
        loss.backward()
        with torch.no_grad():
            for q, st in zip(params, state):
                g = q.grad
                st["m"].mul_(b1).add_(g, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)

            mhat = [st["m"] / (1 - b1 ** step) for st in state]
            s_vals = [(st["v"] / (1 - b2 ** step)).sqrt() for st in state]
            all_s = torch.cat([s.reshape(-1) for s in s_vals])
            if scalar == "geomean":
                scalar_denom = torch.exp(torch.log(all_s + EPS).mean())
            elif scalar == "rms":
                scalar_denom = torch.sqrt((all_s.pow(2)).mean() + EPS)
            else:
                raise ValueError("scalar must be 'geomean' or 'rms'")

            for q, mh, s in zip(params, mhat, s_vals):
                denom = (s + EPS).pow(p) * scalar_denom.pow(1 - p)
                q.add_(mh / (denom + EPS), alpha=-lr)

        if step % 200 == 0:
            train = loss.item()
            if not np.isfinite(train) or train > 1e8:
                break
            if train < TRAIN_TOL:
                break

    rec, nuc, er = stats(U, V, Xs)
    return dict(U=U.detach(), V=V.detach(), W=(U @ V.T).detach(),
                train=loss_of(U, V, A, y).item(), rec=rec, nuc=nuc, er=er,
                bal=balancedness(U, V), steps=step)


def run_scaledgd(U0, V0, A, y, Xs, lr, damp):
    """Factor-aware preconditioned GD: grad_U (V^T V + damp I)^-1, grad_V (U^T U + damp I)^-1."""
    U = nn.Parameter(U0.clone())
    V = nn.Parameter(V0.clone())
    eye = torch.eye(K)
    loss = None
    for step in range(1, MAX_STEPS + 1):
        if U.grad is not None:
            U.grad.zero_()
            V.grad.zero_()
        loss = loss_of(U, V, A, y)
        loss.backward()
        with torch.no_grad():
            GU = U.grad.clone()
            GV = V.grad.clone()
            pre_u = torch.linalg.solve(V.T @ V + damp * eye, torch.eye(K)).T
            pre_v = torch.linalg.solve(U.T @ U + damp * eye, torch.eye(K)).T
            U.add_(GU @ pre_u, alpha=-lr)
            V.add_(GV @ pre_v, alpha=-lr)
        if step % 200 == 0:
            train = loss.item()
            if not np.isfinite(train) or train > 1e8:
                break
            if train < TRAIN_TOL:
                break
    rec, nuc, er = stats(U, V, Xs)
    return dict(U=U.detach(), V=V.detach(), W=(U @ V.T).detach(),
                train=loss_of(U, V, A, y).item(), rec=rec, nuc=nuc, er=er,
                bal=balancedness(U, V), steps=step)


def run_pair(method, seed, lr, p=None, scalar="geomean", damp=None):
    Xs, A, y = make_problem(seed)
    U, V = factors(seed)
    U0, V0 = U.detach(), V.detach()
    Q = orthogonal_q(seed)
    Ug, Vg = U0 @ Q, V0 @ Q

    if method == "gd":
        base = run_gd(U0, V0, A, y, Xs, lr)
        gauge = run_gd(Ug, Vg, A, y, Xs, lr)
    elif method == "scaledgd":
        base = run_scaledgd(U0, V0, A, y, Xs, lr, damp=damp)
        gauge = run_scaledgd(Ug, Vg, A, y, Xs, lr, damp=damp)
    else:
        base = run_adam_p(U0, V0, A, y, Xs, lr, p=p, scalar=scalar)
        gauge = run_adam_p(Ug, Vg, A, y, Xs, lr, p=p, scalar=scalar)

    denom = base["W"].norm().item() + EPS
    drift = (base["W"] - gauge["W"]).norm().item() / denom
    return dict(drift=drift,
                base_train=base["train"], gauge_train=gauge["train"],
                base_rec=base["rec"], gauge_rec=gauge["rec"],
                base_er=base["er"], gauge_er=gauge["er"],
                base_bal=base["bal"], gauge_bal=gauge["bal"],
                base_steps=base["steps"], gauge_steps=gauge["steps"])


def summarize(label, cfg, rows):
    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    print(f"{label:>18} | {mean('drift'):>10.3e} | "
          f"{mean('base_rec'):>8.4f}/{mean('gauge_rec'):<8.4f} | "
          f"{mean('base_er'):>6.2f}/{mean('gauge_er'):<6.2f} | "
          f"{mean('base_bal'):>8.2e}/{mean('gauge_bal'):<8.2e} | "
          f"{mean('base_train'):>8.1e}/{mean('gauge_train'):<8.1e} | {cfg}", flush=True)


def best_pair(method, cfgs, p=None, scalar="geomean"):
    best = None
    for cfg in cfgs:
        lr = cfg[0]
        damp = cfg[1] if len(cfg) > 1 else None
        rows = [run_pair(method, seed, lr, p=p, scalar=scalar, damp=damp) for seed in SEEDS]
        base_train = np.mean([r["base_train"] for r in rows])
        gauge_train = np.mean([r["gauge_train"] for r in rows])
        base_rec = np.mean([r["base_rec"] for r in rows])
        gauge_rec = np.mean([r["gauge_rec"] for r in rows])
        interp = base_train < 1e-4 and gauge_train < 1e-4
        score = 0.5 * (base_rec + gauge_rec) if interp else 10.0 + 0.5 * (base_rec + gauge_rec)
        if best is None or score < best[0]:
            cfg_text = f"lr={lr:g}" if damp is None else f"lr={lr:g} damp={damp:g}"
            best = (score, rows, cfg_text)
    return best[1], best[2]


def main():
    t0 = time.time()
    print("=" * 118, flush=True)
    print(f"EQUIVARIANCE / BALANCEDNESS | sensing {N}x{N} r*{R_STAR} k{K} m={M} "
          f"({M/DOF:.1f}x dof) init={INIT} | {len(SEEDS)} seeds", flush=True)
    print("Same W0, different latent gauge: U,V vs UQ,VQ. Product drift should be ~0 for gauge-respecting dynamics.",
          flush=True)
    print("=" * 118, flush=True)
    print(f"{'method':>18} | {'W drift':>10} | {'rec base/gauge':>19} | "
          f"{'er base/gauge':>15} | {'bal base/gauge':>21} | {'train base/gauge':>19} | cfg", flush=True)
    print("-" * 118, flush=True)

    configs = [
        ("GD", "gd", [(x,) for x in [0.003, 0.01, 0.03, 0.1]], None, "geomean"),
        ("ScaledGD", "scaledgd",
         [(lr, damp) for lr in [0.001, 0.003, 0.01, 0.03, 0.1] for damp in [1e-3, 1e-2, 1e-1, 1.0]],
         None, "geomean"),
        ("Adam p=1", "adam_p", [(x,) for x in [0.001, 0.003, 0.01, 0.03]], 1.0, "geomean"),
        ("Adam p=.5 geom", "adam_p", [(x,) for x in [0.001, 0.003, 0.01, 0.03, 0.1]], 0.5, "geomean"),
        ("Adam p=0 geom", "adam_p", [(x,) for x in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]], 0.0, "geomean"),
        ("Adam p=0 RMS", "adam_p", [(x,) for x in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]], 0.0, "rms"),
    ]
    for label, method, cfgs, p, scalar in configs:
        rows, cfg = best_pair(method, cfgs, p=p, scalar=scalar)
        summarize(label, cfg, rows)

    print("-" * 118, flush=True)
    print("READ: If Adam has large W drift while GD/RMS-scalar does not, the mechanism is gauge/equivariance breaking.",
          flush=True)
    print(f"[done in {(time.time() - t0) / 60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
