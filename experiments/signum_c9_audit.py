"""
Recovery without equivariance: annealed sign descent (Appendix C9).

Audits the one coordinate-wise method that recovers at long horizons (signum):
  P1 mechanism: is signum's windowed displacement rank-1 dominated? (falsification test)
  P2 horizon x schedule: is the fail-at-2e4 / work-at-4e4 boundary a schedule-shape artifact?
  P3 spectral tail: does signum's long-anneal recovery survive spectral tails?
Protocol identical to optimizer_zoo_bias.py and muon_phase_diagram.py (n=40, rank 3, m = 2 x dof,
init 1e-3, wd=0).
"""
import math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from restoration_probe import make_problem, factors, stats, N, R_STAR, K, M
from muon_phase_diagram import make_tail_problem

torch.set_num_threads(max(1, os.cpu_count() - 2))
B1 = 0.9
TRAIN_TOL = 1e-7


def sched_lr(lr, step, T, shape):
    if shape == 'cos':
        return lr * 0.5 * (1 + math.cos(math.pi * step / T))
    return lr * (1 - step / T)


def top_stats(Mx):
    sv = torch.linalg.svdvals(Mx)
    s1s2 = (sv[0] / (sv[1] + 1e-30)).item()
    en = (sv[0] ** 2 / (sv ** 2).sum()).item()
    return s1s2, en


def run_signum(problem, seed, lr, T, shape='cos', diag=False, Xs=None, A=None, y=None):
    U, V = factors(seed)
    params = [U, V]
    mom = [torch.zeros_like(p) for p in params]
    Uprev, Vprev = U.detach().clone(), V.detach().clone()
    rows = []
    train = 1.0
    for step in range(1, T + 1):
        for p in params:
            if p.grad is not None:
                p.grad.detach_(); p.grad.zero_()
        W = (U @ V.T).reshape(-1)
        loss = ((A @ W - y) ** 2).mean()
        train = loss.item()
        if not np.isfinite(train) or train > 1e8:
            return dict(rec=float('nan'), er=float('nan'), train=float('nan')), rows
        loss.backward()
        lr_t = sched_lr(lr, step, T, shape)
        with torch.no_grad():
            for i, p in enumerate(params):
                mom[i].mul_(B1).add_(p.grad, alpha=1 - B1)
                p.add_(torch.sign(mom[i]), alpha=-lr_t)
            if diag and step % 2000 == 0:
                rec, _, er = stats(U, V, Xs)
                m1, me = top_stats(mom[0])
                s1, se = top_stats(torch.sign(mom[0]))
                dU = U.detach() - Uprev
                d1, de = top_stats(dU)
                dW = U.detach() @ V.detach().T - Uprev @ Vprev.T
                align = (dW * Xs).sum() / (dW.norm() * Xs.norm() + 1e-30)
                svW = torch.linalg.svdvals(U.detach() @ V.detach().T)[:4]
                rows.append((step, rec, m1, me, s1, se, d1, de, align.item(),
                             [f"{v:.3f}" for v in svW.tolist()], er))
                Uprev, Vprev = U.detach().clone(), V.detach().clone()
    rec, _, er = stats(U, V, Xs)
    return dict(rec=rec, er=er, train=train), rows


def p1():
    print("=" * 100)
    print("P1 MECHANISM DIAG | signum cosine T=4e4 lr=3e-3 | is anything rank-1 dominated?")
    print(f"{'step':>6} {'rec':>8} | mom s1/s2, topE | sign(mom) s1/s2, topE | windowed-dU s1/s2, topE | "
          f"<dW,X*> | sv(W)[:4] | erank")
    for seed in [42, 123, 456]:
        Xs, A, y = make_problem(seed)
        res, rows = run_signum(None, seed, 3e-3, 40000, 'cos', diag=True, Xs=Xs, A=A, y=y)
        print(f"-- seed {seed}: final rec={res['rec']:.4f} er={res['er']:.2f} train={res['train']:.2e}")
        for r in rows:
            print(f"{r[0]:>6} {r[1]:>8.4f} | {r[2]:>6.2f} {r[3]:>5.1%} | {r[4]:>6.2f} {r[5]:>5.1%} | "
                  f"{r[6]:>6.2f} {r[7]:>5.1%} | {r[8]:>7.3f} | {r[9]} | {r[10]:.2f}", flush=True)


def p2():
    print("=" * 100)
    print("P2 HORIZON x SCHEDULE | recovery (mean of 3 seeds) vs anneal length and shape")
    print(f"{'T':>6} {'shape':>5} {'lr':>7} | {'rec mean':>9} {'per-seed':>30}")
    for T in [10000, 20000, 40000, 60000]:
        for shape in ['cos', 'lin']:
            for lr in [1e-3, 3e-3]:
                recs = []
                for seed in [42, 123, 456]:
                    Xs, A, y = make_problem(seed)
                    res, _ = run_signum(None, seed, lr, T, shape, Xs=Xs, A=A, y=y)
                    recs.append(res['rec'])
                print(f"{T:>6} {shape:>5} {lr:>7g} | {np.mean(recs):>9.4f} "
                      f"{'  '.join(f'{r:.4f}' for r in recs):>30}", flush=True)


def p3():
    print("=" * 100)
    print("P3 TAU TAIL SWEEP | signum cosine T=4e4 | recovery vs tail energy tau^2 (Table-3 grid)")
    print(f"{'tau':>5} {'lr':>7} | {'rec mean':>9} {'er mean':>8} {'train mean':>10} {'per-seed rec':>30}")
    for tau in [0.0, 0.05, 0.1, 0.2, 0.35, 0.5]:
        for lr in [1e-3, 3e-3]:
            recs, ers, trs = [], [], []
            for seed in [42, 123, 456]:
                X, A, y = make_tail_problem(seed, tau)
                res, _ = run_signum(None, seed, lr, 40000, 'cos', Xs=X, A=A, y=y)
                recs.append(res['rec']); ers.append(res['er']); trs.append(res['train'])
            print(f"{tau:>5.2f} {lr:>7g} | {np.mean(recs):>9.4f} {np.mean(ers):>8.2f} "
                  f"{np.mean(trs):>10.2e} {'  '.join(f'{r:.4f}' for r in recs):>30}", flush=True)


if __name__ == '__main__':
    t0 = time.time()
    p1(); print(f"[p1 done {(time.time()-t0)/60:.1f}m]", flush=True)
    p2(); print(f"[p2 done {(time.time()-t0)/60:.1f}m]", flush=True)
    p3(); print(f"[p3 done {(time.time()-t0)/60:.1f}m]", flush=True)
