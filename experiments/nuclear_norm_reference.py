"""Min-nuclear-norm interpolant on the exact zoo problem instances (seeds 42/123/456).

min ||X||_*  s.t.  A vec(X) = y,  by Douglas-Rachford / ADMM:
  X <- SVT_rho(Z - Uu);  Z <- Proj_{Avec=y}(X + Uu);  Uu <- Uu + X - Z
The projection is exact: Proj(W) = W + A^T (AA^T)^{-1} (y - A vec(W)).
"""
import numpy as np, torch

N, R_STAR, K = 40, 3, 40
DOF = R_STAR * (2 * N - R_STAR)
M = int(2.0 * DOF)
SEEDS = [42, 123, 456]


def make_problem(seed):  # byte-identical to experiments/restoration_probe.py
    g = torch.Generator().manual_seed(seed)
    Us = torch.randn(N, R_STAR, generator=g) / np.sqrt(R_STAR)
    Vs = torch.randn(N, R_STAR, generator=g) / np.sqrt(R_STAR)
    Xs = Us @ Vs.T
    A = torch.randn(M, N * N, generator=g)
    y = A @ Xs.reshape(-1)
    return Xs.numpy().astype(np.float64), A.numpy().astype(np.float64), y.numpy().astype(np.float64)


def erank(sv):
    p = sv / sv.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def svt(W, tau):
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    s = np.maximum(s - tau, 0.0)
    return (U * s) @ Vt


for seed in SEEDS:
    Xs, A, y = make_problem(seed)
    AAt = A @ A.T
    cho = np.linalg.cholesky(AAt)

    def proj(W):
        r = y - A @ W.reshape(-1)
        z = np.linalg.solve(cho.T, np.linalg.solve(cho, r))
        return W + (A.T @ z).reshape(N, N)

    rho = 1.0
    X = np.zeros((N, N)); Z = proj(np.zeros((N, N))); Uu = np.zeros((N, N))
    for it in range(20000):
        X = svt(Z - Uu, rho)
        Znew = proj(X + Uu)
        Uu = Uu + X - Znew
        pr = np.linalg.norm(X - Znew); dr = rho * np.linalg.norm(Znew - Z)
        Z = Znew
        if pr < 1e-11 and dr < 1e-11:
            break
    Xhat = proj(X)  # report the exactly-feasible iterate
    feas = np.linalg.norm(A @ Xhat.reshape(-1) - y) / np.linalg.norm(y)
    sv = np.linalg.svd(Xhat, compute_uv=False)
    svt_true = np.linalg.svd(Xs, compute_uv=False)
    rec = np.linalg.norm(Xhat - Xs) / np.linalg.norm(Xs)
    print(f"seed {seed}: iters {it+1}  feas {feas:.2e}  "
          f"nuc {sv.sum():.6f} (X* {svt_true.sum():.6f})  "
          f"rec {rec:.6f}  erank {erank(sv):.3f}  pr {pr:.1e}")
