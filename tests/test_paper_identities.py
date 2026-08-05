"""Numerical verification of the paper's algebraic claims.

Each test names the paper object it checks, evaluates both sides of the claimed identity in
float64 at random points, and asserts agreement at machine precision. Nothing is mocked or
compared against a hand-copied constant: every number comes from a real loss, real autograd
gradients and the real update rules, with Muon's Newton-Schulz iteration and Shampoo's inverse
roots imported from the experiment code behind the paper's tables.

Every positive claim is paired with a negative control that has to fail the same check. That is
what rules out a vacuous pass: a harness bug making the two runs identical (Q = I, a dead
gradient, a no-op update) would satisfy the equivariance tests, but the coordinate-wise tests
assert a large violation and would catch it.

Run:  pytest -q tests/test_paper_identities.py
  or: python tests/test_paper_identities.py
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from optimizer_zoo_bias import newton_schulz, matrix_pow  # noqa: E402  (real code under test)

torch.set_default_dtype(torch.float64)

EXACT = 1e-11        # float64 round-off budget for a few dozen flops
LOOSE = 1e-9         # after several steps of eigendecomposition-based updates
BROKEN = 1e-3        # a gauge violation must be at least this large to count as structural


# --------------------------------------------------------------------------------------------
# problem: matrix sensing, the testbed of Section 3
# --------------------------------------------------------------------------------------------

def make_problem(n=6, k=3, r=2, m=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    Us = torch.randn(n, r, generator=g) / math.sqrt(r)
    Vs = torch.randn(n, r, generator=g) / math.sqrt(r)
    Xs = Us @ Vs.T
    A = torch.randn(m, n * n, generator=g)
    y = A @ Xs.reshape(-1)
    return A, y, n, k


def make_factors(n, k, seed=0, scale=0.3):
    g = torch.Generator().manual_seed(seed + 7919)
    U = (torch.randn(n, k, generator=g) * scale).requires_grad_(True)
    V = (torch.randn(n, k, generator=g) * scale).requires_grad_(True)
    return U, V


def loss_fn(U, V, A, y):
    return ((A @ (U @ V.T).reshape(-1) - y) ** 2).mean()


def rand_orthogonal(k, seed=0):
    g = torch.Generator().manual_seed(seed + 104729)
    Q, R = torch.linalg.qr(torch.randn(k, k, generator=g))
    return Q * torch.sign(torch.diagonal(R))          # unique QR -> genuinely random O(k)


def grads(U, V, A, y):
    U = U.detach().clone().requires_grad_(True)
    V = V.detach().clone().requires_grad_(True)
    loss_fn(U, V, A, y).backward()
    return U.grad.clone(), V.grad.clone()


def rel(a, b):
    return (a - b).norm().item() / max(b.norm().item(), 1e-300)


# --------------------------------------------------------------------------------------------
# Lemma 3.2 / Definition 3.1: the gauge, and why O(k) is the maximal isometric part
# --------------------------------------------------------------------------------------------

def test_lemma32_gl_preserves_product_but_only_ok_is_isometric():
    """Lemma 3.2.  (U,V) -> (UA, VA^-T) preserves UV^T for all A in GL(k); it preserves
    ||U||_F^2 + ||V||_F^2 iff A is orthogonal."""
    A_, y, n, k = make_problem()
    U, V = make_factors(n, k)
    U, V = U.detach(), V.detach()
    g = torch.Generator().manual_seed(11)

    for _ in range(20):
        M = torch.randn(k, k, generator=g)
        if torch.linalg.matrix_rank(M) < k:
            continue
        Ua, Va = U @ M, V @ torch.linalg.inv(M).T
        assert rel(Ua @ Va.T, U @ V.T) < EXACT, "GL action must preserve the represented matrix"

        before = U.norm() ** 2 + V.norm() ** 2
        after = Ua.norm() ** 2 + Va.norm() ** 2
        is_orth = rel(M @ M.T, torch.eye(k)) < EXACT
        is_isom = abs((after - before).item()) / before.item() < EXACT
        assert is_orth == is_isom, "isometry must hold exactly on O(k) and fail off it"

    Q = rand_orthogonal(k, 3)                          # the orthogonal case, checked positively
    Uq, Vq = U @ Q, V @ torch.linalg.inv(Q).T
    assert rel(Uq @ Vq.T, U @ V.T) < EXACT
    assert abs((Uq.norm() ** 2 + Vq.norm() ** 2 - U.norm() ** 2 - V.norm() ** 2).item()) < 1e-10


def test_lemma41_gradient_covariance_by_autograd():
    """Lemma 4.1.  grad_U L(UQ, VQ) = (grad_U L(U,V)) Q, and likewise for V.  Both sides from
    autograd on the real sensing loss, not from the analytic formula."""
    A, y, n, k = make_problem()
    U, V = make_factors(n, k)
    Q = rand_orthogonal(k, 5)

    gU, gV = grads(U, V, A, y)
    gUq, gVq = grads(U.detach() @ Q, V.detach() @ Q, A, y)

    assert rel(gUq, gU @ Q) < EXACT
    assert rel(gVq, gV @ Q) < EXACT
    # negative control: a non-orthogonal change of basis does NOT transform this way
    B = torch.eye(k) + 0.4 * torch.randn(k, k, generator=torch.Generator().manual_seed(2))
    gUb, _ = grads(U.detach() @ B, V.detach() @ B, A, y)
    assert rel(gUb, gU @ B) > BROKEN


def test_corollary_depth_interface_gauge_covariance():
    """Corollary A.3.  For W = U1 U2 U3 the interface gauge (Ui Q, Q^T U(i+1)) leaves the loss
    invariant and transforms the two gradients covariantly on the acted side."""
    g = torch.Generator().manual_seed(31)
    d, k1, k2 = 5, 4, 4
    A = torch.randn(12, d * d, generator=g)
    y = torch.randn(12, generator=g)
    Us = [torch.randn(d, k1, generator=g) * 0.4,
          torch.randn(k1, k2, generator=g) * 0.4,
          torch.randn(k2, d, generator=g) * 0.4]

    def deep_grads(mats):
        ps = [m.detach().clone().requires_grad_(True) for m in mats]
        W = ps[0] @ ps[1] @ ps[2]
        ((A @ W.reshape(-1) - y) ** 2).mean().backward()
        return [p.grad.clone() for p in ps], (A @ W.detach().reshape(-1) - y).pow(2).mean().item()

    Q = rand_orthogonal(k1, 9)
    base_g, base_l = deep_grads(Us)
    rot = [Us[0] @ Q, Q.T @ Us[1], Us[2]]
    rot_g, rot_l = deep_grads(rot)

    assert abs(rot_l - base_l) < 1e-12, "the interface gauge must not change the loss"
    assert rel(rot_g[0], base_g[0] @ Q) < EXACT          # acted on the right
    assert rel(rot_g[1], Q.T @ base_g[1]) < EXACT        # acted on the left


# --------------------------------------------------------------------------------------------
# update rules, written exactly as Section 4 states them
# --------------------------------------------------------------------------------------------

def _gd(state, gs, lr, **kw):
    return [-lr * g for g in gs]


def _momentum(state, gs, lr, beta=0.9, **kw):
    state.setdefault("m", [torch.zeros_like(g) for g in gs])
    for i, g in enumerate(gs):
        state["m"][i] = beta * state["m"][i] + g
    return [-lr * m for m in state["m"]]


def _scalar_adam(state, gs, lr, b1=0.9, b2=0.999, eps=1e-8, **kw):
    """Proposition 4.2(2): shared scalar nu, an EMA of mean(g^2) pooled over both factors."""
    state.setdefault("m", [torch.zeros_like(g) for g in gs])
    state.setdefault("nu", 0.0)
    state["t"] = state.get("t", 0) + 1
    pooled = torch.cat([g.reshape(-1) for g in gs])
    state["nu"] = b2 * state["nu"] + (1 - b2) * (pooled ** 2).mean().item()
    nu_hat = state["nu"] / (1 - b2 ** state["t"])
    out = []
    for i, g in enumerate(gs):
        state["m"][i] = b1 * state["m"][i] + (1 - b1) * g
        out.append(-lr * (state["m"][i] / (1 - b1 ** state["t"])) / (math.sqrt(nu_hat) + eps))
    return out


def _muon(state, gs, lr, beta=0.9, **kw):
    state.setdefault("m", [torch.zeros_like(g) for g in gs])
    out = []
    for i, g in enumerate(gs):
        state["m"][i] = beta * state["m"][i] + g
        out.append(-lr * newton_schulz(state["m"][i]))
    return out


def _shampoo(state, gs, lr, lam=1.0, **kw):
    state.setdefault("L", [torch.zeros(g.shape[0], g.shape[0]) for g in gs])
    state.setdefault("R", [torch.zeros(g.shape[1], g.shape[1]) for g in gs])
    out = []
    for i, g in enumerate(gs):
        state["L"][i] = state["L"][i] + g @ g.T
        state["R"][i] = state["R"][i] + g.T @ g
        pl = matrix_pow(state["L"][i] + lam * torch.eye(g.shape[0]), -0.25)
        pr = matrix_pow(state["R"][i] + lam * torch.eye(g.shape[1]), -0.25)
        out.append(-lr * (pl @ g @ pr))
    return out


def _adam(state, gs, lr, b1=0.9, b2=0.999, eps=1e-8, **kw):
    state.setdefault("m", [torch.zeros_like(g) for g in gs])
    state.setdefault("v", [torch.zeros_like(g) for g in gs])
    state["t"] = state.get("t", 0) + 1
    out = []
    for i, g in enumerate(gs):
        state["m"][i] = b1 * state["m"][i] + (1 - b1) * g
        state["v"][i] = b2 * state["v"][i] + (1 - b2) * g ** 2
        mh = state["m"][i] / (1 - b1 ** state["t"])
        vh = state["v"][i] / (1 - b2 ** state["t"])
        out.append(-lr * mh / (vh.sqrt() + eps))
    return out


def _rmsprop(state, gs, lr, rho=0.99, eps=1e-8, **kw):
    state.setdefault("v", [torch.zeros_like(g) for g in gs])
    out = []
    for i, g in enumerate(gs):
        state["v"][i] = rho * state["v"][i] + (1 - rho) * g ** 2
        out.append(-lr * g / (state["v"][i].sqrt() + eps))
    return out


def _signum(state, gs, lr, beta=0.9, **kw):
    state.setdefault("m", [torch.zeros_like(g) for g in gs])
    out = []
    for i, g in enumerate(gs):
        state["m"][i] = beta * state["m"][i] + (1 - beta) * g
        out.append(-lr * torch.sign(state["m"][i]))
    return out


def _lion(state, gs, lr, b1=0.9, b2=0.99, **kw):
    state.setdefault("m", [torch.zeros_like(g) for g in gs])
    out = []
    for i, g in enumerate(gs):
        c = torch.sign(b1 * state["m"][i] + (1 - b1) * g)
        state["m"][i] = b2 * state["m"][i] + (1 - b2) * g
        out.append(-lr * c)
    return out


def adafactor_update(G, eps=0.0):
    """Adafactor's zero-state step: rank-one reconstruction of the second moment."""
    v = G ** 2 + eps
    r = v.sum(dim=1, keepdim=True)
    c = v.sum(dim=0, keepdim=True)
    vhat = (r @ c) / r.sum()
    return G / vhat.sqrt()


def _adafactor(state, gs, lr, **kw):
    return [-lr * adafactor_update(g) for g in gs]


EQUIVARIANT = {"gd": _gd, "momentum": _momentum, "scalar_adam": _scalar_adam,
               "muon": _muon, "shampoo": _shampoo}
COORDINATE_WISE = {"adam": _adam, "rmsprop": _rmsprop, "signum": _signum,
                   "lion": _lion, "adafactor": _adafactor}


def run_steps(rule, U0, V0, A, y, lr, steps):
    U, V = U0.detach().clone(), V0.detach().clone()
    state = {}
    for _ in range(steps):
        gU, gV = grads(U, V, A, y)
        dU, dV = rule(state, [gU, gV], lr)
        U, V = U + dU, V + dV
    return U, V


# --------------------------------------------------------------------------------------------
# Propositions 4.2 and 4.3: the classification, positive and negative
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(EQUIVARIANT))
def test_prop42_equivariant_rules_commute_with_the_gauge_over_many_steps(name):
    """Proposition 4.2.  Running the rule from (U0 Q, V0 Q) must give exactly (Ut Q, Vt Q) for
    every t, with the optimizer's own state (momentum, EMA, Shampoo accumulators) live."""
    A, y, n, k = make_problem()
    U0, V0 = make_factors(n, k)
    Q = rand_orthogonal(k, 13)
    lr = 3e-2

    Ut, Vt = run_steps(EQUIVARIANT[name], U0, V0, A, y, lr, steps=6)
    Ur, Vr = run_steps(EQUIVARIANT[name], U0.detach() @ Q, V0.detach() @ Q, A, y, lr, steps=6)

    assert rel(Ur, Ut @ Q) < LOOSE, f"{name} broke gauge equivariance on U"
    assert rel(Vr, Vt @ Q) < LOOSE, f"{name} broke gauge equivariance on V"
    # the represented matrix, the quantity the paper actually claims is well defined
    assert rel(Ur @ Vr.T, Ut @ Vt.T) < LOOSE
    # guard against a vacuous pass: the two runs must genuinely differ in parameter space
    assert rel(Ur, Ut) > BROKEN, "Q was too close to the identity for this test to mean anything"


@pytest.mark.parametrize("name", sorted(COORDINATE_WISE))
def test_prop43_coordinatewise_rules_break_the_gauge_at_step_one(name):
    """Proposition 4.3.  The zero-state first step must already violate equivariance, and by a
    margin far above float noise -- otherwise the paper's dichotomy is not observable."""
    A, y, n, k = make_problem()
    U0, V0 = make_factors(n, k)
    Q = rand_orthogonal(k, 13)
    lr = 3e-2

    Ut, Vt = run_steps(COORDINATE_WISE[name], U0, V0, A, y, lr, steps=1)
    Ur, Vr = run_steps(COORDINATE_WISE[name], U0.detach() @ Q, V0.detach() @ Q, A, y, lr, steps=1)

    assert rel(Ur, Ut @ Q) > BROKEN, f"{name} did NOT break the gauge; the dichotomy would fail"
    # and it is a failure of the represented matrix, not only of internal state
    assert rel(Ur @ Vr.T, Ut @ Vt.T) > 1e-9


def test_prop42_scalar_adam_is_the_repair_of_adam():
    """The dichotomy is caused by the denominator alone: Adam and scalar-Adam share every other
    ingredient, and only the shared-scalar one is equivariant."""
    A, y, n, k = make_problem()
    U0, V0 = make_factors(n, k)
    Q = rand_orthogonal(k, 13)
    err = {}
    for nm, rule in (("adam", _adam), ("scalar_adam", _scalar_adam)):
        Ut, _ = run_steps(rule, U0, V0, A, y, 3e-2, 4)
        Ur, _ = run_steps(rule, U0.detach() @ Q, V0.detach() @ Q, A, y, 3e-2, 4)
        err[nm] = rel(Ur, Ut @ Q)
    assert err["scalar_adam"] < LOOSE < BROKEN < err["adam"]


def test_prop42_scalar_ema_equals_mean_of_the_entrywise_ema():
    """Proof of Proposition 4.2(2): because the mean is linear it commutes with the EMA, so
    storing one scalar changes no number relative to averaging Adam's entrywise v."""
    A, y, n, k = make_problem()
    U, V = make_factors(n, k)
    b2 = 0.999
    nu, v = 0.0, [torch.zeros(n, k), torch.zeros(n, k)]
    for _ in range(15):
        gU, gV = grads(U, V, A, y)
        pooled = torch.cat([gU.reshape(-1), gV.reshape(-1)])
        nu = b2 * nu + (1 - b2) * (pooled ** 2).mean().item()
        v = [b2 * v[0] + (1 - b2) * gU ** 2, b2 * v[1] + (1 - b2) * gV ** 2]
        U = U.detach() - 3e-2 * gU
        V = V.detach() - 3e-2 * gV
    mean_v = torch.cat([v[0].reshape(-1), v[1].reshape(-1)]).mean().item()
    assert abs(nu - mean_v) / mean_v < 1e-13

    # and the rule as implemented must accumulate exactly that scalar, pooled over both factors.
    # (Pooling over one factor alone would still be gauge-invariant, so the equivariance tests
    # cannot see this; it has to be pinned directly against the definition.)
    A2, y2, n2, k2 = make_problem(seed=3)
    U2, V2 = make_factors(n2, k2, seed=3)
    st, nu_ref, t = {}, 0.0, 0
    Uc, Vc = U2.detach().clone(), V2.detach().clone()
    for _ in range(5):
        gU, gV = grads(Uc, Vc, A2, y2)
        t += 1
        nu_ref = b2 * nu_ref + (1 - b2) * torch.cat([gU.reshape(-1), gV.reshape(-1)]).pow(2).mean().item()
        dU, dV = _scalar_adam(st, [gU, gV], 1e-2)
        Uc, Vc = Uc + dU, Vc + dV
    assert abs(st["nu"] - nu_ref) / nu_ref < 1e-13, "nu must pool g^2 over both factors"


def test_prop43_sign_witness_from_the_proof():
    """The theta = pi/4 counterexample printed in the proof of Proposition 4.3."""
    G = torch.tensor([[1.0, 1.0]])
    c = s = 1 / math.sqrt(2)
    Q = torch.tensor([[c, s], [-s, c]])
    assert rel(G @ Q, torch.tensor([[0.0, math.sqrt(2)]])) < EXACT
    assert rel(torch.sign(G @ Q), torch.tensor([[0.0, 1.0]])) < EXACT
    assert rel(torch.sign(G) @ Q, torch.tensor([[0.0, math.sqrt(2)]])) < EXACT
    assert (torch.sign(G @ Q) - torch.sign(G) @ Q).norm() > 0.4


def test_prop43_adafactor_witness_matrices_from_the_proof():
    """The three displayed matrices in the Adafactor half of the proof of Proposition 4.3,
    recomputed from Adafactor's definition rather than copied."""
    for e in (0.1, 0.3, 0.5, 2.0):
        G = torch.tensor([[1.0, 0.0], [0.0, e]])
        c = 1 / math.sqrt(2)
        Q = torch.tensor([[c, c], [-c, c]])
        root = math.sqrt(1 + e ** 2)

        assert rel(adafactor_update(G),
                   root * torch.tensor([[1.0, 0.0], [0.0, 1 / e]])) < EXACT
        assert rel(adafactor_update(G @ Q),
                   torch.tensor([[1.0, 1.0], [-1.0, 1.0]])) < EXACT
        assert rel(adafactor_update(G) @ Q,
                   (root / math.sqrt(2)) * torch.tensor([[1.0, 1.0], [-1 / e, 1 / e]])) < EXACT
        # row sums invariant, column sums conjugated: the stated cause of the failure
        v, vq = G ** 2, (G @ Q) ** 2
        assert rel(v.sum(1), vq.sum(1)) < EXACT
        assert rel(vq.sum(0), 0.5 * (1 + e ** 2) * torch.ones(2)) < EXACT


# --------------------------------------------------------------------------------------------
# Proposition 4.4: the exact first-step Adam defect identity
# --------------------------------------------------------------------------------------------

def d_eps(G, eps):
    return G / (G.abs() + eps)


@pytest.mark.parametrize("eps", [1e-8, 1e-3, 0.1])
@pytest.mark.parametrize("eta", [1e-3, 0.05, 0.7])
def test_prop44_exact_first_step_defect_identity(eta, eps):
    """Proposition 4.4.  The left-hand side is computed by actually running one zero-state Adam
    step on both the reference and the gauge-rotated initialisation; the right-hand side is the
    paper's closed form.  A wrong transpose or sign anywhere in the display fails this."""
    A, y, n, k = make_problem(seed=4)
    U0, V0 = make_factors(n, k, seed=4)
    U0, V0 = U0.detach(), V0.detach()
    Q = rand_orthogonal(k, 21)

    GU, GV = grads(U0, V0, A, y)
    W1 = (U0 - eta * d_eps(GU, eps)) @ (V0 - eta * d_eps(GV, eps)).T

    GUq, GVq = grads(U0 @ Q, V0 @ Q, A, y)
    Ur = U0 @ Q - eta * d_eps(GUq, eps)
    Vr = V0 @ Q - eta * d_eps(GVq, eps)
    W1t = Ur @ Vr.T                                     # gauge alignment cancels in the product

    EU = d_eps(GU @ Q, eps) @ Q.T - d_eps(GU, eps)
    EV = d_eps(GV @ Q, eps) @ Q.T - d_eps(GV, eps)
    DU, DV = d_eps(GU, eps), d_eps(GV, eps)

    rhs = (-eta * (EU @ V0.T + U0 @ EV.T)
           + eta ** 2 * (EU @ DV.T + DU @ EV.T + EU @ EV.T))
    assert rel(W1t - W1, rhs) < 1e-10

    # the alternative quadratic form stated right after the display
    alt = (-eta * (EU @ V0.T + U0 @ EV.T)
           + eta ** 2 * (d_eps(GU @ Q, eps) @ d_eps(GV @ Q, eps).T - DU @ DV.T))
    assert rel(W1t - W1, alt) < 1e-10

    # the defect is real, not a rounding artefact
    assert (W1t - W1).norm().item() > 1e-8


def test_prop44_scalar_witness_products():
    """The n=1, k=2 witness in the statement: the two first-step products must be
    (1 - eta/(1+eps))^2 and (1 - eta/(2^-1/2 + eps))^2, and must differ for 0 < eta < 2^-1/2+eps."""
    eps = 1e-8
    c = 1 / math.sqrt(2)
    Q = torch.tensor([[c, c], [-c, c]])
    U0 = torch.tensor([[1.0, 0.0]])
    V0 = U0.clone()

    for eta in (1e-3, 0.1, 0.5):
        # f(w) = w^2/2 at w = U0 V0^T = 1  =>  grad_U = w * V0 = V0, grad_V = w * U0 = U0
        gU, gV = V0.clone(), U0.clone()
        w_ref = ((U0 - eta * d_eps(gU, eps)) @ (V0 - eta * d_eps(gV, eps)).T).item()
        Uq, Vq = U0 @ Q, V0 @ Q
        gUq, gVq = Vq.clone(), Uq.clone()
        w_rot = ((Uq - eta * d_eps(gUq, eps)) @ (Vq - eta * d_eps(gVq, eps)).T).item()

        assert abs(w_ref - (1 - eta / (1 + eps)) ** 2) < 1e-12
        assert abs(w_rot - (1 - eta / (c + eps)) ** 2) < 1e-12
        assert abs(w_ref - w_rot) > 1e-6


# --------------------------------------------------------------------------------------------
# Theorem 4.5 and Proposition A.1: structure and the spectral transfer function
# --------------------------------------------------------------------------------------------

def msign(M):
    A, _, Bh = torch.linalg.svd(M, full_matrices=False)
    return A @ Bh


def test_thm45_equivariant_map_is_gram_determined_left_preconditioner():
    """Theorem 4.5.  For an equivariant Phi, X(G) = Phi(G) G^+ must be gauge-invariant and a
    function of G G^T alone; and the proof's Q := G1^+ G2 must be orthogonal when the Grams agree."""
    g = torch.Generator().manual_seed(77)
    n, k = 7, 4
    G = torch.randn(n, k, generator=g)
    Q = rand_orthogonal(k, 33)

    for Phi in (msign, lambda M: M.norm() * M, lambda M: torch.diag(torch.arange(1., n + 1)) @ M):
        X1 = Phi(G) @ torch.linalg.pinv(G)
        X2 = Phi(G @ Q) @ torch.linalg.pinv(G @ Q)
        assert rel(X2, X1) < 1e-10, "X(G) must be gauge-invariant"
        assert rel(X1 @ G, Phi(G)) < 1e-10, "H(GG^T) G must reconstruct Phi(G)"

    G2 = G @ Q
    assert rel(G @ G.T, G2 @ G2.T) < 1e-12, "equal Grams on one orbit"
    Qhat = torch.linalg.pinv(G) @ G2
    assert rel(Qhat.T @ Qhat, torch.eye(k)) < 1e-10, "the proof's Q must come out orthogonal"
    assert rel(G @ Qhat, G2) < 1e-10

    # converse direction: any H(GG^T) G is equivariant
    H = lambda P: torch.linalg.inv(P + torch.eye(n))
    assert rel(H(G @ G.T) @ (G @ Q), (H(G @ G.T) @ G) @ Q) < EXACT


def test_propa1_spectral_transfer_function():
    """Proposition A.1.  psi(GG^T) G must equal sum_i h(sigma_i) u_i v_i^T with h(s) = psi(s^2) s."""
    g = torch.Generator().manual_seed(101)
    n, k = 6, 4
    G = torch.randn(n, k, generator=g)
    Uu, S, Vh = torch.linalg.svd(G, full_matrices=False)

    for psi, h in ((lambda lam: lam ** -0.25, lambda s: s ** 0.5),
                   (lambda lam: torch.ones_like(lam), lambda s: s),
                   (lambda lam: lam ** -0.5, lambda s: torch.ones_like(s))):
        P = G @ G.T
        vals, vecs = torch.linalg.eigh(P)
        keep = vals > 1e-10
        fP = (vecs[:, keep] * psi(vals[keep])) @ vecs[:, keep].T
        lhs = fP @ G
        rhs = (Uu * h(S)) @ Vh
        assert rel(lhs, rhs) < 1e-9


def test_propa1_undamped_shampoo_one_step_equals_msign():
    """The identity displayed in the proof of Proposition A.1:
    (GG^T)^-1/4 G (G^T G)^-1/4 = msign(G) on the support."""
    g = torch.Generator().manual_seed(202)
    for n, k in ((6, 4), (5, 5), (4, 6)):
        G = torch.randn(n, k, generator=g)

        def inv_root(P):
            vals, vecs = torch.linalg.eigh(P)
            keep = vals > 1e-10
            return (vecs[:, keep] * vals[keep] ** -0.25) @ vecs[:, keep].T

        lhs = inv_root(G @ G.T) @ G @ inv_root(G.T @ G)
        assert rel(lhs, msign(G)) < 1e-9


def test_propa1_lipschitz_witness_diverges_for_flat_schedules_only():
    """Consequence (ii): the quotient |h(delta)|/delta at G_pm = diag(1, +-delta) must blow up as
    delta -> 0 for the exact polar map and stay bounded (= eta) for GD."""
    quotients_msign, quotients_gd = [], []
    for delta in (1e-1, 1e-2, 1e-3, 1e-4):
        Gp = torch.diag(torch.tensor([1.0, delta]))
        Gm = torch.diag(torch.tensor([1.0, -delta]))
        denom = (Gp - Gm).norm().item()
        quotients_msign.append((msign(Gp) - msign(Gm)).norm().item() / denom)
        quotients_gd.append((Gp - Gm).norm().item() / denom)

    assert quotients_msign[-1] > quotients_msign[0] * 500, "msign quotient must diverge"
    assert all(abs(q - 1.0) < 1e-12 for q in quotients_gd), "GD's quotient must stay at eta"
    for i, delta in enumerate((1e-1, 1e-2, 1e-3, 1e-4)):
        assert abs(quotients_msign[i] - 1.0 / delta) < 1e-6 * (1.0 / delta)


def test_muon_newton_schulz_equivariance_at_every_truncation():
    """Proposition 4.2(3).  Equivariance must hold exactly at every finite Newton-Schulz
    truncation, for tall, square and wide gradients (the repo transposes tall inputs)."""
    g = torch.Generator().manual_seed(303)
    for n, k in ((8, 4), (5, 5), (3, 6)):
        M = torch.randn(n, k, generator=g)
        Q = rand_orthogonal(k, n * 13)
        for steps in range(1, 6):
            assert rel(newton_schulz(M @ Q, steps=steps),
                       newton_schulz(M, steps=steps) @ Q) < 1e-10, (n, k, steps)


def test_shampoo_step_matches_the_displayed_two_sided_rule():
    """Proposition 4.2(4) as displayed: Delta = (L+lam I)^-1/4 G (R+lam I)^-1/4, with both roots.
    Dropping the right root would leave the rule gauge-equivariant (it is then a Gram-determined
    left preconditioner, Theorem 4.5), so the equivariance tests cannot detect that error and
    the shape of the rule has to be pinned against the display directly."""
    A, y, n, k = make_problem(seed=17)
    U, V = make_factors(n, k, seed=17)
    lam, lr = 1.0, 0.1
    gU, gV = grads(U, V, A, y)

    st = {}
    dU, _ = _shampoo(st, [gU, gV], lr, lam=lam)

    def root(P):
        vals, vecs = torch.linalg.eigh(0.5 * (P + P.T))
        return (vecs * vals.clamp(min=1e-6) ** -0.25) @ vecs.T

    expected = -lr * (root(gU @ gU.T + lam * torch.eye(n)) @ gU
                      @ root(gU.T @ gU + lam * torch.eye(k)))
    assert rel(dU, expected) < 1e-10

    left_only = -lr * (root(gU @ gU.T + lam * torch.eye(n)) @ gU)
    assert rel(dU, left_only) > 1e-2, "the two-sided rule must differ materially from left-only"


def test_shampoo_spectral_function_commutes_with_conjugation():
    """Proof of Proposition 4.2(4): h(Q^T R Q) = Q^T h(R) Q for h(R) = (R + lam I)^-1/4,
    damping included."""
    g = torch.Generator().manual_seed(404)
    k = 5
    X = torch.randn(9, k, generator=g)
    R = X.T @ X
    Q = rand_orthogonal(k, 55)
    for lam in (0.0, 1.0, 7.5):
        lhs = matrix_pow(Q.T @ R @ Q + lam * torch.eye(k), -0.25)
        rhs = Q.T @ matrix_pow(R + lam * torch.eye(k), -0.25) @ Q
        assert rel(lhs, rhs) < 1e-9


# --------------------------------------------------------------------------------------------
# Theorem 4.6 (transfer) and Proposition A.4 (balancedness): the two dynamical claims
# --------------------------------------------------------------------------------------------

def rk4(f, x0, t0, t1, n_steps):
    x, t = x0.clone(), t0
    h = (t1 - t0) / n_steps
    for _ in range(n_steps):
        k1 = f(t, x)
        k2 = f(t + h / 2, x + h / 2 * k1)
        k3 = f(t + h / 2, x + h / 2 * k2)
        k4 = f(t + h, x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h
    return x


def test_thm46_transfer_theorem_time_change_direction():
    """Theorem 4.6.  The scalar-preconditioned flow at physical time T must coincide with
    gradient flow at reparameterised time tau(T) = int_0^T du/a(u).  The negative control uses
    the opposite time change, int_0^T a du, which must NOT match -- this is the error the test
    exists to catch."""
    A, y, n, k = make_problem(seed=8)
    U0, V0 = make_factors(n, k, seed=8)
    theta0 = torch.cat([U0.detach().reshape(-1), V0.detach().reshape(-1)])

    def grad_vec(theta):
        U = theta[:n * k].reshape(n, k).clone().requires_grad_(True)
        V = theta[n * k:].reshape(n, k).clone().requires_grad_(True)
        loss_fn(U, V, A, y).backward()
        return torch.cat([U.grad.reshape(-1), V.grad.reshape(-1)])

    a = lambda t: 1.0 + 0.5 * math.sin(t) ** 2                      # positive, bounded
    T = 1.3
    N = 4000

    theta_pre = rk4(lambda t, x: -grad_vec(x) / a(t), theta0, 0.0, T, N)

    tau = 0.0                                                        # Simpson on 1/a
    h = T / N
    for i in range(N):
        t0_, t1_ = i * h, (i + 1) * h
        tau += h / 6 * (1 / a(t0_) + 4 / a(0.5 * (t0_ + t1_)) + 1 / a(t1_))
    assert abs(tau - T) > 0.05, "a(t) must be non-trivial for this test to have content"

    theta_gf = rk4(lambda t, x: -grad_vec(x), theta0, 0.0, tau, N)
    assert rel(theta_pre, theta_gf) < 1e-8

    tau_wrong = 0.0
    for i in range(N):
        t0_, t1_ = i * h, (i + 1) * h
        tau_wrong += h / 6 * (a(t0_) + 4 * a(0.5 * (t0_ + t1_)) + a(t1_))
    theta_wrong = rk4(lambda t, x: -grad_vec(x), theta0, 0.0, tau_wrong, N)
    assert rel(theta_pre, theta_wrong) > 1e-3, "the inverse time change must not also work"


def test_propa4_balancedness_conserved_by_scalar_class_broken_by_anisotropy():
    """Proposition A.4.  B = U^T U - V^T V is conserved under gradient flow and under any
    COMMON positive scalar preconditioner, and drifts once the two factors see different
    diagonal preconditioners."""
    A, y, n, k = make_problem(seed=12)
    U0, V0 = make_factors(n, k, seed=12)
    theta0 = torch.cat([U0.detach().reshape(-1), V0.detach().reshape(-1)])

    def split(theta):
        return theta[:n * k].reshape(n, k), theta[n * k:].reshape(n, k)

    def grad_vec(theta):
        U, V = split(theta)
        U = U.clone().requires_grad_(True)
        V = V.clone().requires_grad_(True)
        loss_fn(U, V, A, y).backward()
        return torch.cat([U.grad.reshape(-1), V.grad.reshape(-1)])

    def bal(theta):
        U, V = split(theta)
        return (U.T @ U - V.T @ V)

    B0 = bal(theta0)

    for a_fn in (lambda t: 1.0, lambda t: 2.0 + math.cos(t)):          # common scalars
        th = rk4(lambda t, x: -grad_vec(x) / a_fn(t), theta0, 0.0, 2.0, 3000)
        assert rel(bal(th), B0) < 1e-8, "common-scalar flow must conserve balancedness"

    D = torch.ones(2 * n * k)                                          # anisotropic control
    D[: n * k] = 3.0
    th = rk4(lambda t, x: -D * grad_vec(x), theta0, 0.0, 2.0, 3000)
    assert rel(bal(th), B0) > 1e-2, "a non-common preconditioner must break conservation"


# --------------------------------------------------------------------------------------------
# Proposition 8.1: the solvable two-timescale boundary
# --------------------------------------------------------------------------------------------

def test_prop81_greedy_closed_form_matches_numerical_integration():
    """Part (i): w(t) = s (1 + (s/w0 - 1) e^{-2 eta s t})^-1 must solve dw/dt = 2 eta w (s - w)."""
    for s, w0, eta in ((1.0, 1e-3, 0.7), (0.4, 1e-5, 2.0), (2.5, 0.1, 0.3)):
        closed = lambda t: s / (1 + (s / w0 - 1) * math.exp(-2 * eta * s * t))
        for T in (0.5, 2.0, 6.0):
            num = rk4(lambda t, x: 2 * eta * x * (s - x), torch.tensor([w0]), 0.0, T, 20000)
            assert abs(num.item() - closed(T)) / closed(T) < 1e-9


def test_prop81_greedy_tail_bound_holds_and_is_not_vacuous():
    """Part (i): at head fit T1 the tail obeys w2(T1) <= C_{delta,rho} s1 (w0/s1)^{1-rho} with
    C = 2 ((1-delta)/delta)^rho, over a grid respecting 0 < w0 <= s2/2 and 0 < delta < 1 - s2/s1."""
    eta = 1.0
    checked = 0
    for s1 in (1.0, 3.0):
        for rho in (0.2, 0.5, 0.8):
            s2 = rho * s1
            for w0 in (1e-6, 1e-4, 1e-2):
                if w0 > s2 / 2:
                    continue
                for delta in (0.05, 0.2, 0.5):
                    if not (0 < delta < 1 - rho):
                        continue
                    w = lambda t, s: s / (1 + (s / w0 - 1) * math.exp(-2 * eta * s * t))
                    T1 = math.log((1 - delta) / delta * (s1 - w0) / w0) / (2 * eta * s1)
                    assert abs(w(T1, s1) - (1 - delta) * s1) < 1e-9 * s1
                    bound = 2 * ((1 - delta) / delta) ** rho * s1 * (w0 / s1) ** (1 - rho)
                    assert w(T1, s2) <= bound + 1e-12, (s1, rho, w0, delta)
                    assert bound < 10 * s1, "bound must not be vacuously large here"
                    checked += 1
    assert checked >= 10


def test_prop81_greedy_tail_vanishes_as_init_shrinks():
    """Part (i): the transient head-tail separation sharpens without bound as w0 -> 0."""
    eta, s1, s2, delta = 1.0, 1.0, 0.5, 0.2
    prev = None
    for w0 in (1e-2, 1e-4, 1e-6, 1e-8):
        w = lambda t, s: s / (1 + (s / w0 - 1) * math.exp(-2 * eta * s * t))
        T1 = math.log((1 - delta) / delta * (s1 - w0) / w0) / (2 * eta * s1)
        tail = w(T1, s2)
        if prev is not None:
            assert tail < prev / 5, "tail at head fit must shrink with the initialisation"
        prev = tail
    assert prev < 1e-3 * s2


def test_prop81_equal_rate_fits_the_tail_before_head_fit_regardless_of_w0():
    """Part (ii): sqrt(w) grows linearly and caps at s, so w2(T1) = s2 exactly, with no
    dependence on w0 whenever delta < 1 - s2/s1."""
    eta = 1.0
    for s1 in (1.0, 4.0):
        for rho in (0.2, 0.6, 0.9):
            s2 = rho * s1
            for delta in (0.01, 0.05):
                if not (0 < delta < 1 - rho):
                    continue
                for w0 in (1e-8, 1e-4, 1e-2):
                    if w0 > s2 / 2:
                        continue
                    T1 = (math.sqrt((1 - delta) * s1) - math.sqrt(w0)) / eta
                    t2 = (math.sqrt(s2) - math.sqrt(w0)) / eta
                    # T1 must be exactly the first time the head reaches (1-delta) s1: the cap at
                    # s2 makes w2(T1) = s2 for any large enough T1, so T1 has to be pinned here
                    assert abs((math.sqrt(w0) + eta * T1) ** 2 - (1 - delta) * s1) < 1e-12 * s1
                    assert abs((math.sqrt(w0) + eta * t2) ** 2 - s2) < 1e-12 * s2
                    assert t2 < T1, "the tail must cap before the head reaches (1-delta) s1"
                    w2 = min(s2, (math.sqrt(w0) + eta * T1) ** 2)
                    assert abs(w2 - s2) < 1e-12 * s2, "tail must be fully fit, independent of w0"

    # the hypothesis delta < 1 - s2/s1 is load-bearing: violate it and the tail is NOT yet fit
    s1, s2, w0, eta = 1.0, 0.5, 1e-6, 1.0
    delta_bad = 1 - s2 / s1 + 0.2                                   # = 0.7 > 1 - rho = 0.5
    T1_bad = (math.sqrt((1 - delta_bad) * s1) - math.sqrt(w0)) / eta
    assert (math.sqrt(w0) + eta * T1_bad) ** 2 < s2, "hypothesis must be necessary"


def test_prop81_the_two_schedules_actually_disagree():
    """The proposition is only interesting if the two schedules give different tails at head
    fit: greedy leaves the tail asleep where equal-rate has already fully fit it."""
    eta, s1, s2, delta, w0 = 1.0, 1.0, 0.5, 0.2, 1e-6
    w = lambda t, s: s / (1 + (s / w0 - 1) * math.exp(-2 * eta * s * t))
    T1_greedy = math.log((1 - delta) / delta * (s1 - w0) / w0) / (2 * eta * s1)
    tail_greedy = w(T1_greedy, s2)
    tail_flat = s2
    assert tail_greedy < 0.02 * tail_flat


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
