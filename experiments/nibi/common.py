"""
Shared core for the H100 replication ladder (Appendix C8).

Ports the CPU protocols (optimizer_zoo_bias.py, muon_phase_diagram.py,
precond_dial_scalar_check.py) to device- and dtype-agnostic code.

Fidelity contract:
  * Problem instances are generated with CPU float32 generators (the same bits as the n=40 CPU
    runs), then cast to the run dtype and moved to the device; dynamics default to float64.
  * All optimizer updates are literal ports of the CPU implementations.
  * Protocol constants match the paper: loss = mean_i(<A_i,W>-y_i)^2, wd=0, interpolation bar 1e-7
    checked every 200 steps, init 1e-3, k=n, cosine decay for the constant-update-norm methods
    {muon, signum, lion} under the 'protocol' schedule or for every method under 'cosine-all'
    (Appendix C1).
  * Dial (Adam-p): denom_i = (s_i+eps)^p * (sbar+eps)^(1-p), s_i = sqrt(vhat_i),
    sbar = sqrt(mean vhat) (RMS, gauge-invariant); p=1 is stock Adam, p=0 is scalar-Adam.

Results are appended as JSON lines; every driver is resume-safe.
"""
from __future__ import annotations

import json
import math
import os
import socket
import time

import numpy as np
import torch

EPS = 1e-8
TRAIN_TOL = 1e-7
LOOSE_TOL = 1e-4
DECAYED_PROTOCOL = {'muon', 'signum', 'lion'}

ZOO9 = ['gd', 'adam_p0rms', 'muon', 'shampoo',
        'adam', 'rmsprop', 'signum', 'lion', 'adafactor']
EQUIVARIANT = {'gd', 'adam_p0rms', 'muon', 'shampoo', 'scaledgd'}


def pick_device(arg: str = 'auto') -> torch.device:
    if arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(arg)


def dtype_of(name: str) -> torch.dtype:
    return {'float64': torch.float64, 'float32': torch.float32}[name]


def make_sensing(seed, n, r, mdof, tau=None, device='cpu', dtype=torch.float64):
    """tau=None -> zoo generation path (restoration_probe.make_problem);
    tau=float (incl. 0.0) -> tail generation path (muon_phase_diagram.make_tail_problem).
    The two paths consume the generator differently, exactly as the local scripts do."""
    dof = r * (2 * n - r)
    m = int(round(mdof * dof))
    g = torch.Generator().manual_seed(seed)
    Us = torch.randn(n, r, generator=g) / np.sqrt(r)
    Vs = torch.randn(n, r, generator=g) / np.sqrt(r)
    X3 = Us @ Vs.T
    if tau is None:
        X = X3
    else:
        E = torch.randn(n, n, generator=g)
        Qu, _ = torch.linalg.qr(Us)
        Qv, _ = torch.linalg.qr(Vs)
        E = E - Qu @ (Qu.T @ E)
        E = E - (E @ Qv) @ Qv.T
        E = E / (E.norm() + 1e-12) * X3.norm()
        X = math.sqrt(1 - tau ** 2) * X3 + tau * E
    A = torch.randn(m, n * n, generator=g)
    X = X.to(device=device, dtype=dtype)
    A = A.to(device=device, dtype=dtype)
    y = A @ X.reshape(-1)
    return X, A, y


def factors(seed, n, k, init, device='cpu', dtype=torch.float64):
    g = torch.Generator().manual_seed(seed + 7)
    U = (torch.randn(n, k, generator=g) * init).to(device=device, dtype=dtype)
    V = (torch.randn(n, k, generator=g) * init).to(device=device, dtype=dtype)
    U.requires_grad_(True)
    V.requires_grad_(True)
    return U, V


def stats(U, V, X):
    with torch.no_grad():
        W = U @ V.T
        rec = ((W - X).norm() / (X.norm() + 1e-30)).item()
        sv = torch.linalg.svdvals(W)
        nuc = sv.sum().item()
        p = sv / (sv.sum() + 1e-12)
        er = torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()
        bal = (U.T @ U - V.T @ V).norm().item()
    return dict(rec=rec, nuc=nuc, er=er, bal=bal)


def newton_schulz(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)
    transpose = G.shape[0] > G.shape[1]
    if transpose:
        X = X.T
    for _ in range(steps):
        AA = X @ X.T
        B = b * AA + c * (AA @ AA)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X


def matrix_pow(Mx, p, eps=1e-6):
    Mx = 0.5 * (Mx + Mx.T)
    vals, vecs = torch.linalg.eigh(Mx)
    vals = torch.clamp(vals, min=eps)
    return (vecs * vals.pow(p)) @ vecs.T


class Stepper:
    """Literal ports of the local update rules; state on the params' device/dtype."""

    def __init__(self, kind, params, p_pow=None):
        self.kind = kind
        self.params = params
        self.p_pow = p_pow
        z = lambda p: torch.zeros_like(p)
        self.mom = [z(p) for p in params]
        self.v = [z(p) for p in params]
        self.Lp = [torch.zeros(p.shape[0], p.shape[0], device=p.device, dtype=p.dtype)
                   for p in params]
        self.Rp = [torch.zeros(p.shape[1], p.shape[1], device=p.device, dtype=p.dtype)
                   for p in params]
        self.Linv = [None for _ in params]
        self.Rinv = [None for _ in params]
        self.Rrow = [torch.zeros(p.shape[0], device=p.device, dtype=p.dtype) for p in params]
        self.Ccol = [torch.zeros(p.shape[1], device=p.device, dtype=p.dtype) for p in params]
        self.eye = [torch.eye(p.shape[0], device=p.device, dtype=p.dtype) for p in params]
        self.eyeR = [torch.eye(p.shape[1], device=p.device, dtype=p.dtype) for p in params]

    @torch.no_grad()
    def step(self, lr_t, t):
        kind, params = self.kind, self.params
        b1, b2 = 0.9, 0.999
        if kind == 'adam_p0rms' or (kind == 'adam_p' and self.p_pow is not None):
            p_pow = 0.0 if kind == 'adam_p0rms' else float(self.p_pow)
            tot, cnt = 0.0, 0
            mh_l, vh_l = [], []
            for i, p in enumerate(params):
                G = p.grad
                self.mom[i].mul_(b1).add_(G, alpha=1 - b1)
                self.v[i].mul_(b2).addcmul_(G, G, value=1 - b2)
                vh = self.v[i] / (1 - b2 ** t)
                mh = self.mom[i] / (1 - b1 ** t)
                tot += vh.sum().item()
                cnt += vh.numel()
                mh_l.append(mh)
                vh_l.append(vh)
            sbar = math.sqrt(tot / cnt)
            if p_pow <= 0.0:
                s = sbar + EPS
                for i, p in enumerate(params):
                    p.add_(mh_l[i], alpha=-lr_t / s)
            else:
                for i, p in enumerate(params):
                    denom = (vh_l[i].sqrt() + EPS).pow(p_pow) * (sbar + EPS) ** (1 - p_pow)
                    p.add_(-lr_t * mh_l[i] / denom)
            return
        for i, p in enumerate(params):
            G = p.grad
            if kind == 'gd':
                p.add_(G, alpha=-lr_t)
            elif kind == 'adam':
                self.mom[i].mul_(b1).add_(G, alpha=1 - b1)
                self.v[i].mul_(b2).addcmul_(G, G, value=1 - b2)
                mh = self.mom[i] / (1 - b1 ** t)
                vh = self.v[i] / (1 - b2 ** t)
                p.addcdiv_(mh, vh.sqrt() + EPS, value=-lr_t)
            elif kind == 'rmsprop':
                self.v[i].mul_(0.99).addcmul_(G, G, value=0.01)
                p.addcdiv_(G, self.v[i].sqrt() + EPS, value=-lr_t)
            elif kind == 'signum':
                self.mom[i].mul_(b1).add_(G, alpha=1 - b1)
                p.add_(torch.sign(self.mom[i]), alpha=-lr_t)
            elif kind == 'lion':
                upd = torch.sign(b1 * self.mom[i] + (1 - b1) * G)
                self.mom[i].mul_(0.99).add_(G, alpha=0.01)
                p.add_(upd, alpha=-lr_t)
            elif kind == 'adafactor':
                g2 = G * G
                self.Rrow[i].mul_(b2).add_(g2.mean(dim=1), alpha=1 - b2)
                self.Ccol[i].mul_(b2).add_(g2.mean(dim=0), alpha=1 - b2)
                vhat = torch.outer(self.Rrow[i], self.Ccol[i]) / (self.Rrow[i].mean() + 1e-30)
                p.addcdiv_(G, vhat.sqrt() + EPS, value=-lr_t)
            elif kind == 'muon':
                self.mom[i].mul_(b1).add_(G)
                p.add_(newton_schulz(self.mom[i]), alpha=-lr_t)
            elif kind == 'shampoo':
                self.Lp[i] += G @ G.T
                self.Rp[i] += G.T @ G
                if (t - 1) % 20 == 0 or self.Linv[i] is None:
                    self.Linv[i] = matrix_pow(self.Lp[i] + 1.0 * self.eye[i], -0.25)
                    self.Rinv[i] = matrix_pow(self.Rp[i] + 1.0 * self.eyeR[i], -0.25)
                p.add_(self.Linv[i] @ G @ self.Rinv[i], alpha=-lr_t)
            elif kind == 'scaledgd':
                U, V = params[0], params[1]
                other = V if i == 0 else U
                gram = other.T @ other
                lam = 1e-4 * (gram.diagonal().mean().item() + 1e-30) + 1e-12
                inv = torch.linalg.inv(gram + lam * torch.eye(
                    gram.shape[0], device=gram.device, dtype=gram.dtype))
                p.add_(G @ inv, alpha=-lr_t)
            else:
                raise ValueError(f'unknown optimizer kind {kind}')


def run_sensing(kind, seed, n, r, mdof, lr, max_steps, tau=None, k=None, init=1e-3,
                schedule='protocol', p_pow=None, tol=TRAIN_TOL, check_every=200,
                device='cpu', dtype=torch.float64, decay_horizon=None):
    """One training run; returns final metrics. Semantics match the local scripts:
    train is the pre-update loss of the last executed step; early stop when the
    (checked every `check_every`) train loss dips under `tol`."""
    device = torch.device(device)
    X, A, y = make_sensing(seed, n, r, mdof, tau=tau, device=device, dtype=dtype)
    U, V = factors(seed, n, k or n, init, device=device, dtype=dtype)
    params = [U, V]
    stp = Stepper(kind, params, p_pow=p_pow)
    decayed = (schedule == 'cosine-all') or (schedule == 'protocol' and kind in DECAYED_PROTOCOL)
    H = decay_horizon or max_steps
    train = float('inf')
    t0 = time.time()
    step = 0
    for step in range(1, max_steps + 1):
        for p in params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        W = (U @ V.T).reshape(-1)
        loss = ((A @ W - y) ** 2).mean()
        train = loss.item()
        if not np.isfinite(train) or train > 1e8:
            return dict(rec=float('nan'), nuc=float('nan'), er=float('nan'),
                        bal=float('nan'), train=train, steps=step,
                        secs=time.time() - t0, status='diverged')
        loss.backward()
        lr_t = lr * 0.5 * (1 + math.cos(math.pi * step / H)) if decayed else lr
        stp.step(lr_t, step)
        if step % check_every == 0 and train < tol:
            break
    out = stats(U, V, X)
    status = 'interp' if train < tol else ('loose' if train < LOOSE_TOL else 'n/i')
    out.update(train=train, steps=step, secs=round(time.time() - t0, 2), status=status)
    return out


class Sink:
    """Append-only JSONL results with resume support."""

    def __init__(self, path, meta):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.done = set()
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if 'key' in rec:
                            self.done.add(rec['key'])
                    except json.JSONDecodeError:
                        pass
        meta = dict(meta, host=socket.gethostname(), torch=torch.__version__,
                    cuda=torch.cuda.is_available(), ts=time.strftime('%F %T'))
        self._write(dict(key=f'__meta__{time.time()}', kind='meta', **meta))

    def _write(self, rec):
        rec = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
               for k, v in rec.items()}
        with open(self.path, 'a') as f:
            f.write(json.dumps(rec, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())

    def has(self, key):
        return key in self.done

    def add(self, key, **rec):
        self._write(dict(key=key, **rec))
        self.done.add(key)


def load_cells(path):
    cells = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                rec = {k: (float('nan') if v is None else v) for k, v in rec.items()}
            except json.JSONDecodeError:
                continue
            if rec.get('kind') != 'meta' and not str(rec.get('key', '')).startswith('__meta__'):
                cells.append(rec)
    return cells


def select_lr(cells, prefer='rec'):
    """Local best(): among lrs whose mean train < 1e-4, lowest mean recovery;
    else lowest (rec + 10 + train). cells: list of dicts with lr, rec, train."""
    by_lr = {}
    for c in cells:
        by_lr.setdefault(c['lr'], []).append(c)
    best_lr, best_score = None, None
    for lr, rows in sorted(by_lr.items()):
        rec = float(np.mean([r['rec'] for r in rows]))
        tr = float(np.mean([r['train'] for r in rows]))
        if not (np.isfinite(rec) and np.isfinite(tr)):
            continue
        score = rec if tr < LOOSE_TOL else rec + 10 + tr
        if best_score is None or score < best_score:
            best_lr, best_score = lr, score
    return best_lr
