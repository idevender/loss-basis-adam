"""Figure 5: the matched-training-loss trajectory on Indian Pines (Section 9).

Reads the file behind Table 5 (experiments/nibi_results/indianpines_gpu.jsonl), re-applies the
train-only lr rule of Appendix D.6, and asserts the published numbers before plotting. If an
assertion fires, no figure is written.

Usage: python figures/make_realdata_trajectory.py
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_figures import C, INK, MUTE, FAINT, _save  # noqa: E402  (shared paper style)

RESULTS = os.path.join(HERE, "..", "experiments", "nibi_results", "indianpines_gpu.jsonl")
LEVELS = ["0.003", "0.001", "0.0003", "0.0001", "3e-05", "1e-05"]
DENS = 0.15                      # the m/dof_24 ~ 1.15 cell narrated in Section 9
METHODS = ["gd", "adam", "muon"]  # exactly the three the text discusses
SEL_SEEDS = [42, 123]            # Appendix D.6: the lr is chosen on these two seeds only
EVAL_SEEDS = [42, 123, 456, 789]  # all four are plotted, none of them selects the lr


def load():
    rows = []
    with open(RESULTS) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("kind") == "meta" or r.get("dens") != DENS:
                continue
            rows.append(r)
    return rows


def select_lr(rows, opt):
    """Appendix D.6's train-only rule: deepest matched train level reached by every seed,
    ties broken by fewest steps.  SEL_SEEDS only, so 456 and 789 stay held out."""
    by_lr = {}
    for r in rows:
        if r["opt"] != opt or r["seed"] not in SEL_SEEDS:
            continue
        by_lr.setdefault(r["lr"], []).append(r)
    best, best_key = None, None
    for lr, rs in by_lr.items():
        depth = 0
        for i, lev in enumerate(LEVELS):
            if all(lev in r["hits"] for r in rs):
                depth = i + 1
        if depth == 0:
            continue
        deepest = LEVELS[depth - 1]
        steps = np.mean([r["hits"][deepest][2] for r in rs])
        key = (depth, -steps)
        if best_key is None or key > best_key:
            best, best_key = lr, key
    return best


def curve(rows, opt, lr, idx):
    """Per-level mean and s.d. over seeds of (held-out RMSE, effective rank).

    The s.d. is the sample one (ddof=1), the convention Table 5 prints; ddof=0 would draw a band
    a factor sqrt(3)/2 narrower than the +-0.013 / +-0.047 / +-0.082 the table quotes."""
    rs = [r for r in rows if r["opt"] == opt and r["lr"] == lr and r["seed"] in EVAL_SEEDS]
    rmse_m, rmse_s, rank_m, n = [], [], [], []
    for lev in LEVELS:
        vals = [r["hits"][lev] for r in rs if lev in r["hits"]]
        if not vals:
            rmse_m.append(np.nan); rmse_s.append(np.nan); rank_m.append(np.nan); n.append(0)
            continue
        rmse_m.append(float(np.mean([v[0] for v in vals])))
        rmse_s.append(float(np.std([v[0] for v in vals], ddof=1)) if len(vals) > 1 else 0.0)
        rank_m.append(float(np.mean([v[1] for v in vals])))
        n.append(len(vals))
    return np.array(rmse_m), np.array(rmse_s), np.array(rank_m), n


def close(a, b, tol):
    return abs(a - b) <= tol


def main():
    rows = load()
    lrs = {m: select_lr(rows, m) for m in METHODS}

    # ---- guard 1: the train-only rule must reproduce the rates recorded in SUMMARY.md ----
    assert lrs == {"gd": 30.0, "adam": 0.01, "muon": 0.1}, f"selection changed: {lrs}"
    for m in METHODS:  # the two selection seeds must be present, and the plot must add the other two
        sel = {r["seed"] for r in rows if r["opt"] == m and r["seed"] in SEL_SEEDS}
        assert sel == set(SEL_SEEDS), f"{m}: selection seeds {sorted(sel)}"
        ev = {r["seed"] for r in rows if r["opt"] == m and r["lr"] == lrs[m]}
        assert ev == set(EVAL_SEEDS), f"{m}: evaluation seeds {sorted(ev)}"

    data = {m: curve(rows, m, lrs[m], i) for i, m in enumerate(METHODS)}
    for m in METHODS:
        assert all(k == 4 for k in data[m][3]), f"{m}: expected 4 seeds at every level"

    gd, adam, muon = data["gd"][0], data["adam"][0], data["muon"][0]
    gdr, adamr, muonr = data["gd"][2], data["adam"][2], data["muon"][2]
    L = {lev: i for i, lev in enumerate(LEVELS)}

    # ---- guard 2: every number Section 9 and Table 5 already print ----
    # Table 5, Indian Pines m/dof ~ 1.15, matched train loss <= 1e-5
    assert close(gd[L["1e-05"]], 0.01481, 5e-6), gd[L["1e-05"]]
    assert close(adam[L["1e-05"]], 0.02600, 5e-6), adam[L["1e-05"]]
    assert close(muon[L["1e-05"]], 0.03397, 5e-6), muon[L["1e-05"]]
    assert close(round(muonr[L["1e-05"]]), 36, 0), muonr[L["1e-05"]]
    # ...and the band drawn around them is the same dispersion the table prints (x 10^-2)
    for m, sd_tab in (("gd", 0.013), ("adam", 0.047), ("muon", 0.082)):
        assert close(data[m][1][L["1e-05"]] * 100, sd_tab, 5e-4), (m, data[m][1][L["1e-05"]])
    red = (adam[L["1e-05"]] - gd[L["1e-05"]]) / adam[L["1e-05"]] * 100
    assert close(red, 43.0, 0.06), red
    # Section 9 narrative, matched train loss <= 3e-5
    assert close(gd[L["3e-05"]], 0.0150, 5e-5), gd[L["3e-05"]]
    assert close(adam[L["3e-05"]], 0.0268, 5e-5), adam[L["3e-05"]]
    assert close(round(gdr[L["3e-05"]]), 11, 0), gdr[L["3e-05"]]
    assert close(round(adamr[L["3e-05"]]), 28, 0), adamr[L["3e-05"]]
    # Section 9: "Adam's held-out error rises as it interpolates, .0251 -> .0268, rank 22 -> 28"
    assert close(adam[L["0.0003"]], 0.0251, 5e-5), adam[L["0.0003"]]
    assert close(round(adamr[L["0.0003"]]), 22, 0), adamr[L["0.0003"]]
    assert adam[L["3e-05"]] > adam[L["0.0003"]], "the claimed rise is not in the data"
    # Section 9: "GD's held-out error falls monotonically as it fits deeper"
    assert all(np.diff(gd) < 0), "GD is not monotone in this run"
    # Section 9: Muon's rank pinned near the model cap, 46 -> 36
    assert close(round(muonr[L["0.003"]]), 46, 0) and close(round(muonr[L["1e-05"]]), 36, 0)
    assert muonr.min() > 35 and muonr.max() < 48

    # ---------------------------------- plot ----------------------------------
    x = np.arange(len(LEVELS))
    labels = [r"$3{\times}10^{-3}$", r"$10^{-3}$", r"$3{\times}10^{-4}$",
              r"$10^{-4}$", r"$3{\times}10^{-5}$", r"$10^{-5}$"]
    style = {"gd": ("GD", C["gd"], "-", "o"),
             "adam": ("Adam", C["adam"], "-", "s"),
             "muon": ("Muon", C["muon"], "-", "^")}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.45))

    for m in METHODS:
        lab, col, ls, mk = style[m]
        mu, sd, _, _ = data[m]
        ax1.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
        ax1.plot(x, mu, ls, color=col, marker=mk, ms=3.6, lw=1.6, label=lab)
    ax1.set_ylabel("held-out RMSE")
    ax1.set_xlabel("matched train loss (fitting deeper $\\rightarrow$)")
    ax1.legend(frameon=False, fontsize=8.5, loc="lower left")   # the only empty quadrant here
    ax1.annotate("Adam's error rises\nas it fits deeper",
                 xy=(3, adam[3]), xytext=(1.35, 0.0385),
                 fontsize=8.5, color=C["adam"], ha="left",
                 arrowprops=dict(arrowstyle="->", color=C["adam"], lw=0.9,
                                 shrinkA=0, shrinkB=3))

    for m in METHODS:
        lab, col, ls, mk = style[m]
        ax2.plot(x, data[m][2], ls, color=col, marker=mk, ms=3.6, lw=1.6, label=lab)
    # reference lines labelled at the right edge, where the three curves are furthest apart;
    # a white halo keeps the text legible where it grazes Adam's line
    halo = dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.9)
    ax2.axhline(48, color=FAINT, ls=(0, (4, 3)), lw=0.9)
    ax2.text(5.22, 49.2, "model rank cap (48)", fontsize=8, color=MUTE, ha="right")
    ax2.axhline(24, color=FAINT, ls=(0, (1, 2.5)), lw=0.9)
    ax2.text(5.22, 20.0, "intrinsic rank (24)", fontsize=8, color=MUTE, ha="right", bbox=halo)
    ax2.set_ylabel("effective rank of solution")
    ax2.set_xlabel("matched train loss (fitting deeper $\\rightarrow$)")
    ax2.set_ylim(0, 56)

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlim(-0.3, len(LEVELS) - 0.7)

    fig.tight_layout(w_pad=1.6)
    _save(fig, "realdata_trajectory")
    print("wrote realdata_trajectory.pdf   selected lrs:", lrs)
    for m in METHODS:
        print(f"  {m:5s} rmse {np.round(data[m][0], 5)}  rank {np.round(data[m][2], 1)}")


if __name__ == "__main__":
    main()
