"""
Generate the paper figures.

Renders the attention-gauge, dial, phase-diagram, and zoo-map figures as vector PDFs, using a
colorblind-safe palette with one fixed hue per entity and serif typography matched to the paper body.
The plotted numbers are the final experiment outputs; the dial uses the RMS
(exactly-equivariant-at-p=0) convention.

Run: python make_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

C = dict(gd="#000000", adam="#D55E00", muon="#0072B2", shampoo="#009E73",
         scalar="#56B4E9", sgd="#8C8C8C", rmsprop="#E69F00", adafactor="#CC79A7",
         lion="#C7A100", signum="#7A5C00")
EQ_FILL, CW_FILL = "#0072B2", "#D55E00"
INK, MUTE, FAINT = "#1A1A1A", "#5A5A5A", "#9A9A9A"

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.edgecolor": "#3C3C3C", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E8E8E8", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#3C3C3C", "ytick.color": "#3C3C3C",
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def _save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    if os.environ.get("FIG_PNG"):
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)


def fig_zoo():
    rows = [
        ("Muon",              0.0000, True),
        ("GD",                0.1312, True),
        ("scalar-Adam ($p{=}0$)", 0.2010, True),
        ("Shampoo",           0.2856, True),
        ("Lion",              0.4248, False),
        ("signSGD-m",         0.4454, False),
        ("RMSProp",           0.5270, False),
        ("Adafactor",         0.5430, False),
        ("Adam",              0.5734, False),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, rec, eq) in zip(y, rows):
        col = EQ_FILL if eq else CW_FILL
        ax.barh(yi, rec, height=0.66, color=col, alpha=0.92,
                edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(rec + 0.010, yi, "0.00005" if rec == 0 else f"{rec:.3f}",
                va="center", ha="left", fontsize=8.6, color=MUTE, zorder=4)

    ax.axvspan(0.286, 0.425, color="#F1F1F1", zorder=0)
    ax.plot([0.3555, 0.3555], [-0.6, 8.6], color=FAINT, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.text(0.3555, 4.5, "empty gap", ha="center", va="center", fontsize=8.4,
            color=MUTE, style="italic",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    pos_eq = {int(yi): eq for yi, (_, _, eq) in zip(y, rows)}
    for tl in ax.get_yticklabels():
        tl.set_color(EQ_FILL if pos_eq[int(round(tl.get_position()[1]))] else CW_FILL)
        tl.set_fontsize(9.5)
    ax.set_xlabel(r"ground-truth recovery error $\|W-X^\ast\|_F/\|X^\ast\|_F$"
                  "\n(lower = low-rank bias preserved)")
    ax.set_xlim(0, 0.72)
    ax.set_ylim(-0.7, 8.9)
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0, axis="y")

    leg = [Patch(fc=EQ_FILL, ec="white", label="gauge-equivariant"),
           Patch(fc=CW_FILL, ec="white", label="coordinate-wise")]
    ax.legend(handles=leg, loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=True,
              framealpha=0.96, edgecolor="#DDDDDD", fontsize=8.2, borderpad=0.6,
              handlelength=1.1)
    fig.tight_layout()
    _save(fig, "zoo_map")


def fig_attention():
    steps = np.array([0, 1, 10, 50, 100, 300, 600, 1000, 1500])
    x = steps + 1
    curves = [
        ("Adam, gauge twin",          C["adam"],   "-",  "o",
            [1.8e-7, 3.6e-3, 3.7e-2, 3.4e-1, 6.5e-1, 7.2e-1, 7.4e-1, 7.6e-1, 7.7e-1]),
        ("Adam, noise twin ($10^{-7}$)", C["adam"], ":",  "s",
            [2.6e-7, 2.9e-7, 6.5e-7, 3.3e-6, 1.7e-5, 1.2e-5, 1.4e-5, 1.5e-5, 1.6e-5]),
        ("Muon, gauge twin",          C["muon"],   "-",  "o",
            [1.8e-7, 2.2e-7, 4.9e-7, 3.9e-6, 2.3e-2, 3.2e-1, 5.8e-1, 7.5e-1, 8.5e-1]),
        ("Muon, noise twin ($10^{-7}$)", C["muon"], ":",  "s",
            [2.6e-7, 2.9e-7, 7.0e-7, 8.8e-6, 4.0e-2, 3.4e-1, 5.9e-1, 7.4e-1, 8.7e-1]),
        ("SGD, gauge twin",           C["sgd"],    "-",  "^",
            [1.8e-7, 2.9e-7, 3.3e-7, 1.3e-5, 3.7e-5, 2.4e-5, 2.1e-5, 2.1e-5, 2.0e-5]),
        ("scalar-Adam, gauge twin",   C["scalar"], "-",  "D",
            [1.8e-7, 1.6e-7, 3.1e-7, 5.8e-6, 4.1e-6, 3.8e-6, 4.0e-6, 4.1e-6, 4.3e-6]),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for lab, col, ls, mk, ys in curves:
        ax.plot(x, ys, ls, color=col, lw=1.8, label=lab,
                marker=mk, ms=3.4, mew=0, alpha=0.95)

    ax.annotate("structural split:\none step, $10^{4}\\times$ the noise twin",
                xy=(2, 3.6e-3), xytext=(2.4, 5.0e-1), fontsize=8.2, color=C["adam"],
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=C["adam"], lw=1.0,
                                connectionstyle="arc3,rad=0.25"))
    ax.annotate("chaos: Muon's gauge and\nnoise twins trace one curve",
                xy=(101, 2.3e-2), xytext=(200, 2.4e-3), fontsize=8.2, color=C["muon"],
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=C["muon"], lw=1.0,
                                connectionstyle="arc3,rad=0.3"))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1, 2, 11, 101, 1001])
    ax.set_xticklabels(["0", "1", "10", "100", "1000"])
    ax.set_xlabel("training step")
    ax.set_ylabel("relative logit distance between twins")
    ax.set_ylim(6e-8, 4)
    ax.set_xlim(0.9, 2400)
    ax.legend(fontsize=8.0, frameon=False, loc="upper left",
              bbox_to_anchor=(1.005, 1.02), labelspacing=0.4, handlelength=1.9)
    fig.tight_layout()
    _save(fig, "attention_gauge")


def fig_phase():
    taus = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.5])
    rec = dict(
        gd=[0.1125, 0.1503, 0.2144, 0.3512, 0.5488, 0.7274],
        adam=[0.5419, 0.5427, 0.5552, 0.5968, 0.6849, 0.7758],
        muon=[0.0000, 0.0946, 0.1912, 0.3538, 0.5802, 0.7516],
        shampoo=[0.3340, 0.3428, 0.3750, 0.4488, 0.5763, 0.7233],
    )
    std = dict(
        gd=[0.028, 0.025, 0.023, 0.022, 0.024, 0.029],
        adam=[0.045, 0.048, 0.049, 0.043, 0.041, 0.033],
        muon=[0.000, 0.003, 0.007, 0.012, 0.024, 0.022],
        shampoo=[0.083, 0.083, 0.096, 0.090, 0.039, 0.035],
    )
    names = dict(gd="GD", adam="Adam", muon="Muon", shampoo="Shampoo")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for k in ("adam", "shampoo", "gd", "muon"):
        r, s = np.array(rec[k]), np.array(std[k])
        ax.fill_between(taus, r - s, r + s, color=C[k], alpha=0.12, lw=0)
        ax.plot(taus, r, "-", color=C[k], lw=1.9, marker="o", ms=3.8, mew=0,
                label=names[k])

    ax.axvline(0.2, color=MUTE, lw=0.9, ls=(0, (4, 3)))
    ax.text(0.188, 0.135, r"$\tau^\ast\!\approx\!0.2$" "\n(4% tail energy)",
            fontsize=8.4, color=MUTE, ha="right", va="center")
    ax.annotate("Muon cedes to GD", xy=(0.2, 0.354), xytext=(0.315, 0.235),
                fontsize=8.2, color=C["muon"], ha="center",
                arrowprops=dict(arrowstyle="->", color=C["muon"], lw=1.0,
                                connectionstyle="arc3,rad=-0.25"))
    ax.annotate("coordinate-wise baseline:\nAdam highest at every $\\tau$",
                xy=(0.24, 0.61), xytext=(0.145, 0.86), fontsize=8.2, color=C["adam"],
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=C["adam"], lw=1.0,
                                connectionstyle="arc3,rad=-0.2"))

    ax.set_xlabel(r"target tail fraction $\tau$   ($\tau^2$ = fraction of energy off the rank-3 subspace)")
    ax.set_ylabel("recovery error")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.01, 0.52)
    ax.legend(fontsize=8.6, frameon=False, loc="upper left", handlelength=1.6,
              labelspacing=0.3, borderaxespad=0.2)
    fig.tight_layout()
    _save(fig, "phase_diagram")


def fig_dial():
    p = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
    er = np.array([14.5, 11.1, 8.2, 6.4, 5.4])
    rc = [0.570, 0.459, 0.348, 0.260, 0.201]
    fig, ax = plt.subplots(figsize=(4.6, 3.5))

    _wbox = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9)
    ax.axhline(4.51, color=C["gd"], lw=1.1, ls=(0, (5, 3)), zorder=1)
    ax.text(0.80, 4.62, "GD  (4.51)", fontsize=8.4, color=C["gd"],
            ha="left", va="bottom", bbox=_wbox, zorder=5)
    ax.axhline(3.0, color=FAINT, lw=1.0, ls=(0, (1, 2)), zorder=1)
    ax.text(0.80, 3.10, "planted rank  (3)", fontsize=8.4, color=MUTE,
            ha="left", va="bottom", bbox=_wbox, zorder=5)

    ax.plot(p, er, "-", color=C["adam"], lw=2.0, zorder=3)
    ax.scatter(p, er, s=34, color=C["adam"], zorder=4, edgecolor="white", linewidth=0.7)

    ax.annotate(f"Adam ($p{{=}}1$)\nrec {rc[0]:.3f}", xy=(1.0, 14.5),
                xytext=(0.72, 15.4), fontsize=8.4, color=C["adam"], ha="center", va="top")
    ax.annotate(f"scalar-Adam ($p{{=}}0$)\nrec {rc[-1]:.3f}", xy=(0.0, 5.4),
                xytext=(0.16, 7.6), fontsize=8.4, color=C["adam"], ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color=C["adam"], lw=0.9,
                                connectionstyle="arc3,rad=0.2"))

    ax.set_xlabel(r"preconditioner anisotropy $p$")
    ax.set_ylabel("effective rank of solution")
    ax.set_xticks([1.0, 0.75, 0.5, 0.25, 0.0])
    ax.set_xticklabels(["1\n(Adam)", "0.75", "0.5", "0.25", "0\n(scalar)"])
    ax.set_xlim(1.08, -0.12)
    ax.set_ylim(2.2, 15.6)
    fig.tight_layout()
    _save(fig, "dial")


if __name__ == "__main__":
    fig_zoo(); fig_attention(); fig_phase(); fig_dial()
    print("wrote figure PDFs to", OUT,
          "(set FIG_PNG=1 for raster QA copies)")
