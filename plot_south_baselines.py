"""plot_south_baselines.py - Baseline skill vs. target instant q, South region.

Replaces the two full-grid South tables in the body of paper_datasets_short.tex
(the tables themselves move to the appendix). One figure, three panels sharing
the q axis: regression WI, regression R^2, classification AUC-ROC (kappa=-2.0).
Each learned model is drawn as the mean over p in {3,6,9,12} with a min-max band
across p; persistence has no fitted parameters and is identical for every p, so
it is a single line.

    python plot_south_baselines.py   ->   images/south_baselines_vs_q.pdf/.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
Q = [1, 3, 6, 9, 12]
P = [3, 6, 9, 12]

# Reference categorical palette, slots 1-3 (fixed order), plus a distinct
# marker and dash per series so identity never rests on color alone.
STYLE = {
    "persistence": dict(color="#2a78d6", marker="s", ls="-", label="Persistence"),
    "linear": dict(color="#eb6834", marker="^", ls="--", label="Linear / logistic regression"),
    "random_forest": dict(color="#1baf7a", marker="o", ls="-.", label="Random forest"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"


def load(task):
    rows = json.load(open(HERE / f"baselines_{task}_Sul.json"))
    return {(r["p"], r["q"]): r["results"] for r in rows}


def series(grid, method, metric):
    """(mean, min, max) over p for each q."""
    vals = np.array([[grid[(p, q)][method][metric] for p in P] for q in Q])
    return vals.mean(1), vals.min(1), vals.max(1)


def draw(ax, grid, metric, methods, ref=None, ref_label=None, label_at=(7.5, "bottom")):
    for key, method in methods:
        st = STYLE[key]
        mean, lo, hi = series(grid, method, metric)
        if key != "persistence":
            ax.fill_between(Q, lo, hi, color=st["color"], alpha=0.15, lw=0)
        ax.plot(Q, mean, color=st["color"], ls=st["ls"], lw=2, marker=st["marker"],
                ms=6, mec="white", mew=1.2, label=st["label"], zorder=3)
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=1, ls=":", zorder=1)
        x, va = label_at
        ax.text(x, ref + (0.01 if va == "bottom" else -0.01) * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                ref_label, color=MUTED, fontsize=8, va=va, ha="center")
    ax.set_xticks(Q)
    ax.set_xlabel("Target instant $q$ (months)", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def main():
    reg, clf = load("regression"), load("classification")
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)

    draw(axes[0], reg, "wi", [("persistence", "persistence"), ("linear", "linear_regression"),
                              ("random_forest", "random_forest")])
    axes[0].set_title("(a) Continuous: Willmott's Index", fontsize=10, color=INK, loc="left")
    axes[0].set_ylim(0.2, 0.9)

    draw(axes[1], reg, "r2", [("persistence", "persistence"), ("linear", "linear_regression"),
                              ("random_forest", "random_forest")],
         ref=0.0, ref_label="test-period mean")
    axes[1].set_title("(b) Continuous: $R^2$", fontsize=10, color=INK, loc="left")

    draw(axes[2], clf, "auc_roc", [("persistence", "persistence"), ("linear", "logistic_regression"),
                                   ("random_forest", "random_forest")],
         ref=0.5, ref_label="chance", label_at=(4.5, "top"))
    axes[2].set_title(r"(c) Binary, $\kappa=-2.0$: AUC-ROC", fontsize=10, color=INK, loc="left")
    axes[2].set_ylim(0.3, 0.9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out = HERE / "images" / "south_baselines_vs_q"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved {out}.pdf/.png")


if __name__ == "__main__":
    main()
