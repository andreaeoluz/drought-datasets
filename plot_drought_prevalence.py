"""plot_drought_prevalence.py - Drought prevalence and domain-level occurrence.

Replaces the drought-severity table in the body of paper_datasets_short.tex
(the table itself moves to the appendix). One column per severity threshold
kappa; regions ordered by number of valid pixels.
  Top row:    pixel-level drought prevalence over the full record, with the
              standard-normal expectation Phi(kappa) as a dashed reference.
  Bottom row: domain-level drought occurrence (fraction of target months with
              at least one drought pixel), mean over the 20 (p, q) pairs; the
              range across (p, q) is at most 1.6 points, so it is not drawn.

    python plot_drought_prevalence.py           ->   images/drought_prevalence.pdf/.png
    python plot_drought_prevalence.py --slide   ->   prevalence-only row, deck colors and
                                                     sans font, written into the Beamer deck's figs/
"""
import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
KAPPAS = [-1.0, -1.5, -2.0]
# Ordered by number of valid pixels (Table "Study domains").
REGIONS = [("Sul", "S", 1642), ("Sudeste", "SE", 4778),
           ("Centro-Oeste", "CW", 6594), ("Norte", "N", 6994),
           ("Nordeste", "NE", 7766)]
BAR = "#2a78d6"                       # reference palette, categorical slot 1
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"


def load():
    stats = {r["region"]: r["per_pqk"] for r in json.load(open(HERE / "dataset_statistics_binary.json"))}
    prev, occ = {}, {}
    for pt, _, _ in REGIONS:
        for k in KAPPAS:
            items = [v for key, v in stats[pt].items() if key.endswith(f"_k{k}")]
            prev[pt, k] = items[0]["drought_prevalence"] * 100
            occ[pt, k] = np.mean([v["drought_ratio"] for v in items]) * 100
    return prev, occ


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def main():
    prev, occ = load()
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True)
    x = np.arange(len(REGIONS))
    labels = [f"{en}\n({n:,})" for _, en, n in REGIONS]

    for j, k in enumerate(KAPPAS):
        ax = axes[0, j]
        vals = [prev[pt, k] for pt, _, _ in REGIONS]
        ax.bar(x, vals, width=0.62, color=BAR, edgecolor="white", linewidth=2, zorder=2)
        phi = norm.cdf(k) * 100
        ax.axhline(phi, color=INK, lw=1.2, ls="--", zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.set_title(rf"$\kappa = {k}$   ($\Phi(\kappa)$ = {phi:.2f}%)", fontsize=10.5, color=INK)
        style(ax)

        ax = axes[1, j]
        vals = [occ[pt, k] for pt, _, _ in REGIONS]
        ax.bar(x, vals, width=0.62, color=BAR, edgecolor="white", linewidth=2, zorder=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_ylim(0, 110)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        style(ax)

    axes[0, 0].set_ylabel("Drought prevalence (%)\n(pixel-months)", color=INK)
    axes[1, 0].set_ylabel("Drought occurrence (%)\n(target months, any pixel)", color=INK)
    fig.text(0.5, 0.005, "Region (number of valid pixels)", ha="center", color=INK)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0], [0], color=INK, lw=1.2, ls="--")],
               [r"$\Phi(\kappa)$: prevalence expected if SPI were exactly standard normal over the whole record"],
               loc="upper center", bbox_to_anchor=(0.5, 1.04), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out = HERE / "images" / "drought_prevalence"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved {out}.pdf/.png")


def slide():
    """Prevalence row only, sized and colored for the Beamer deck (slide 10)."""
    prev, _ = load()
    wet, dry, ink, muted = "#1D6A86", "#B4432A", "#17212B", "#6B7480"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.7))
    x = np.arange(len(REGIONS))
    for ax, k in zip(axes, KAPPAS):
        vals = [prev[pt, k] for pt, _, _ in REGIONS]
        ax.bar(x, vals, width=0.62, color=wet, edgecolor="white", linewidth=2, zorder=2)
        phi = norm.cdf(k) * 100
        ax.axhline(phi, color=dry, lw=1.6, ls="--", zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9.5, color=ink)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_title(rf"$\kappa = {k}$   ($\Phi$ = {phi:.2f}%)", fontsize=11.5, color=ink)
        ax.set_xticks(x)
        ax.set_xticklabels([{"CW": "CO"}.get(en, en) for _, en, _ in REGIONS], fontsize=10, color=ink)
        ax.grid(axis="y", color="#E4E6E9", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(muted)
        ax.tick_params(colors=muted, labelsize=9.5)
    axes[0].set_ylabel("Prevalência de seca (%)", color=ink)
    # The dashed line is explained in the slide caption, which saves vertical space.
    fig.tight_layout()
    out = HERE.parent / "docs_reunião" / "apresentacao_beamer" / "figs" / "drought_prevalence_slide.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(HERE / "images" / "drought_prevalence_slide.png", dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    slide() if "--slide" in sys.argv else main()
