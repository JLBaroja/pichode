"""Shared plotting conventions for the MathPsych poster."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_colors import MODEL_COLORS, BIRD_COLORS


FONT = {
    "title": 17,
    "axis": 13,
    "panel": 11,
    "tick": 8,
    "legend": 9,
}


def apply_style():
    """Set defaults for subsequent Matplotlib figures."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": FONT["tick"],
        "axes.titlesize": FONT["panel"],
        "axes.labelsize": FONT["axis"],
        "xtick.labelsize": FONT["tick"],
        "ytick.labelsize": FONT["tick"],
        "legend.fontsize": FONT["legend"],
        "axes.spines.top": True,
        "axes.spines.right": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,
    })


def style_axes(ax, *, square=False, vertical_y_ticks=True):
    """Style an individual panel without changing its data."""
    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.tick_params(
        axis="y",
        labelrotation=90 if vertical_y_ticks else 0,
        pad=2,
        labelsize=FONT["tick"],
    )
    ax.tick_params(axis="x", labelsize=FONT["tick"])

    if square:
        ax.set_aspect("equal", adjustable="box")

    return ax


def make_figure(
    rows=1,
    cols=1,
    *,
    width=9.5,
    height=None,
    square=False,
):
    """Create single- or multipanel figures."""
    if height is None:
        height = width * rows / cols if square else width * 0.65

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(width, height),
        squeeze=False,
    )

    for ax in axes.flat:
        style_axes(ax, square=square)

    return fig, axes


def add_labels(fig, axes, xlabel=None, ylabel=None):
    """Add shared axis labels, or labels to a single panel."""
    if axes.size == 1:
        ax = axes.flat[0]
        if xlabel:
            ax.set_xlabel(xlabel, labelpad=10)
        if ylabel:
            ax.set_ylabel(ylabel, labelpad=8)
    else:
        if xlabel:
            fig.supxlabel(xlabel, fontsize=FONT["axis"])
        if ylabel:
            fig.supylabel(ylabel, fontsize=FONT["axis"])


def add_title(fig, title):
    fig.suptitle(title, fontsize=FONT["title"])


def predictive_legend(fig, *, anchor=(0.5, 0.94)):
    """Shared legend for posterior predictive figures."""
    handles = [
        Line2D([0], [0], color="black", lw=2.3,
               label="Observed behavior"),
        Line2D([0], [0], color="0.5", lw=0.8, alpha=0.5,
               label="Posterior predictive draws"),
        Line2D([0], [0], color="0.5", lw=1.8,
               label="Posterior predictive median"),
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor="white", markeredgecolor="black",
               markersize=6, label="Contingency reversal"),
    ]

    return fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=anchor,
        ncol=4,
        frameon=False,
        fontsize=FONT["legend"],
    )


def save_figure(fig, path):
    """Save a tightly cropped vector PDF."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path
