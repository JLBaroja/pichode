"""Explore reinforcement within visits, by bird and VI condition.

Run from the PICHODE repository root:
    uv run python analyses/plot_visit_reinforcement.py

Produces one multipage PDF per bird, one page per VI condition.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# SETTINGS — change these once to update ALL birds/conditions
# ============================================================
BASE = Path("data/processed/concurrent")
OUTPUT_DIR = Path("results/figures/visit_reinforcement")
BIRDS = ("p004", "p054", "p138", "p510", "p530", "p736")
COLORS = {"left": "tab:blue", "right": "tab:orange"}

# Set any limit to None to let Matplotlib choose automatically.
VISIT_LENGTH_XLIM = (0, 60)       # responses
VISIT_DURATION_XLIM = (0, 400)    # seconds
REINFORCER_COUNT_YLIM = (0, 5)    # reinforcers per visit
FRACTION_YLIM = (0, 1)            # reinforced responses / responses
POSITION_XLIM = (0.5, 20.5)      # response index within visit
POSITION_YLIM = None             # histogram counts; optional fixed limit
SINGLE_VISIT_YLIM = (0, 1)       # proportions

SCATTER_SIZE = 9
SCATTER_ALPHA = 0.12
HIST_ALPHA = 0.25
HIST_LINEWIDTH = 1.2
SINGLE_ALPHA = 0.22
FIGSIZE = (15, 18)


def valid_file(path):
    """Exclude duplicate _bis files and the aborted _b recording."""
    return not path.stem.endswith(("_bis", "_b"))


def load_bird(bird):
    visit_parts, reinforcer_parts = [], []
    files = sorted(p for p in (BASE / "visits").glob(f"{bird}_*.csv") if valid_file(p))
    if not files:
        raise FileNotFoundError(f"No visit CSV files for {bird} in {BASE / 'visits'}")

    for file in files:
        reinforcer_file = BASE / "reinforcers" / file.name
        if not reinforcer_file.is_file():
            raise FileNotFoundError(reinforcer_file)
        v = pd.read_csv(file)
        r = pd.read_csv(reinforcer_file)
        v["session_file"] = file.name
        r["session_file"] = file.name
        visit_parts.append(v)
        reinforcer_parts.append(r)

    visits = pd.concat(visit_parts, ignore_index=True)
    reinforcers = pd.concat(reinforcer_parts, ignore_index=True)
    visits["reinforcement_fraction"] = visits["n_reinforcers"] / visits["n_responses"]

    # Join BEFORE filtering by VI: a reinforcer and its visit can straddle a VI change.
    reinforcers = reinforcers.merge(
        visits[["session_file", "visit", "n_responses"]],
        on=["session_file", "visit"],
        how="left",
        validate="many_to_one",
    )
    if reinforcers["n_responses"].isna().any():
        raise ValueError(f"{bird}: reinforcers with unmatched visit IDs")
    return visits, reinforcers


def set_xlim(ax, limits):
    if limits is not None:
        ax.set_xlim(*limits)


def set_ylim(ax, limits):
    if limits is not None:
        ax.set_ylim(*limits)


def plot_condition(bird, vi_left, vi_right, visits, reinforcers):
    fig, axes = plt.subplots(4, 2, figsize=FIGSIZE)
    fig.suptitle(f"{bird} — VI {vi_left:g}/{vi_right:g}: Reinforcement within visits", fontsize=17)

    specs = [
        ("n_responses", "n_reinforcers", "Visit length (responses)", "Reinforcers obtained", VISIT_LENGTH_XLIM, REINFORCER_COUNT_YLIM),
        ("duration_sec", "n_reinforcers", "Visit duration (seconds)", "Reinforcers obtained", VISIT_DURATION_XLIM, REINFORCER_COUNT_YLIM),
        ("n_responses", "reinforcement_fraction", "Visit length (responses)", "Fraction reinforced", VISIT_LENGTH_XLIM, FRACTION_YLIM),
        ("duration_sec", "reinforcement_fraction", "Visit duration (seconds)", "Fraction reinforced", VISIT_DURATION_XLIM, FRACTION_YLIM),
    ]
    for ax, (x, y, xlabel, ylabel, xlim, ylim) in zip(axes[:2].flat, specs):
        for key in ("left", "right"):
            d = visits.loc[visits["response"].eq(key)]
            ax.scatter(d[x], d[y], s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
                       color=COLORS[key], label=key.capitalize(), rasterized=True)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        set_xlim(ax, xlim)
        set_ylim(ax, ylim)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)

    for col, key in enumerate(("left", "right")):
        ax = axes[2, col]
        d = visits.loc[visits["response"].eq(key) & visits["n_responses"].eq(1)]
        for _, group in d.groupby("session_file", sort=False):
            p = group["n_reinforcers"].gt(0).mean()
            ax.plot([0, 1], [1 - p, p], marker="o", linewidth=1,
                    alpha=SINGLE_ALPHA, color=COLORS[key])
        ax.set(title=f"{key.capitalize()} — Single-response visits",
               ylabel="Proportion of visits", xticks=[0, 1],
               xticklabels=["Unreinforced", "Reinforced"])
        set_ylim(ax, SINGLE_VISIT_YLIM)
        ax.grid(alpha=0.2)

    # Integer bins, one bin for each response position; display range is configurable.
    maximum_position = int(np.ceil(POSITION_XLIM[1] - 0.5)) if POSITION_XLIM else int(reinforcers["response_in_visit"].max())
    bins = np.arange(0.5, maximum_position + 1.5, 1)
    for col, key in enumerate(("left", "right")):
        ax = axes[3, col]
        d = reinforcers.loc[reinforcers["response"].eq(key) & reinforcers["n_responses"].ge(2)]
        for _, group in d.groupby("session_file", sort=False):
            ax.hist(group["response_in_visit"], bins=bins, histtype="step",
                    linewidth=HIST_LINEWIDTH, alpha=HIST_ALPHA, color=COLORS[key])
        ax.set(title=f"{key.capitalize()} — Visits with ≥2 responses",
               xlabel="Reinforcer position within visit (responses)",
               ylabel="Number of reinforcers")
        set_xlim(ax, POSITION_XLIM)
        set_ylim(ax, POSITION_YLIM)
        ax.grid(alpha=0.2)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for bird in BIRDS:
        visits, reinforcers = load_bird(bird)
        conditions = sorted(set(zip(visits["vi_left"], visits["vi_right"])))
        destination = OUTPUT_DIR / f"{bird}_visit_reinforcement.pdf"
        with PdfPages(destination) as pdf:
            for vi_left, vi_right in conditions:
                v = visits.loc[visits["vi_left"].eq(vi_left) & visits["vi_right"].eq(vi_right)]
                r = reinforcers.loc[reinforcers["vi_left"].eq(vi_left) & reinforcers["vi_right"].eq(vi_right)]
                fig = plot_condition(bird, vi_left, vi_right, v, r)
                pdf.savefig(fig)
                plt.close(fig)
        print(f"{bird}: {len(conditions)} conditions → {destination}", flush=True)


if __name__ == "__main__":
    main()
