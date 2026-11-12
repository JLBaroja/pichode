"""Plot reinforcer-cost distributions and relationships by bird and VI condition.

Run from the PICHODE repository root:
    uv run python analyses/plot_reinforcer_costs.py

Creates one multipage PDF per bird, with one 4x2 page per VI condition.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# SETTINGS — apply to all birds and VI conditions
# ============================================================
BASE = Path("data/processed/concurrent/reinforcers")
OUTPUT_DIR = Path("results/figures/reinforcer_costs")
BIRDS = ("p004", "p054", "p138", "p510", "p530", "p736")
COLORS = {"left": "tab:blue", "right": "tab:orange"}

FIGSIZE = (15, 18)
HIST_BINS = 40
HIST_ALPHA = 0.25
HIST_LINEWIDTH = 1.2
MEAN_LINEWIDTH = 2
SCATTER_ALPHA = 0.12
SCATTER_SIZE = 9

# None means automatic limits. Set a tuple to compare axes across pages.
GLOBAL_RESPONSE_LIM = 50
GLOBAL_TIME_LIM = 300
KEY_RESPONSE_LIM = 50
KEY_TIME_LIM = 300
HIST_FREQUENCY_LIM = None

GLOBAL_RESP = "cost_responses_total"
GLOBAL_TIME = "cost_time_total_sec"
KEY_RESP = "cost_responses_since_same_key_reinforcer"
KEY_TIME = "cost_time_since_same_key_reinforcer_sec"


def valid_file(path):
    """Exclude duplicate _bis files and the aborted _b recording."""
    return not path.stem.endswith(("_bis", "_b"))


def load_bird(bird):
    files = sorted(p for p in BASE.glob(f"{bird}_*.csv") if valid_file(p))
    if not files:
        raise FileNotFoundError(f"No reinforcer CSV files for {bird} in {BASE}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["session_file"] = path.stem
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_condition(data, bird, vi_left, vi_right):
    fig, axes = plt.subplots(4, 2, figsize=FIGSIZE)
    fig.suptitle(f"{bird} — VI {vi_left:g}/{vi_right:g} — Reinforcer costs", fontsize=16)

    histograms = [
        (axes[0, 0], GLOBAL_RESP, "Global response cost", "Responses", GLOBAL_RESPONSE_LIM),
        (axes[0, 1], GLOBAL_TIME, "Global time cost", "Seconds", GLOBAL_TIME_LIM),
        (axes[1, 0], KEY_RESP, "Key-specific response cost", "Responses", KEY_RESPONSE_LIM),
        (axes[1, 1], KEY_TIME, "Key-specific time cost", "Seconds", KEY_TIME_LIM),
    ]

    for ax, variable, title, xlabel, limits in histograms:
        values = data[variable].dropna()
        if not values.empty:
            upper = float(values.max())
            bins = np.linspace(0, upper if upper > 0 else 1, HIST_BINS + 1)
            for (_, key), group in data.groupby(["session_file", "response"]):
                if key not in COLORS:
                    continue
                ax.hist(
                    group[variable].dropna(), bins=bins, histtype="step",
                    color=COLORS[key], alpha=HIST_ALPHA, linewidth=HIST_LINEWIDTH,
                )
            for key in COLORS:
                mean = data.loc[data["response"] == key, variable].mean()
                if pd.notna(mean):
                    ax.axvline(
                        mean, color=COLORS[key], linestyle="--",
                        linewidth=MEAN_LINEWIDTH, alpha=0.9,
                        label=f"{key.capitalize()} mean = {mean:.1f}",
                    )
        ax.set(title=title, xlabel=xlabel, ylabel="Frequency")
        if limits is not None:
            ax.set_xlim(0, limits)
        if HIST_FREQUENCY_LIM is not None:
            ax.set_ylim(HIST_FREQUENCY_LIM)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9)

    scatters = [
        (axes[2, 0], GLOBAL_RESP, GLOBAL_TIME, "Global: responses vs. time", "Responses", "Seconds", GLOBAL_RESPONSE_LIM, GLOBAL_TIME_LIM),
        (axes[2, 1], KEY_RESP, KEY_TIME, "Key-specific: responses vs. time", "Responses", "Seconds", KEY_RESPONSE_LIM, KEY_TIME_LIM),
        (axes[3, 0], GLOBAL_RESP, KEY_RESP, "Global vs. key-specific response cost", "Global responses", "Key-specific responses", GLOBAL_RESPONSE_LIM, KEY_RESPONSE_LIM),
        (axes[3, 1], GLOBAL_TIME, KEY_TIME, "Global vs. key-specific time cost", "Global seconds", "Key-specific seconds", GLOBAL_TIME_LIM, KEY_TIME_LIM),
    ]

    for ax, x, y, title, xlabel, ylabel, xlim, ylim in scatters:
        for key in COLORS:
            group = data.loc[data["response"] == key]
            ax.scatter(
                group[x], group[y], s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
                color=COLORS[key], rasterized=True, label=key.capitalize(),
            )
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        if xlim is not None:
            ax.set_xlim(0, xlim)
        if ylim is not None:
            ax.set_ylim(0, ylim)
        ax.legend()

    for ax in axes.flat:
        ax.grid(alpha=0.15)

    fig.tight_layout()
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for bird in BIRDS:
        data = load_bird(bird)
        conditions = (
            data[["vi_left", "vi_right"]]
            .dropna().drop_duplicates().sort_values(["vi_left", "vi_right"])
        )
        output = OUTPUT_DIR / f"{bird}_reinforcer_costs.pdf"
        with PdfPages(output) as pdf:
            for vi_left, vi_right in conditions.itertuples(index=False, name=None):
                subset = data.loc[
                    (data["vi_left"] == vi_left) & (data["vi_right"] == vi_right)
                ]
                fig = plot_condition(subset, bird, vi_left, vi_right)
                pdf.savefig(fig)
                plt.close(fig)
        print(f"{output} ({len(conditions)} conditions)", flush=True)


if __name__ == "__main__":
    main()
