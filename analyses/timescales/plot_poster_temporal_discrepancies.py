"""Poster: temporal absolute discrepancies, individual sessions and bird means."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_colors import BIRD_COLORS
from plot_style import apply_style, make_figure, add_labels, save_figure

BASE = Path("results/timescales")
OUTPUT = BASE / "poster/figures"

FILES = [
    BASE / "comparison/temporal_discrepancies/window_discrepancies.csv",
    BASE / "initial_conditions/temporal_discrepancies/window_discrepancies.csv",
]

FIGURES = [
    ("M0_two_timescales", ["single", "two"]),
    ("M1_M2", ["M1", "M2"]),
]

MODEL_LABELS = {
    "single": "M0",
    "two": "Two timescales",
    "M1": "M1",
    "M2": "M2",
}


def load_data():
    data = pd.concat(
        [pd.read_csv(path) for path in FILES],
        ignore_index=True,
    )

    dynamic = data["session_type"].eq("dynamic")

    data["plot_time"] = data["midpoint_min"]

    data.loc[dynamic, "plot_time"] = (
        data.loc[dynamic, "midpoint_min"]
        - data.loc[dynamic, "first_change_min"]
    )

    # Exclude stable-session observations beyond 60 minutes.
    data = data[
        dynamic | (data["start_min"] < 60)
    ].copy()

    # Five-minute bins for mean trajectories.
    data["plot_bin"] = (
        np.floor(data["plot_time"] / 5) * 5 + 2.5
    )

    return data


def plot_panel(ax, data, model, session_type):
    subset = data[
        (data["model"] == model)
        & (data["session_type"] == session_type)
    ]

    for bird, bird_data in subset.groupby("bird"):
        color = BIRD_COLORS[bird]

        # Every individual session.
        for _, session_data in bird_data.groupby("session"):
            session_data = session_data.sort_values("plot_time")

            ax.plot(
                session_data["plot_time"],
                session_data["absolute_error"],
                color=color,
                linewidth=0.55,
                alpha=0.035 if session_type == "stable" else 0.075,
                zorder=1,
            )

        # Mean trajectory for this pigeon.
        summary = (
            bird_data.groupby("plot_bin")["absolute_error"]
            .mean()
            .sort_index()
        )

        ax.plot(
            summary.index,
            summary.values,
            color=color,
            linewidth=2.2,
            zorder=3,
        )

    if session_type == "dynamic":
        ax.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=1,
            zorder=2,
        )
        ax.set_xlim(-40, 40)
    else:
        ax.set_xlim(0, 60)

    ax.set_ylim(0, 0.3)
    ax.set_title(
        f"{MODEL_LABELS[model]} — {session_type.capitalize()} sessions"
    )
    ax.grid(alpha=0.12)


def make_poster_figure(data, name, models):
    fig, axes = make_figure(
        rows=2,
        cols=2,
        width=9.5,
        height=5.6,
    )

    for row, model in enumerate(models):
        for col, session_type in enumerate(("stable", "dynamic")):
            plot_panel(
                axes[row, col],
                data,
                model,
                session_type,
            )

    for ax in axes.flat:
        ax.set_ylim(0, 0.3)

    fig.supylabel(
        "Mean absolute prediction discrepancy",
        x=0.065,
        fontsize=13,
    )

    fig.suptitle(
        "Absolute prediction discrepancies",
        fontsize=17,
        y=1.035,
    )

    axes[1, 0].set_xlabel(
        "Minutes since session start", fontsize=13
    )
    axes[1, 1].set_xlabel(
        "Minutes relative to first change", fontsize=13
    )

    handles = [
        Line2D(
            [0], [0],
            color=color,
            linewidth=2.2,
            label=bird,
        )
        for bird, color in BIRD_COLORS.items()
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=6,
        frameon=False,
        fontsize=9,
    )

    fig.subplots_adjust(
        top=0.89,
        bottom=0.12,
        left=0.11,
        right=0.98,
        hspace=0.34,
        wspace=0.19,
    )

    path = OUTPUT / f"absolute_discrepancy_{name}.pdf"
    save_figure(fig, path)
    print(f"Saved {path}")


def main():
    apply_style()
    data = load_data()

    for name, models in FIGURES:
        make_poster_figure(data, name, models)


if __name__ == "__main__":
    main()
