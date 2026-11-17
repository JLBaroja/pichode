"""Temporal discrepancy plots with individual sessions colored by bird."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BASE = Path("results/timescales/comparison/temporal_discrepancies")
OUTPUT = BASE / "by_bird"
EVENTS = Path("data/processed/concurrent/events")

BIRDS = ("p004", "p054", "p138", "p510", "p530", "p736")

COLORS = {
    "p004": "tab:blue",
    "p054": "tab:orange",
    "p138": "tab:green",
    "p510": "tab:red",
    "p530": "tab:purple",
    "p736": "tab:brown",
}

MODELS = {
    "single": "Single-timescale",
    "two": "Two-timescale",
}

METRICS = {
    "absolute_error": "Absolute allocation discrepancy",
    "signed_error": "Observed minus predicted P(left)",
}


def load_data():
    data = pd.read_csv(
        BASE / "window_discrepancies_60min.csv"
    )

    data = data[
        (data["start_min"] >= 0)
        & (data["start_min"] < 60)
    ].copy()

    # Attach the first contingency-change time.
    changes = []

    dynamic = data[
        data["session_type"] == "dynamic"
    ][["bird", "session"]].drop_duplicates()

    for row in dynamic.itertuples(index=False):
        path = EVENTS / (
            f"{row.bird}_s{int(row.session):03d}_japede.csv"
        )

        event = pd.read_csv(path, nrows=1).iloc[0]

        changes.append({
            "bird": row.bird,
            "session": row.session,
            "change_min": float(event["first_cp_left"]) / 60,
        })

    data = data.drop(columns=["change_min"], errors="ignore")

    data = data.merge(
        pd.DataFrame(changes),
        on=["bird", "session"],
        how="left",
    )

    data["plot_time"] = np.where(
        data["session_type"] == "dynamic",
        data["midpoint_min"] - data["change_min"],
        data["midpoint_min"],
    )

    return data


def plot_figure(data, session_type, metric):

    fig, axes = plt.subplots(
        1, 2, figsize=(15, 6), sharey=True
    )

    for ax, (model, model_label) in zip(
        axes, MODELS.items()
    ):

        subset = data[
            (data["session_type"] == session_type)
            & (data["model"] == model)
        ]

        # Individual sessions.
        for (bird, session), group in subset.groupby(
            ["bird", "session"]
        ):
            group = group.sort_values("plot_time")

            ax.plot(
                group["plot_time"],
                group[metric],
                color=COLORS[bird],
                linewidth=0.5,
                alpha=0.09,
                rasterized=True,
            )

        # Bird-specific means.
        for bird in BIRDS:
            bird_data = subset[
                subset["bird"] == bird
            ].copy()

            if bird_data.empty:
                continue

            if session_type == "dynamic":
                bird_data["bin"] = (
                    np.floor(bird_data["plot_time"] / 5) * 5 + 2.5
                )
            else:
                bird_data["bin"] = bird_data["midpoint_min"]

            means = (
                bird_data.groupby("bin")[metric]
                .mean()
                .sort_index()
            )

            ax.plot(
                means.index,
                means.values,
                color=COLORS[bird],
                linewidth=2.2,
            )

        # Overall mean, giving each bird equal weight.
        all_data = subset.copy()

        if session_type == "dynamic":
            all_data["bin"] = (
                np.floor(all_data["plot_time"] / 5) * 5 + 2.5
            )
        else:
            all_data["bin"] = all_data["midpoint_min"]

        bird_means = (
            all_data.groupby(["bird", "bin"])[metric]
            .mean()
            .reset_index()
        )

        overall = (
            bird_means.groupby("bin")[metric]
            .mean()
            .sort_index()
        )

        ax.plot(
            overall.index,
            overall.values,
            color="black",
            linestyle="--",
            linewidth=2.5,
        )

        if session_type == "dynamic":
            ax.axvline(
                0,
                color="gray",
                linestyle=":",
                linewidth=1.5,
            )
            ax.set_xlabel(
                "Minutes relative to first contingency change"
            )
        else:
            ax.set_xlabel("Minutes since session start")
            ax.set_xlim(0, 60)

        if metric == "signed_error":
            ax.axhline(
                0,
                color="gray",
                linestyle=":",
                linewidth=1,
            )

        ax.set_title(model_label)
        ax.grid(alpha=0.15)

    axes[0].set_ylabel(METRICS[metric])

    legend = [
        Line2D(
            [0], [0],
            color=COLORS[bird],
            linewidth=2.5,
            label=bird,
        )
        for bird in BIRDS
    ]

    legend.append(
        Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            linewidth=2.5,
            label="Overall mean",
        )
    )

    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=7,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        f"{session_type.capitalize()} sessions — {METRICS[metric]}"
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    OUTPUT.mkdir(parents=True, exist_ok=True)

    filename = f"{session_type}_{metric}_by_bird.pdf"

    fig.savefig(OUTPUT / filename, dpi=200)
    plt.close(fig)

    print(f"Saved {OUTPUT / filename}")


def main():
    data = load_data()

    for session_type in ("stable", "dynamic"):
        for metric in METRICS:
            plot_figure(data, session_type, metric)


if __name__ == "__main__":
    main()
