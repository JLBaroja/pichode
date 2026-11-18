"""Four-model conditional posterior predictive cumulative records for poster."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import plot_single_timescale_cumulative_records as single
import plot_predictive_cumulative_records as two
import plot_initial_conditions_cumulative_records as initial
from plot_colors import MODEL_COLORS


OUTPUT_DIR = Path("results/timescales/poster/figures")
INITIAL_ROOT = Path("results/timescales/initial_conditions")

MODELS = [
    ("M0", "M0: Single timescale"),
    ("two_timescales", "Two-timescales"),
    ("M1", "M1: Flexible initial strength"),
    ("M2", "M2: Flexible initial strength + preference"),
]

N_DRAWS_DISPLAYED = 30
N_PREDICTIVE_DRAWS = 100


def get_predictions(df, bird, session, model_key, seed):
    """Reuse the existing conditional posterior predictive simulators."""

    filename = f"{bird}_s{session:03d}.csv"

    if model_key == "M0":
        path = single.DRAWS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(path)

        single.RNG = np.random.default_rng(seed)
        return single.predictive_trajectories(df, path)

    if model_key == "two_timescales":
        path = two.DRAWS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(path)

        two.RNG = np.random.default_rng(seed)
        return two.predictive_trajectories(df, path)

    path = (
        INITIAL_ROOT / model_key
        / f"{bird}_s{session:03d}"
        / "posterior.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    return initial.predictive_trajectories(
        df, path, model_key, np.random.default_rng(seed)
    )


def plot_figure(bird, session):
    filename = f"{bird}_s{session:03d}_japede.csv"
    path = two.BASE / filename

    if not path.exists():
        raise FileNotFoundError(path)

    df = two.prepare(pd.read_csv(path))

    if df.empty:
        raise ValueError(f"Empty session: {path}")

    segments = two.segment_information(df)

    if len(segments) < 2:
        raise ValueError(
            f"{bird} session {session} has no detected contingency reversal."
        )

    observed_left = np.r_[0, df["N_left"].to_numpy()]
    observed_right = np.r_[0, df["N_right"].to_numpy()]

    predictions = {}

    for i, (model_key, _) in enumerate(MODELS):
        left, right = get_predictions(
            df, bird, session, model_key, seed=123 + i
        )
        predictions[model_key] = (left, right)

    # Same limits for every panel, including predictive trajectories.
    maximum = max(
        observed_left.max(),
        observed_right.max(),
        *[
            max(left.max(), right.max())
            for left, right in predictions.values()
        ],
    )
    limit = np.ceil(maximum * 1.04 / 100) * 100

    fig, axes = plt.subplots(
        2, 2,
        figsize=(9.5, 9.5),
        sharex=True,
        sharey=True,
    )

    for i, (ax, (model_key, model_title)) in enumerate(
        zip(axes.flat, MODELS)
    ):
        left, right = predictions[model_key]
        color = MODEL_COLORS[model_key]

        # Posterior predictive sample paths.
        for j in range(min(N_DRAWS_DISPLAYED, len(left))):
            ax.plot(
                right[j], left[j],
                color=color,
                alpha=0.08,
                linewidth=0.65,
                zorder=1,
            )

        # Coordinatewise posterior predictive median.
        ax.plot(
            np.median(right, axis=0),
            np.median(left, axis=0),
            color=color,
            linewidth=1.8,
            zorder=3,
        )

        # Identical observed data in every panel.
        ax.plot(
            observed_right,
            observed_left,
            color="black",
            linewidth=2.3,
            zorder=4,
        )

        # Equal cumulative responding.
        ax.plot(
            [0, limit], [0, limit],
            color="0.7",
            linestyle="--",
            linewidth=0.8,
            zorder=0,
        )

        # Mark the contingency reversal(s) on observed data.
        for segment in segments[1:]:
            ax.scatter(
                segment["start"]["N_right"],
                segment["start"]["N_left"],
                color="black",
                facecolors="white",
                edgecolors="black",
                s=34,
                linewidths=1.1,
                zorder=6,
            )

        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            model_title,
            fontsize=11,
            color=color,
            pad=7,
        )
        ax.tick_params(labelsize=8)
        ax.tick_params(axis="y", labelrotation=90, pad=2)
        for tick in ax.get_yticklabels():
            tick.set_horizontalalignment("right")
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    
    

    fig.suptitle(
        f"{bird} · Session {session:03d}  |  "
        f"{two.contingency_label(segments)}",
        fontsize=17,
        y=0.965,
    )

    # Explicit square axes with a narrow inter-column gap.
    # Positions are in figure coordinates: (left, bottom, width, height).
    positions = [
        (0.115, 0.505, 0.365, 0.365),
        (0.505, 0.505, 0.365, 0.365),
        (0.115, 0.105, 0.365, 0.365),
        (0.505, 0.105, 0.365, 0.365),
    ]

    for ax, position in zip(axes.flat, positions):
        ax.set_position(position)


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

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=False,
        fontsize=9,
    )


    # Single shared labels, close to the tick labels.
    fig.text(
        0.075, 0.502,
        "Cumulative left responses",
        rotation=90, ha="center", va="center", fontsize=13,
    )

    fig.text(
        0.492, 0.060,
        "Cumulative right responses",
        ha="center", va="center", fontsize=13,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{bird}_s{session:03d}_four_models.pdf"

    fig.savefig(output, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bird", default="p736")
    parser.add_argument("--session", type=int, default=169)
    args = parser.parse_args()

    output = plot_figure(args.bird, args.session)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
