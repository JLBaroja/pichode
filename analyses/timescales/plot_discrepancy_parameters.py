"""Plot temporal prediction discrepancies against session and model parameters."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BASE = Path("results/timescales/comparison/temporal_discrepancies")
OUTPUT = BASE / "parameter_relationships"

BIRDS = ("p004", "p054", "p138", "p510", "p530", "p736")

PARAMETERS = {
    "session": ("Session number", None),
    "tau_mean": ("Single-timescale tau", "single"),
    "tau_fast_mean": ("Two-timescale tau_fast", "two"),
    "tau_slow_mean": ("Two-timescale tau_slow", "two"),
    "w_mean": ("Two-timescale w", "two"),
}

METRICS = {
    "absolute_error": "Absolute allocation discrepancy",
    "signed_error": "Observed minus predicted P(left)",
}

MODEL_COLORS = {
    "single": "darkviolet",
    "two": "steelblue",
}


def load_data():
    data = pd.read_csv(
        BASE / "window_discrepancies_60min.csv"
    )

    single = pd.read_csv(
        "results/timescales/single_timescale/summary.csv"
    )

    two = pd.read_csv(
        "results/timescales/preliminary/summary.csv"
    )

    single = single[
        ["bird", "session", "tau_mean"]
    ]

    two = two[
        [
            "bird",
            "session",
            "tau_fast_mean",
            "tau_slow_mean",
            "w_mean",
        ]
    ]

    data = data.merge(
        single,
        on=["bird", "session"],
        how="left",
        validate="many_to_one",
    )

    data = data.merge(
        two,
        on=["bird", "session"],
        how="left",
        validate="many_to_one",
    )

    # Retain the twelve five-minute windows.
    data = data[
        (data["start_min"] >= 0)
        & (data["start_min"] < 60)
    ].copy()

    return data


def plot_page(data, bird, metric, parameter, pdf):

    xlabel, parameter_model = PARAMETERS[parameter]

    fig, ax = plt.subplots(figsize=(11, 6.5))

    windows = np.arange(0, 60, 5)
    cmap = plt.get_cmap("turbo", len(windows))

    for i, start in enumerate(windows):

        window_data = data[data["start_min"] == start]

        for session_type, marker in (
            ("stable", "o"),
            ("dynamic", "^"),
        ):
            subset = window_data[
                window_data["session_type"] == session_type
            ]

            models = (
                ("single", "two")
                if parameter_model is None
                else (parameter_model,)
            )

            for model in models:
                group = subset[subset["model"] == model]

                ax.scatter(
                    group[parameter],
                    group[metric],
                    color=cmap(i),
                    marker=marker,
                    s=15,
                    alpha=0.45,
                    edgecolors="none",
                    rasterized=True,
                )

    if metric == "signed_error":
        ax.axhline(
            0, color="gray", linestyle="--", linewidth=1
        )

    ax.set(
        title=f"{bird}: {METRICS[metric]} vs {xlabel}",
        xlabel=xlabel,
        ylabel=METRICS[metric],
    )

    if parameter == "session":
        ax.set_xlim(left=0)

    if parameter == "w_mean":
        ax.set_xlim(0, 1)

    ax.grid(alpha=0.15)

    # Color legend: five-minute windows.
    window_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            color=cmap(i),
            markersize=6,
            label=f"{start}-{start + 5} min",
        )
        for i, start in enumerate(windows)
    ]

    phase_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            color="gray",
            label="Stable",
        ),
        Line2D(
            [0], [0],
            marker="^",
            linestyle="none",
            color="gray",
            label="Dynamic",
        ),
    ]

    ax.legend(
        handles=window_handles + phase_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9,
    )

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():

    OUTPUT.mkdir(parents=True, exist_ok=True)

    data = load_data()

    # Save the merged dataset for subsequent analyses.
    data.to_csv(
        OUTPUT / "discrepancies_with_parameters.csv",
        index=False,
    )

    for bird in BIRDS:

        bird_data = data[data["bird"] == bird]

        for metric in METRICS:

            path = OUTPUT / f"{bird}_{metric}.pdf"

            with PdfPages(path) as pdf:

                for parameter in PARAMETERS:

                    plot_page(
                        bird_data,
                        bird,
                        metric,
                        parameter,
                        pdf,
                    )

            print(f"Saved {path}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
