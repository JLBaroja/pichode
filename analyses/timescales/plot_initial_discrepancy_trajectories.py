"""Plot first-five-minute prediction discrepancies across sessions."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


BASE = Path("results/timescales/comparison/temporal_discrepancies")
OUTPUT = BASE / "initial_discrepancy_trajectories.pdf"

BIRDS = ("p004", "p054", "p138", "p510", "p530", "p736")

COLORS = {
    "single": "darkviolet",
    "two": "steelblue",
}

LABELS = {
    "single": "Single-timescale",
    "two": "Two-timescale",
}

METRICS = {
    "absolute_error": "Absolute discrepancy",
    "signed_error": "Observed minus predicted P(left)",
}


def main():

    data = pd.read_csv(
        BASE / "window_discrepancies_60min.csv"
    )

    # Only the first five minutes of each session.
    data = data[data["start_min"] == 0].copy()

    with PdfPages(OUTPUT) as pdf:

        for bird in BIRDS:

            subset = data[data["bird"] == bird]

            fig, axes = plt.subplots(
                1, 2,
                figsize=(14, 5),
                sharex=True,
            )

            for ax, (metric, ylabel) in zip(
                axes, METRICS.items()
            ):

                for model in ("single", "two"):

                    group = subset[
                        subset["model"] == model
                    ].sort_values("session")

                    ax.plot(
                        group["session"],
                        group[metric],
                        color=COLORS[model],
                        linewidth=1.2,
                        alpha=0.8,
                        marker="o",
                        markersize=2,
                        label=LABELS[model],
                    )

                if metric == "signed_error":
                    ax.axhline(
                        0,
                        color="gray",
                        linestyle="--",
                        linewidth=1,
                    )

                ax.set(
                    xlabel="Session number",
                    ylabel=ylabel,
                    title=ylabel,
                    xlim=(1, 180),
                )

                ax.grid(alpha=0.2)
                ax.legend()

            fig.suptitle(
                f"{bird} — Prediction discrepancy during first 5 minutes"
            )

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
