"""Plot session-level visit distributions for each pigeon and VI condition."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# --------------------------------------------------
# Configuration — change these whenever you want
# --------------------------------------------------

INPUT_DIR = Path("data/processed/concurrent/visits")
OUTPUT_DIR = Path("results/figures")

BIRDS = ["p004", "p054", "p138", "p510", "p530", "p736"]

VI_PAIRS = [
    (24, 360),
    (25, 225),
    (27, 135),
    (30, 90),
    (35, 63),
    (45, 45),
    (63, 35),
    (90, 30),
    (135, 27),
    (225, 25),
    (360, 24),
]

EXCLUDED = {"p530_s145_japede_b"}

MAX_VISIT_LENGTH = 15
MAX_VISIT_DURATION = 15

DURATION_BINS = 100

COLORS = {
    "left": "tab:blue",
    "right": "tab:orange",
}

RESPONSE_BINS = np.arange(0.5, MAX_VISIT_LENGTH + 1.5, 1)
DURATION_BIN_EDGES = np.linspace(
    0, MAX_VISIT_DURATION, DURATION_BINS + 1
)


# --------------------------------------------------
# Load and organize data
# --------------------------------------------------

def load_data():
    data = {}

    for bird in BIRDS:
        data[bird] = {}

        files = sorted(INPUT_DIR.glob(f"{bird}_*.csv"))

        valid_files = [
            f for f in files
            if "_bis" not in f.stem
            and f.stem not in EXCLUDED
        ]

        assert len(valid_files) == 180, (
            f"{bird}: expected 180 sessions, found {len(valid_files)}"
        )

        for file in valid_files:
            df = pd.read_csv(
                file,
                usecols=[
                    "response",
                    "n_responses",
                    "duration_sec",
                    "vi_left",
                    "vi_right",
                ],
            )

            for pair in VI_PAIRS:
                vi_left, vi_right = pair

                subset = df[
                    (df["vi_left"] == vi_left)
                    & (df["vi_right"] == vi_right)
                ]

                if subset.empty:
                    continue

                data[bird].setdefault(pair, {})
                data[bird][pair][file.stem] = subset

        print(
            f"Loaded {bird.upper()}: "
            f"{len(data[bird])} VI conditions"
        )

    return data


# --------------------------------------------------
# Plot one bird and VI condition
# --------------------------------------------------

def plot_condition(bird, pair, sessions):
    vi_left, vi_right = pair

    fig, axes = plt.subplots(
        2, 3,
        figsize=(18, 10),
    )

    for row, lever in enumerate(["left", "right"]):
        color = COLORS[lever]

        # Column 1: Visit length (responses)
        ax = axes[row, 0]

        for session, df in sessions.items():
            values = df.loc[
                df["response"] == lever,
                "n_responses",
            ].dropna()

            if len(values):
                ax.hist(
                    values,
                    bins=RESPONSE_BINS,
                    histtype="step",
                    color=color,
                    alpha=0.3,
                    linewidth=0.8,
                )

        ax.set_xlim(0.5, MAX_VISIT_LENGTH + 0.5)
        ax.set_xlabel("Visit length (responses)")
        ax.set_ylabel("Count")
        ax.set_title(f"{lever.capitalize()} key — visit length")
        ax.grid(alpha=0.15)

        # Column 2: Visit duration (seconds)
        ax = axes[row, 1]

        for session, df in sessions.items():
            values = df.loc[
                df["response"] == lever,
                "duration_sec",
            ].dropna()

            if len(values):
                ax.hist(
                    values,
                    bins=DURATION_BIN_EDGES,
                    histtype="step",
                    color=color,
                    alpha=0.3,
                    linewidth=0.8,
                )

        ax.set_xlim(0, MAX_VISIT_DURATION)
        ax.set_xlabel("Visit duration (seconds)")
        ax.set_ylabel("Count")
        ax.set_title(f"{lever.capitalize()} key — visit duration")
        ax.grid(alpha=0.15)

        # Column 3: Visit length vs. duration
        ax = axes[row, 2]

        for session, df in sessions.items():
            subset = df[df["response"] == lever]

            if subset.empty:
                continue

            ax.scatter(
                subset["n_responses"],
                subset["duration_sec"],
                color=color,
                alpha=0.12,
                s=5,
                edgecolors="none",
                rasterized=True,
            )

        ax.set_xlim(0.5, MAX_VISIT_LENGTH + 0.5)
        ax.set_ylim(0, MAX_VISIT_DURATION)
        ax.set_xlabel("Visit length (responses)")
        ax.set_ylabel("Visit duration (seconds)")
        ax.set_title(
            f"{lever.capitalize()} key — length vs. duration"
        )
        ax.grid(alpha=0.15)

    fig.suptitle(
        f"{bird.upper()} — VI {vi_left}/{vi_right} — "
        f"{len(sessions)} sessions",
        fontsize=16,
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    return fig


# --------------------------------------------------
# Generate PDFs
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    for bird in BIRDS:
        filename = OUTPUT_DIR / f"{bird}_visit_distributions.pdf"

        print(f"\nGenerating {filename}...")

        with PdfPages(filename) as pdf:
            for pair in VI_PAIRS:
                if pair not in data[bird]:
                    continue

                fig = plot_condition(
                    bird,
                    pair,
                    data[bird][pair],
                )

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        print(f"Saved: {filename}")

    print("\nAll visit-distribution PDFs generated successfully!")


if __name__ == "__main__":
    main()
