"""Plot session-level IRT distributions for each pigeon and VI condition."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_DIR = Path("data/processed/concurrent/responses")
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

RESOLUTION = 0.01

LINEAR_MAX = 10
LOG_MAX = 1000

N_BINS = 100

LINEAR_BINS = np.linspace(0, LINEAR_MAX, N_BINS + 1)
LOG_BINS = np.geomspace(RESOLUTION, LOG_MAX, N_BINS + 1)

COLORS = {
    "within": "tab:blue",
    "switch": "tab:orange",
}

LEGEND = [
    Line2D([0], [0], color="tab:blue", label="Within-key"),
    Line2D([0], [0], color="tab:orange", label="Switch"),
    Line2D(
        [0], [0],
        color="red",
        linestyle="--",
        label="0.01 s resolution",
    ),
]


# --------------------------------------------------
# Load and organize responses
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
                    "irt_sec",
                    "irt_type",
                    "vi_left",
                    "vi_right",
                ],
            )

            df = df[df["irt_sec"] > 0]

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
# Calculate global histogram limits
# --------------------------------------------------

def calculate_limits(data):
    global_max = {
        "linear": 0,
        "log": 0,
    }

    for bird in BIRDS:
        for sessions in data[bird].values():
            for df in sessions.values():
                for lever in ["left", "right"]:
                    for irt_type in ["within", "switch"]:

                        values = df.loc[
                            (df["response"] == lever)
                            & (df["irt_type"] == irt_type),
                            "irt_sec",
                        ].dropna().to_numpy()

                        if len(values) == 0:
                            continue

                        for scale, bins in [
                            ("linear", LINEAR_BINS),
                            ("log", LOG_BINS),
                        ]:
                            counts, _ = np.histogram(
                                values,
                                bins=bins,
                            )

                            global_max[scale] = max(
                                global_max[scale],
                                counts.max(),
                            )

    def nice_limit(value):
        if value <= 0:
            return 1

        target = value * 1.05
        magnitude = 10 ** np.floor(np.log10(target))

        return int(np.ceil(target / magnitude) * magnitude)

    return {
        scale: nice_limit(value)
        for scale, value in global_max.items()
    }


# --------------------------------------------------
# Plot one bird and VI condition
# --------------------------------------------------

def plot_condition(bird, pair, sessions, ylim):
    vi_left, vi_right = pair

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 10),
    )

    for row, lever in enumerate(["left", "right"]):
        for col, (scale, bins) in enumerate([
            ("linear", LINEAR_BINS),
            ("log", LOG_BINS),
        ]):

            ax = axes[row, col]

            for df in sessions.values():
                for irt_type, color in COLORS.items():

                    values = df.loc[
                        (df["response"] == lever)
                        & (df["irt_type"] == irt_type),
                        "irt_sec",
                    ].dropna()

                    if len(values):
                        ax.hist(
                            values,
                            bins=bins,
                            histtype="step",
                            color=color,
                            alpha=0.25,
                            linewidth=0.8,
                        )

            ax.axvline(
                RESOLUTION,
                color="red",
                linestyle="--",
                linewidth=1.2,
            )

            ax.set_xscale(scale)
            ax.set_xlabel("IRT (seconds)")
            ax.set_ylabel("Count")

            ax.set_title(
                f"{lever.capitalize()} lever — {scale} scale"
            )

            if scale == "linear":
                ax.set_xlim(0, LINEAR_MAX)
            else:
                ax.set_xlim(RESOLUTION, LOG_MAX)

            ax.set_ylim(0, ylim[scale])
            ax.grid(alpha=0.15)

    fig.suptitle(
        f"{bird.upper()} — VI {vi_left}/{vi_right} — "
        f"{len(sessions)} sessions",
        fontsize=16,
        y=0.98,
    )

    fig.legend(
        handles=LEGEND,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    return fig


# --------------------------------------------------
# Generate PDFs
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    print("\nCalculating global axis limits...")
    ylim = calculate_limits(data)

    print(f"Linear y-axis: 0–{ylim['linear']}")
    print(f"Log y-axis:    0–{ylim['log']}")

    for bird in BIRDS:
        filename = OUTPUT_DIR / f"{bird}_irt_distributions.pdf"

        print(f"\nGenerating {filename}...")

        with PdfPages(filename) as pdf:
            for pair in VI_PAIRS:
                if pair not in data[bird]:
                    continue

                fig = plot_condition(
                    bird,
                    pair,
                    data[bird][pair],
                    ylim,
                )

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        print(f"Saved: {filename}")

    print("\nAll IRT-distribution PDFs generated successfully!")


if __name__ == "__main__":
    main()
