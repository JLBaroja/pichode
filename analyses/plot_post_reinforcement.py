"""Eight-panel post-response choice exploration, one PDF per bird."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# --------------------------------------------------
# Settings
# --------------------------------------------------

BASE = Path("data/processed/concurrent/responses")
OUTPUT_DIR = Path("results/figures/post_reinforcement")

BIRDS = ("p004", "p054", "p138", "p510", "p530", "p736")
COLORS = {"left": "tab:blue", "right": "tab:orange"}

MAX_LAG = 20
MAX_RUN = 40
FIGSIZE = (15, 19)

HIST_ALPHA = {0: 0.45, 1: 1.0}
LINESTYLES = {0: "--", 1: "-"}
LABELS = {0: "Unreinforced", 1: "Reinforced"}


# --------------------------------------------------
# Data preparation
# --------------------------------------------------

def valid_file(path):
    return not path.stem.endswith(("_bis", "_b"))


def load_bird(bird):
    frames = []

    for path in sorted(BASE.glob(f"{bird}_*.csv")):
        if not valid_file(path):
            continue

        df = pd.read_csv(path)

        if df.empty:
            continue

        choices = df["response"].to_numpy()
        n = len(choices)

        # Immediate next response, within session.
        stay = np.full(n, np.nan)
        stay[:-1] = (
            choices[:-1] == choices[1:]
        ).astype(float)

        df["stay"] = stay

        # Additional same-key responses before first switch.
        # Runs without a subsequent switch are censored.
        run = np.full(n, np.nan)
        next_switch = n

        for i in range(n - 2, -1, -1):
            if choices[i + 1] != choices[i]:
                next_switch = i + 1

            if next_switch < n:
                run[i] = next_switch - i - 1

        df["subsequent_run"] = run

        # Subsequent choices at each response lag.
        # Calculate BEFORE filtering by VI condition.
        for lag in range(1, MAX_LAG + 1):
            same = np.full(n, np.nan)

            if n > lag:
                same[:-lag] = (
                    choices[:-lag] == choices[lag:]
                ).astype(float)

            df[f"same_lag_{lag}"] = same

        df["session_file"] = path.stem
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No response files found for {bird}")

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------
# Plot one VI condition
# --------------------------------------------------

def plot_condition(data, bird, vi_left, vi_right):

    fig, axes = plt.subplots(4, 2, figsize=FIGSIZE)

    fig.suptitle(
        f"{bird} — VI {vi_left:g}/{vi_right:g}: Post-response behavior",
        fontsize=16,
    )

    for col, key in enumerate(("left", "right")):

        d = data[data["response"] == key]
        color = COLORS[key]

        # ------------------------------------------
        # Row 1: P(stay), pooled across sessions
        # ------------------------------------------

        ax = axes[0, col]

        pooled = d.groupby("reinforced")["stay"].mean()

        for reinforced in (0, 1):
            value = pooled.get(reinforced, np.nan)

            if pd.notna(value):
                ax.bar(
                    reinforced,
                    value,
                    color=color,
                    alpha=0.7,
                )

                ax.text(
                    reinforced,
                    min(value + 0.025, 0.98),
                    f"{value:.3f}",
                    ha="center",
                    fontsize=10,
                )

        ax.set(
            title=f"{key.capitalize()} key: P(stay)",
            ylabel="P(stay)",
            xticks=[0, 1],
            xticklabels=["Unreinforced", "Reinforced"],
            xlim=(-0.6, 1.6),
            ylim=(0, 1),
        )

        # ------------------------------------------
        # Row 2: Paired session-level P(stay)
        # ------------------------------------------

        ax = axes[1, col]

        session_probs = (
            d.groupby(["session_file", "reinforced"])["stay"]
            .mean()
            .unstack()
        )

        count = 0

        if 0 in session_probs.columns and 1 in session_probs.columns:

            paired = session_probs.dropna(subset=[0, 1])
            count = len(paired)

            for _, row in paired.iterrows():
                ax.plot(
                    [0, 1],
                    [row[0], row[1]],
                    color=color,
                    alpha=0.20,
                    linewidth=1,
                )

            ax.scatter(
                np.zeros(count),
                paired[0],
                color=color,
                alpha=0.5,
                s=15,
            )

            ax.scatter(
                np.ones(count),
                paired[1],
                color=color,
                alpha=0.5,
                s=15,
            )

        ax.set(
            title=f"{key.capitalize()} key: Session-level comparisons (n={count})",
            ylabel="P(stay)",
            xticks=[0, 1],
            xticklabels=["Unreinforced", "Reinforced"],
            xlim=(-0.2, 1.2),
            ylim=(0, 1),
        )

        # ------------------------------------------
        # Row 3: Subsequent run-length distributions
        # ------------------------------------------

        ax = axes[2, col]

        for reinforced in (0, 1):

            values = d.loc[
                d["reinforced"] == reinforced,
                "subsequent_run",
            ].dropna()

            if values.empty:
                continue

            ax.hist(
                values,
                bins=np.arange(-0.5, MAX_RUN + 1.5),
                density=True,
                histtype="step",
                linewidth=2,
                linestyle=LINESTYLES[reinforced],
                color=color,
                alpha=HIST_ALPHA[reinforced],
                label=f"{LABELS[reinforced]} (n={len(values):,})",
            )

        ax.set(
            title=f"{key.capitalize()} key: Subsequent run length",
            xlabel="Additional same-key responses before switching",
            ylabel="Probability density",
            xlim=(-0.5, MAX_RUN + 0.5),
        )

        ax.legend(fontsize=9)

        # ------------------------------------------
        # Row 4: Subsequent choice probabilities
        # ------------------------------------------

        ax = axes[3, col]

        for reinforced in (0, 1):

            subset = d[d["reinforced"] == reinforced]

            probabilities = [
                subset[f"same_lag_{lag}"].mean()
                for lag in range(1, MAX_LAG + 1)
            ]

            ax.plot(
                range(1, MAX_LAG + 1),
                probabilities,
                color=color,
                linestyle=LINESTYLES[reinforced],
                alpha=HIST_ALPHA[reinforced],
                marker="o" if reinforced else None,
                markersize=4,
                linewidth=2,
                label=LABELS[reinforced],
            )

        # Overall response allocation under this VI condition.
        baseline = (data["response"] == key).mean()

        ax.axhline(
            baseline,
            color="black",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label=f"Overall P({key}) = {baseline:.3f}",
        )

        ax.set(
            title=f"{key.capitalize()} key: Subsequent choice",
            xlabel="Responses after reference response",
            ylabel=f"P(choosing {key} key)",
            xlim=(1, MAX_LAG),
            ylim=(0, 1),
        )

        ax.legend(fontsize=9)

    for ax in axes.flat:
        ax.grid(alpha=0.15)

    fig.tight_layout(rect=(0, 0, 1, 0.985))

    return fig


# --------------------------------------------------
# Generate multipage PDFs
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for bird in BIRDS:

        data = load_bird(bird)

        conditions = (
            data[["vi_left", "vi_right"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["vi_left", "vi_right"])
        )

        output = OUTPUT_DIR / f"{bird}_post_reinforcement.pdf"

        with PdfPages(output) as pdf:

            for vi_left, vi_right in conditions.itertuples(
                index=False,
                name=None,
            ):

                subset = data.loc[
                    (data["vi_left"] == vi_left)
                    & (data["vi_right"] == vi_right)
                ]

                fig = plot_condition(
                    subset,
                    bird,
                    vi_left,
                    vi_right,
                )

                pdf.savefig(fig)
                plt.close(fig)

        print(
            f"{bird}: {len(conditions)} conditions, "
            f"{len(data):,} responses -> {output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
