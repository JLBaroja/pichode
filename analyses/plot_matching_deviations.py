"""Explore deviations from matching in stable and dynamic sessions."""

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
OUTPUT_DIR = Path("results/figures/matching_deviations")

BIRDS = ("p004", "p054", "p138", "p510", "p530", "p736")

COLORS = {
    "Stable": "tab:green",
    "Dynamic (first half)": "tab:green",
    "Dynamic (second half)": "tab:purple",
}

GROUPS = (
    "Stable",
    "Dynamic (first half)",
    "Dynamic (second half)",
)

FIGSIZE = (13, 11)
HIST_BINS = 25

# Set to None for automatic limits.
LOG_RATIO_LIM = (-4, 4)
DEVIATION_LIM = (-4, 4)
HIST_XLIM = (-5, 5)
HIST_YLIM = (0, 50)


# --------------------------------------------------
# Load and classify sessions
# --------------------------------------------------

def load_bird(bird):

    rows = []

    for path in sorted(BASE.glob(f"{bird}_*.csv")):

        if path.stem.endswith(("_bis", "_b")):
            continue

        df = pd.read_csv(path)

        if df.empty:
            continue

        df = df.sort_values("time_sec").reset_index(drop=True)

        # Detect actual VI contingency changes.
        change = (
            df["vi_left"].ne(df["vi_left"].shift())
            | df["vi_right"].ne(df["vi_right"].shift())
        )

        segment = change.cumsum() - 1
        n_segments = segment.nunique()

        if n_segments == 1:
            groups = [("Stable", df)]

        elif n_segments == 2:
            groups = [
                ("Dynamic (first half)", df.loc[segment == 0]),
                ("Dynamic (second half)", df.loc[segment == 1]),
            ]

        else:
            print(
                f"Skipping {path.name}: "
                f"found {n_segments} VI segments."
            )
            continue

        for group_type, group in groups:

            if group.empty:
                continue

            left = group["response"] == "left"
            right = group["response"] == "right"

            rows.append({
                "session": path.stem,
                "type": group_type,
                "vi_left": group["vi_left"].iloc[0],
                "vi_right": group["vi_right"].iloc[0],
                "B_L": int(left.sum()),
                "B_R": int(right.sum()),
                "R_L": int(
                    (left & (group["reinforced"] == 1)).sum()
                ),
                "R_R": int(
                    (right & (group["reinforced"] == 1)).sum()
                ),
            })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Calculate matching statistics
# --------------------------------------------------

def calculate_matching(sessions):

    positive = (
        sessions[["B_L", "B_R", "R_L", "R_R"]] > 0
    ).all(axis=1)

    valid = sessions.loc[positive].copy()

    valid["log_B"] = np.log(
        valid["B_L"] / valid["B_R"]
    )

    valid["log_R"] = np.log(
        valid["R_L"] / valid["R_R"]
    )

    valid["D"] = valid["log_B"] - valid["log_R"]

    conditions = (
        sessions.groupby(["type", "vi_left", "vi_right"])
        .agg(
            B_L=("B_L", "sum"),
            B_R=("B_R", "sum"),
            R_L=("R_L", "sum"),
            R_R=("R_R", "sum"),
            n_sessions=("session", "nunique"),
        )
        .reset_index()
    )

    positive = (
        conditions[["B_L", "B_R", "R_L", "R_R"]] > 0
    ).all(axis=1)

    conditions = conditions.loc[positive].copy()

    conditions["log_B"] = np.log(
        conditions["B_L"] / conditions["B_R"]
    )

    conditions["log_R"] = np.log(
        conditions["R_L"] / conditions["R_R"]
    )

    conditions["D"] = (
        conditions["log_B"] - conditions["log_R"]
    )

    conditions["p_B"] = (
        conditions["B_L"]
        / (conditions["B_L"] + conditions["B_R"])
    )

    conditions["p_R"] = (
        conditions["R_L"]
        / (conditions["R_L"] + conditions["R_R"])
    )

    return valid, conditions


# --------------------------------------------------
# Scatterplot helper
# --------------------------------------------------

def scatter_groups(ax, conditions, x, y):

    for group_type in GROUPS:

        subset = conditions[
            conditions["type"] == group_type
        ]

        if subset.empty:
            continue

        color = COLORS[group_type]

        if group_type == "Stable":

            ax.scatter(
                subset[x],
                subset[y],
                color=color,
                marker="o",
                s=65,
                label=group_type,
                zorder=3,
            )

        elif group_type == "Dynamic (first half)":

            ax.scatter(
                subset[x],
                subset[y],
                facecolors="none",
                edgecolors=color,
                marker="o",
                linewidths=1.8,
                s=85,
                label=group_type,
                zorder=4,
            )

        else:

            ax.scatter(
                subset[x],
                subset[y],
                color=color,
                marker="x",
                linewidths=2,
                s=85,
                label=group_type,
                zorder=5,
            )


# --------------------------------------------------
# Plot one bird
# --------------------------------------------------

def plot_bird(bird, valid, conditions):

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)

    fig.suptitle(
        f"{bird} — Deviations from matching",
        fontsize=16,
    )

    # ----------------------------------------------
    # Top left: log response vs. reinforcement ratios
    # ----------------------------------------------

    ax = axes[0, 0]

    scatter_groups(ax, conditions, "log_R", "log_B")

    all_values = pd.concat([
        conditions["log_R"],
        conditions["log_B"],
    ])

    lo, hi = all_values.min(), all_values.max()

    ax.plot(
        [lo, hi],
        [lo, hi],
        "k--",
        label="Strict matching",
    )

    ax.set(
        title="Response allocation vs. reinforcement",
        xlabel="log(R_left / R_right)",
        ylabel="log(B_left / B_right)",
    )

    if LOG_RATIO_LIM is not None:
        ax.set_xlim(LOG_RATIO_LIM)
        ax.set_ylim(LOG_RATIO_LIM)

    ax.legend(fontsize=9)

    # ----------------------------------------------
    # Top right: deviations from matching
    # ----------------------------------------------

    ax = axes[0, 1]

    scatter_groups(ax, conditions, "log_R", "D")

    ax.axhline(
        0,
        color="black",
        linestyle="--",
    )

    ax.set(
        title="Deviation from strict matching",
        xlabel="log(R_left / R_right)",
        ylabel="D = log(B_L/B_R) - log(R_L/R_R)",
    )

    if LOG_RATIO_LIM is not None:
        ax.set_xlim(LOG_RATIO_LIM)

    if DEVIATION_LIM is not None:
        ax.set_ylim(DEVIATION_LIM)

    ax.legend(fontsize=9)

    # ----------------------------------------------
    # Bottom left: relative allocation
    # ----------------------------------------------

    ax = axes[1, 0]

    scatter_groups(ax, conditions, "p_R", "p_B")

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Strict matching",
    )

    ax.set(
        title="Relative response allocation",
        xlabel="Proportion of reinforcers on left",
        ylabel="Proportion of responses on left",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    ax.legend(fontsize=9)

    # ----------------------------------------------
    # Bottom right: session-level deviations
    # ----------------------------------------------

    ax = axes[1, 1]

    bins = np.linspace(
        HIST_XLIM[0],
        HIST_XLIM[1],
        HIST_BINS + 1,
    )

    for group_type in GROUPS:

        values = valid.loc[
            valid["type"] == group_type,
            "D",
        ]

        if values.empty:
            continue

        linestyle = (
            "--" if group_type == "Dynamic (first half)"
            else "-"
        )

        ax.hist(
            values,
            bins=bins,
            histtype="step",
            color=COLORS[group_type],
            linestyle=linestyle,
            linewidth=2,
            label=f"{group_type} (n={len(values)})",
        )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
    )

    ax.set(
        title="Session-level deviations",
        xlabel="D",
        ylabel="Number of sessions",
    )

    if HIST_XLIM is not None:
        ax.set_xlim(HIST_XLIM)

    if HIST_YLIM is not None:
        ax.set_ylim(HIST_YLIM)

    ax.legend(fontsize=9)

    # ----------------------------------------------
    # Finish
    # ----------------------------------------------

    for ax in axes.flat:
        ax.grid(alpha=0.15)

    fig.tight_layout()

    return fig


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = OUTPUT_DIR / "matching_deviations.pdf"

    with PdfPages(output) as pdf:

        for bird in BIRDS:

            sessions = load_bird(bird)

            if sessions.empty:
                print(f"{bird}: No sessions found.")
                continue

            valid, conditions = calculate_matching(sessions)

            if valid.empty or conditions.empty:
                print(f"{bird}: No valid matching observations.")
                continue

            fig = plot_bird(bird, valid, conditions)

            pdf.savefig(fig)
            plt.close(fig)

            print(
                f"{bird}: "
                f"{sessions['session'].nunique()} sessions, "
                f"{len(conditions)} condition/group points",
                flush=True,
            )

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
