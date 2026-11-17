"""Plot individual and mean temporal prediction discrepancies."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("results/timescales/initial_conditions/temporal_discrepancies")
DATA = BASE / "window_discrepancies.csv"
EVENTS = Path("data/processed/concurrent/events")

COLORS = {"M1": "forestgreen", "M2": "firebrick"}
LABELS = {"M1": "Initial strength (M1)", "M2": "Initial preference (M2)"}

def add_change_times(data):
    """Attach the first programmed change point to dynamic sessions."""
    changes = []

    for (bird, session), group in data.groupby(["bird", "session"]):
        if group["session_type"].iloc[0] != "dynamic":
            continue

        path = EVENTS / f"{bird}_s{session:03d}_japede.csv"
        event = pd.read_csv(path, nrows=1).iloc[0]

        changes.append({
            "bird": bird,
            "session": session,
            "change_min": float(event["first_cp_left"]) / 60,
        })

    return data.merge(
        pd.DataFrame(changes),
        on=["bird", "session"],
        how="left",
    )


def plot_spaghetti(data, session_type, metric, filename):
    """Thin individual trajectories with a thick mean trajectory."""

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5), sharey=True
    )

    for ax, model in zip(axes, ("M1", "M2")):
        subset = data[
            (data["session_type"] == session_type)
            & (data["model"] == model)
        ].copy()

        if session_type == "dynamic":
            subset["plot_time"] = (
                subset["midpoint_min"] - subset["change_min"]
            )
        else:
            subset["plot_time"] = subset["midpoint_min"]

        for (_, _), group in subset.groupby(["bird", "session"]):
            group = group.sort_values("plot_time")
            ax.plot(
                group["plot_time"],
                group[metric],
                color=COLORS[model],
                alpha=0.08,
                linewidth=0.6,
            )

        # Mean across sessions in five-minute bins.
        subset["bin"] = (
            np.floor(subset["plot_time"] / 5) * 5 + 2.5
        )

        summary = (
            subset.groupby("bin")[metric]
            .mean()
            .sort_index()
        )

        ax.plot(
            summary.index,
            summary.values,
            color=COLORS[model],
            linewidth=2.8,
            label="Mean",
        )

        if session_type == "dynamic":
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set_xlabel("Minutes relative to first contingency change")
        else:
            ax.set_xlabel("Minutes since session start")

        if metric == "signed_error":
            ax.axhline(0, color="gray", linestyle=":", linewidth=1)

        ax.set_title(LABELS[model])
        ax.grid(alpha=0.15)
        ax.legend()

    axes[0].set_ylabel(
        "Absolute allocation discrepancy"
        if metric == "absolute_error"
        else "Observed minus predicted P(left)"
    )

    fig.suptitle(f"{session_type.capitalize()} sessions")
    fig.tight_layout()
    fig.savefig(BASE / filename)
    plt.close(fig)

def main():
    data = pd.read_csv(DATA)

    # Restrict all analyses to the first 60 minutes.
    data = data[
        (data["start_min"] >= 0)
        & (data["start_min"] < 60)
    ].copy()

    data = add_change_times(data)

    data["relative_time_min"] = np.where(
        data["session_type"] == "dynamic",
        data["midpoint_min"] - data["change_min"],
        np.nan,
    )

    data.to_csv(BASE / "window_discrepancies_60min.csv", index=False)

    for session_type in ("stable", "dynamic"):
        for metric in ("signed_error", "absolute_error"):
            filename = f"{session_type}_{metric}_individual_and_mean.pdf"
            plot_spaghetti(data, session_type, metric, filename)
            print(f"Saved {filename}")

    print("\nDone.")


if __name__ == "__main__":
    main()
