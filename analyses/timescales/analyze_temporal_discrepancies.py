"""Compare model choice-allocation discrepancies across session time."""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_predictive_cumulative_records as two

OUTPUT_DIR = Path("results/timescales/comparison/temporal_discrepancies")
SINGLE_DRAWS = Path("results/timescales/single_timescale/draws")
TWO_DRAWS = Path("results/timescales/preliminary/draws")

BIRDS = ("p004", "p054", "p138", "p510", "p530", "p736")

WINDOW_MIN = 5
MAX_DRAWS = 200
SEED = 123

COLORS = {
    "single": "darkviolet",
    "two": "steelblue",
}


def predicted_probabilities(df, posterior, model):
    """Posterior mean P(left) at each observed response."""

    n = len(posterior)

    if model == "single":
        q = 1.0 / posterior["tau"].to_numpy()
        IL = np.full(n, 0.5)
        IR = np.full(n, 0.5)
    else:
        qf = 1.0 / posterior["tau_fast"].to_numpy()
        qs = 1.0 / posterior["tau_slow"].to_numpy()
        w = posterior["w"].to_numpy()

        ILf = np.full(n, 0.5)
        IRf = np.full(n, 0.5)
        ILs = np.full(n, 0.5)
        IRs = np.full(n, 0.5)

    left = df["response"].eq("left").to_numpy()
    right = df["response"].eq("right").to_numpy()
    reinforced = df["reinforced"].eq(1).to_numpy()

    r_left = left & reinforced
    r_right = right & reinforced

    predictions = np.empty(len(df))

    for t in range(len(df)):

        if model == "single":
            income_left = IL
            income_right = IR
        else:
            income_left = (1 - w) * ILf + w * ILs
            income_right = (1 - w) * IRf + w * IRs

        denominator = income_left + income_right

        p = np.divide(
            income_left,
            denominator,
            out=np.full(n, 0.5),
            where=denominator > 0,
        )

        predictions[t] = p.mean()

        if model == "single":
            IL = (1 - q) * IL + q * r_left[t]
            IR = (1 - q) * IR + q * r_right[t]
        else:
            ILf = (1 - qf) * ILf + qf * r_left[t]
            IRf = (1 - qf) * IRf + qf * r_right[t]
            ILs = (1 - qs) * ILs + qs * r_left[t]
            IRs = (1 - qs) * IRs + qs * r_right[t]

    return predictions


def analyze_session(path, rng):

    match = re.fullmatch(
        r"(p[0-9]+)_s([0-9]+)_japede",
        path.stem,
    )

    if match is None:
        raise ValueError(path.name)

    bird, session_string = match.groups()
    session = int(session_string)

    df = pd.read_csv(path).sort_values(
        "time_sec", kind="stable"
    ).reset_index(drop=True)

    if df.empty:
        return []

    # Use the experimental classification recorded in the data.
    session_types = df["session_type"].dropna().unique()
    
    if len(session_types) != 1:
        raise ValueError(
            f"Unexpected session types in {path}: {session_types}"
        )
    
    session_type = session_types[0]

    # Read programmed contingency-change times from the event file.
    events_path = Path(
        "data/processed/concurrent/events"
    ) / path.name
    
    events = pd.read_csv(events_path, nrows=1)
    
    if session_type == "dynamic":
        first_change_sec = float(events["first_cp_left"].iloc[0])
        second_change_sec = float(events["second_cp_left"].iloc[0])
    else:
        first_change_sec = np.nan
        second_change_sec = np.nan

    window = (
        df["time_sec"].to_numpy() // (WINDOW_MIN * 60)
    ).astype(int)

    observed = df["response"].eq("left").to_numpy()

    rows = []
    filename = f"{bird}_s{session:03d}.csv"

    for model, draws_dir in (
        ("single", SINGLE_DRAWS),
        ("two", TWO_DRAWS),
    ):
        draws_path = draws_dir / filename

        posterior = pd.read_csv(draws_path).dropna()

        if len(posterior) > MAX_DRAWS:
            indices = rng.choice(
                len(posterior), size=MAX_DRAWS, replace=False
            )
            posterior = posterior.iloc[indices]

        predicted = predicted_probabilities(df, posterior, model)

        for k in np.unique(window):
            mask = window == k
            n = int(mask.sum())

            observed_left = observed[mask].mean()
            predicted_left = predicted[mask].mean()
            discrepancy = observed_left - predicted_left

            rows.append({
                "bird": bird,
                "session": session,
                "session_type": session_type,
                "model": model,
                "window": int(k),
                "start_min": int(k * WINDOW_MIN),
                "midpoint_min": (k + 0.5) * WINDOW_MIN,
                "n_responses": n,
                "observed_left": observed_left,
                "predicted_left": predicted_left,
                "signed_error": discrepancy,
                "absolute_error": abs(discrepancy),
                "first_change_min": first_change_sec/60,
                "second_change_min": second_change_sec/60,
            })

    return rows


def plot_summary(data, column, ylabel, filename):

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), sharey=True
    )

    for ax, session_type in zip(
        axes, ("stable", "dynamic")
    ):
        subset = data[data["session_type"] == session_type]

        for model in ("single", "two"):
            group = subset[subset["model"] == model]

            # Each session contributes once per available window.
            summary = (
                group.groupby("midpoint_min")[column]
                .agg(["mean", "count"])
                .reset_index()
            )

            ax.plot(
                summary["midpoint_min"],
                summary["mean"],
                marker="o",
                markersize=3,
                linewidth=1.8,
                color=COLORS[model],
                label=(
                    "Single-timescale"
                    if model == "single"
                    else "Two-timescale"
                ),
            )

        ax.set(
            title=f"{session_type.capitalize()} sessions",
            xlabel="Minutes since session start",
        )
        ax.grid(alpha=0.2)
        ax.legend()

    axes[0].set_ylabel(ylabel)

    if column == "signed_error":
        for ax in axes:
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows = []

    for bird in BIRDS:
        paths = two.session_files(bird)

        for path in paths:
            rows.extend(analyze_session(path, rng))

        print(f"{bird}: {len(paths)} sessions processed", flush=True)

    data = pd.DataFrame(rows)

    output = OUTPUT_DIR / "window_discrepancies.csv"
    data.to_csv(output, index=False)

    plot_summary(
        data,
        "absolute_error",
        "Mean absolute allocation discrepancy",
        "absolute_discrepancy_by_time.pdf",
    )

    plot_summary(
        data,
        "signed_error",
        "Mean observed minus predicted P(left)",
        "signed_discrepancy_by_time.pdf",
    )

    print(f"\nSaved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
