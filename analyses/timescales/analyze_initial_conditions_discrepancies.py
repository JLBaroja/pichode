"""Temporal choice-allocation discrepancies for M1 and M2."""

from pathlib import Path
import re

import numpy as np
import pandas as pd

import plot_predictive_cumulative_records as base


ROOT = Path("results/timescales/initial_conditions")
OUTPUT = ROOT / "temporal_discrepancies"

BIRDS = ("p004", "p054", "p138", "p510", "p530", "p736")

WINDOW_MIN = 5
MAX_DRAWS = 200
SEED = 123


def predicted_probabilities(df, posterior, model):
    """Posterior mean P(left) at each observed response."""

    tau = posterior["tau"].to_numpy()
    A0 = posterior["A0"].to_numpy()
    q = 1.0 / tau

    n = len(posterior)

    if model == "M1":
        IL = q * A0 * 0.5
        IR = q * A0 * 0.5

    elif model == "M2":
        pi0 = posterior["pi0"].to_numpy()
        IL = q * A0 * pi0
        IR = q * A0 * (1 - pi0)

    else:
        raise ValueError(model)

    left = df["response"].eq("left").to_numpy()
    right = df["response"].eq("right").to_numpy()
    reinforced = df["reinforced"].eq(1).to_numpy()

    r_left = left & reinforced
    r_right = right & reinforced

    predictions = np.empty(len(df))

    for t in range(len(df)):
        denominator = IL + IR

        p = np.divide(
            IL,
            denominator,
            out=np.full(n, 0.5),
            where=denominator > 0,
        )

        predictions[t] = p.mean()

        # Update AFTER prediction, using observed reinforcement.
        IL = (1 - q) * IL + q * r_left[t]
        IR = (1 - q) * IR + q * r_right[t]

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

    session_types = df["session_type"].dropna().unique()

    if len(session_types) != 1:
        raise ValueError(
            f"Unexpected session types: {path}"
        )

    session_type = session_types[0]

    events_path = (
        Path("data/processed/concurrent/events") / path.name
    )

    events = pd.read_csv(events_path, nrows=1)

    if session_type == "dynamic":
        first_change_sec = float(
            events["first_cp_left"].iloc[0]
        )
        second_change_sec = float(
            events["second_cp_left"].iloc[0]
        )
    else:
        first_change_sec = np.nan
        second_change_sec = np.nan

    # Restrict analysis to first 60 minutes.
    df = df[
        (df["time_sec"] >= 0)
        & (df["time_sec"] < 3600)
    ].copy()

    if df.empty:
        return []

    window = (
        df["time_sec"].to_numpy() // (WINDOW_MIN * 60)
    ).astype(int)

    observed = df["response"].eq("left").to_numpy()

    rows = []

    for model in ("M1", "M2"):
        draws_path = (
            ROOT / model
            / f"{bird}_s{session:03d}"
            / "posterior.csv"
        )

        posterior = pd.read_csv(draws_path).dropna()

        if len(posterior) > MAX_DRAWS:
            indices = rng.choice(
                len(posterior),
                size=MAX_DRAWS,
                replace=False,
            )
            posterior = posterior.iloc[indices]

        predicted = predicted_probabilities(
            df, posterior, model
        )

        for k in np.unique(window):
            mask = window == k

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
                "n_responses": int(mask.sum()),
                "observed_left": observed_left,
                "predicted_left": predicted_left,
                "signed_error": discrepancy,
                "absolute_error": abs(discrepancy),
                "first_change_min": first_change_sec / 60,
                "second_change_min": second_change_sec / 60,
            })

    return rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    rows = []

    for bird in BIRDS:
        paths = base.session_files(bird)

        for path in paths:
            rows.extend(analyze_session(path, rng))

        print(
            f"{bird}: {len(paths)} sessions processed",
            flush=True,
        )

    data = pd.DataFrame(rows)

    output = OUTPUT / "window_discrepancies.csv"
    data.to_csv(output, index=False)

    print(f"\nSaved {len(data)} rows to {output}")


if __name__ == "__main__":
    main()
