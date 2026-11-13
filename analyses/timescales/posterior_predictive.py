from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from prepare_session import prepare_session


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BIRD = "p530"

N_DRAWS = 200
WINDOW = 100
SEED = 123

ROOT = Path("results/timescales/preliminary")
DRAWS_DIR = ROOT / "draws"
EVENTS_DIR = Path("data/processed/concurrent/events")

OUTPUT = ROOT / "figures" / f"{BIRD}_posterior_predictive.pdf"

rng = np.random.default_rng(SEED)


# --------------------------------------------------
# Find stable and dynamic sessions
# --------------------------------------------------

def find_sessions(bird):

    summary = pd.read_csv(ROOT / "summary.csv")

    summary = summary[
        (summary["bird"] == bird)
        & (summary["preliminary_ok"].astype(str).str.lower() == "true")
    ]

    selected = {}

    for session in sorted(summary["session"].unique()):

        path = EVENTS_DIR / f"{bird}_s{session:03d}_japede.csv"
        draws_path = DRAWS_DIR / f"{bird}_s{session:03d}.csv"

        if not path.exists() or not draws_path.exists():
            continue

        condition = pd.read_csv(
            path,
            usecols=["session_type"],
            nrows=1,
        )["session_type"].iloc[0]

        if condition in ("stable", "dynamic"):
            if condition not in selected:
                selected[condition] = int(session)

        if len(selected) == 2:
            break

    if len(selected) != 2:
        raise RuntimeError(
            f"Could not find both session types for {bird}"
        )

    return selected


# --------------------------------------------------
# Posterior predictive simulation
# --------------------------------------------------

def simulate_session(bird, session):

    df, data = prepare_session(bird, session)

    draws_path = DRAWS_DIR / f"{bird}_s{session:03d}.csv"

    posterior = pd.read_csv(draws_path)

    required = ["tau_fast", "tau_slow", "w"]

    if not set(required).issubset(posterior.columns):
        raise ValueError(
            f"Missing parameters in {draws_path}: "
            f"{set(required) - set(posterior.columns)}"
        )

    posterior = posterior[required].dropna()

    n = min(N_DRAWS, len(posterior))

    posterior = posterior.sample(
        n=n,
        random_state=SEED,
    )

    tau_fast = posterior["tau_fast"].to_numpy()
    tau_slow = posterior["tau_slow"].to_numpy()
    w = posterior["w"].to_numpy()

    q_fast = 1 / tau_fast
    q_slow = 1 / tau_slow

    N = data["N"]

    r_left = np.asarray(data["r_left"])
    r_right = np.asarray(data["r_right"])

    observed = np.asarray(data["y"])

    # Initialize reinforcement traces exactly as in Stan.

    IL_fast = np.full(n, 0.5)
    IR_fast = np.full(n, 0.5)

    IL_slow = np.full(n, 0.5)
    IR_slow = np.full(n, 0.5)

    predictions = np.zeros((n, N), dtype=int)
    probabilities = np.zeros((n, N))

    for t in range(N):

        # Combined reinforcement traces.

        IL = (1 - w) * IL_fast + w * IL_slow
        IR = (1 - w) * IR_fast + w * IR_slow

        # Choice probability BEFORE reinforcement update.

        p_left = IL / (IL + IR)

        probabilities[:, t] = p_left

        # Simulate choices from posterior predictive distribution.

        predictions[:, t] = rng.binomial(1, p_left)

        # Update reinforcement traces.

        IL_fast = (1 - q_fast) * IL_fast + q_fast * r_left[t]
        IR_fast = (1 - q_fast) * IR_fast + q_fast * r_right[t]

        IL_slow = (1 - q_slow) * IL_slow + q_slow * r_left[t]
        IR_slow = (1 - q_slow) * IR_slow + q_slow * r_right[t]

    return observed, predictions, probabilities


# --------------------------------------------------
# Cumulative records
# --------------------------------------------------

def plot_cumulative(ax, observed, predictions):

    N = len(observed)

    # Observed cumulative responses.

    obs_left = np.concatenate([[0], np.cumsum(observed)])
    obs_right = np.arange(N + 1) - obs_left

    # Posterior predictive cumulative responses.

    pred_left = np.concatenate(
        [
            np.zeros((len(predictions), 1)),
            np.cumsum(predictions, axis=1),
        ],
        axis=1,
    )

    pred_right = np.arange(N + 1) - pred_left

    # Individual posterior predictive trajectories.

    for i in range(min(60, len(predictions))):
        ax.plot(
            pred_left[i],
            pred_right[i],
            color="steelblue",
            alpha=0.08,
            linewidth=0.8,
        )

    # Median posterior predictive trajectory.

    ax.plot(
        np.median(pred_left, axis=0),
        np.median(pred_right, axis=0),
        color="blue",
        linewidth=2,
        label="Posterior predictive median",
    )

    # Actual cumulative record.

    ax.plot(
        obs_left,
        obs_right,
        color="black",
        linewidth=2,
        label="Observed",
    )

    ax.set_xlabel("Cumulative left responses")
    ax.set_ylabel("Cumulative right responses")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend()


# --------------------------------------------------
# Local choice proportions
# --------------------------------------------------

def plot_local_choice(ax, observed, predictions):

    N = len(observed)

    centers = []
    observed_props = []
    predicted_props = []

    for start in range(0, N, WINDOW):

        end = min(start + WINDOW, N)

        centers.append((start + end) / 2)

        observed_props.append(
            observed[start:end].mean()
        )

        predicted_props.append(
            predictions[:, start:end].mean(axis=1)
        )

    centers = np.asarray(centers)
    predicted_props = np.asarray(predicted_props)

    median = np.median(predicted_props, axis=1)
    lower = np.quantile(predicted_props, 0.025, axis=1)
    upper = np.quantile(predicted_props, 0.975, axis=1)

    ax.fill_between(
        centers,
        lower,
        upper,
        color="steelblue",
        alpha=0.25,
        label="95% posterior predictive interval",
    )

    ax.plot(
        centers,
        median,
        color="blue",
        linewidth=2,
        label="Posterior predictive median",
    )

    ax.plot(
        centers,
        observed_props,
        color="black",
        linewidth=1.5,
        label="Observed",
    )

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Response number")
    ax.set_ylabel("Proportion choosing left")

    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    sessions = find_sessions(BIRD)

    print(f"\nSelected sessions for {BIRD}:")
    print(sessions)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 12),
        constrained_layout=True,
    )

    for column, condition in enumerate(["stable", "dynamic"]):

        session = sessions[condition]

        print(
            f"Simulating {BIRD} session {session} ({condition})..."
        )

        observed, predictions, probabilities = simulate_session(
            BIRD,
            session,
        )

        plot_cumulative(
            axes[0, column],
            observed,
            predictions,
        )

        plot_local_choice(
            axes[1, column],
            observed,
            predictions,
        )

        axes[0, column].set_title(
            f"{BIRD} — Session {session} ({condition})\n"
            "Cumulative choice record"
        )

        axes[1, column].set_title(
            f"Local choice allocation — {condition}"
        )

    fig.suptitle(
        f"{BIRD} — Posterior predictive checks",
        fontsize=16,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(OUTPUT)
    plt.close(fig)

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
