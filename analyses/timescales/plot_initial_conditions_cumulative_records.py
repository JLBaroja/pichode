"""Conditional posterior predictive cumulative records for M1 and M2."""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import plot_predictive_cumulative_records as base


ROOT = Path("results/timescales/initial_conditions")
OUTPUT_DIR = ROOT / "figures/cumulative_records"

BIRDS = ("p004", "p054", "p138", "p510", "p530", "p736")
COLORS = {"M1": "forestgreen", "M2": "firebrick"}

N_PREDICTIVE_DRAWS = 100
N_DRAWS_DISPLAYED = 30


def predictive_trajectories(df, draws_path, model, rng):
    columns = ["tau", "A0"]
    if model == "M2":
        columns.append("pi0")

    posterior = pd.read_csv(draws_path, usecols=columns).dropna()

    if posterior.empty:
        raise ValueError(f"No posterior draws: {draws_path}")

    indices = rng.choice(
        len(posterior),
        size=min(N_PREDICTIVE_DRAWS, len(posterior)),
        replace=False,
    )

    draws = posterior.iloc[indices]

    tau = draws["tau"].to_numpy()
    A0 = draws["A0"].to_numpy()
    q = 1.0 / tau

    n_draws = len(draws)
    n_responses = len(df)

    left = df["response"].eq("left").to_numpy()
    right = df["response"].eq("right").to_numpy()
    reinforced = df["reinforced"].eq(1).to_numpy()

    r_left = left & reinforced
    r_right = right & reinforced

    if model == "M1":
        IL = q * A0 * 0.5
        IR = q * A0 * 0.5
    else:
        pi0 = draws["pi0"].to_numpy()
        IL = q * A0 * pi0
        IR = q * A0 * (1 - pi0)

    simulated_left = np.zeros(
        (n_draws, n_responses), dtype=np.int8
    )

    for t in range(n_responses):
        denominator = IL + IR

        p_left = np.divide(
            IL,
            denominator,
            out=np.full(n_draws, 0.5),
            where=denominator > 0,
        )

        simulated_left[:, t] = rng.binomial(1, p_left)

        # Conditional prediction: use observed reinforcers.
        IL = (1 - q) * IL + q * r_left[t]
        IR = (1 - q) * IR + q * r_right[t]

    cumulative_left = np.column_stack([
        np.zeros(n_draws),
        np.cumsum(simulated_left, axis=1),
    ])

    cumulative_right = (
        np.arange(n_responses + 1)[None, :] - cumulative_left
    )

    return cumulative_left, cumulative_right


def add_predictions(ax, df, draws_path, model, color, seed):
    rng = np.random.default_rng(seed)

    left, right = predictive_trajectories(
        df, draws_path, model, rng
    )

    for i in range(min(N_DRAWS_DISPLAYED, len(left))):
        ax.plot(
            left[i],
            right[i],
            color=color,
            alpha=0.07,
            linewidth=0.8,
            zorder=1,
        )

    ax.plot(
        np.median(left, axis=0),
        np.median(right, axis=0),
        color=color,
        linewidth=1.8,
        label="Posterior predictive median",
        zorder=4,
    )

    ax.legend(fontsize=8)


def plot_session(path, response_limit):
    df = pd.read_csv(path)

    if df.empty:
        return None

    df = base.prepare(df)
    segments = base.segment_information(df)

    match = re.fullmatch(
        r"(p[0-9]+)_s([0-9]+)_japede",
        path.stem,
    )

    if match is None:
        raise ValueError(f"Unexpected filename: {path.name}")

    bird, session_string = match.groups()
    session = int(session_string)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 7),
        constrained_layout=True,
    )

    fig.suptitle(
        f"{bird} — Session {session:03d}\n"
        f"{base.contingency_label(segments)}",
        fontsize=13,
    )

    for model_id, (ax, model) in enumerate(
        zip(axes, ("M1", "M2"))
    ):
        draws_path = (
            ROOT / model
            / f"{bird}_s{session:03d}"
            / "posterior.csv"
        )

        if not draws_path.exists():
            raise FileNotFoundError(draws_path)

        base.plot_record(
            ax,
            df,
            segments,
            "N_left",
            "N_right",
            response_limit,
            is_response=True,
        )

        seed = (
            123
            + BIRDS.index(bird) * 1000
            + session * 2
            + model_id
        )

        add_predictions(
            ax,
            df,
            draws_path,
            model,
            COLORS[model],
            seed,
        )

        ax.set(
            title=f"{model}: "
                  + ("Initial strength" if model == "M1"
                     else "Initial preference"),
            xlabel="Cumulative left responses",
            ylabel="Cumulative right responses",
        )

    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for bird in BIRDS:
        paths = base.session_files(bird)

        if not paths:
            print(f"{bird}: No sessions found.")
            continue

        response_limit, _ = base.bird_limits(paths)

        output = (
            OUTPUT_DIR
            / f"{bird}_initial_conditions_cumulative_records.pdf"
        )

        pages = 0

        with PdfPages(output) as pdf:
            for path in paths:
                fig = plot_session(path, response_limit)

                if fig is None:
                    continue

                pdf.savefig(fig)
                plt.close(fig)
                pages += 1

        print(f"{bird}: {pages} pages -> {output}")


if __name__ == "__main__":
    main()
