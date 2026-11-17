"""Compare conditional posterior predictive cumulative records across models."""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import plot_predictive_cumulative_records as two
import plot_single_timescale_cumulative_records as single


OUTPUT_DIR = Path(
    "results/timescales/comparison/figures/cumulative_records"
)

BIRDS = ("p004", "p054", "p138", "p510", "p530", "p736")

COLORS = {
    "single": "darkviolet",
    "two": "steelblue",
}

N_DRAWS_DISPLAYED = 30


def add_predictions(ax, df, draws_path, model, color, seed):

    # Give each model its own reproducible random generator.
    model.RNG = np.random.default_rng(seed)

    left, right = model.predictive_trajectories(df, draws_path)

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

    df = two.prepare(df)
    segments = two.segment_information(df)

    match = re.fullmatch(
        r"(p[0-9]+)_s([0-9]+)_japede",
        path.stem,
    )

    if match is None:
        raise ValueError(f"Unexpected filename: {path.name}")

    bird, session_string = match.groups()
    session = int(session_string)

    filename = f"{bird}_s{session:03d}.csv"

    single_draws = single.DRAWS_DIR / filename
    two_draws = two.DRAWS_DIR / filename

    for draws_path in (single_draws, two_draws):
        if not draws_path.exists():
            raise FileNotFoundError(draws_path)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 7),
        constrained_layout=True,
    )

    fig.suptitle(
        f"{bird} — Session {session:03d}\n"
        f"{two.contingency_label(segments)}",
        fontsize=13,
    )

    comparisons = [
        (
            axes[0],
            single,
            single_draws,
            COLORS["single"],
            "Single-timescale model",
            0,
        ),
        (
            axes[1],
            two,
            two_draws,
            COLORS["two"],
            "Two-timescale model",
            1,
        ),
    ]

    for ax, model, draws_path, color, title, model_id in comparisons:

        # Identical observed record and matching benchmarks.
        two.plot_record(
            ax,
            df,
            segments,
            "N_left",
            "N_right",
            response_limit,
            is_response=True,
        )

        # Deterministic seed for each bird/session/model.
        bird_id = BIRDS.index(bird)
        seed = 123 + bird_id * 1000 + session * 2 + model_id

        add_predictions(
            ax,
            df,
            draws_path,
            model,
            color,
            seed,
        )

        ax.set(
            title=title,
            xlabel="Cumulative left responses",
            ylabel="Cumulative right responses",
        )

    return fig


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for bird in BIRDS:

        paths = two.session_files(bird)

        if not paths:
            print(f"{bird}: No sessions found.")
            continue

        response_limit, _ = two.bird_limits(paths)

        output = (
            OUTPUT_DIR
            / f"{bird}_comparative_cumulative_records.pdf"
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
