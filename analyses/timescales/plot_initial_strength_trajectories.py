"""Session trajectories of M1/M2 initial-condition parameters."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import numpy as np
import pandas as pd


ROOT = Path("results/timescales")
DRAWS = ROOT / "initial_conditions"
EVENTS = Path("data/processed/concurrent/events")
OUTPUT = ROOT / "comparison/figures/initial_strength_trajectories.pdf"

BIRDS = ("p004", "p054", "p138", "p510", "p530", "p736")

COLORS = {
    "M1": "forestgreen",
    "M2": "firebrick",
}

QUANTILES = (0.025, 0.5, 0.975)


def summarize(values):
    return np.quantile(values, QUANTILES)


def load_conditions():
    rows = []

    for path in sorted(EVENTS.glob("*_japede.csv")):
        d = pd.read_csv(
            path,
            usecols=["bird", "session", "session_type"],
            nrows=1,
        )
        if not d.empty:
            rows.append(d.iloc[0].to_dict())

    conditions = pd.DataFrame(rows)

    assert not conditions.duplicated(
        ["bird", "session"]
    ).any()

    return conditions


def load_parameters():
    rows = []

    for model in ("M1", "M2"):
        for bird in BIRDS:
            for session in range(1, 181):
                path = (
                    DRAWS / model
                    / f"{bird}_s{session:03d}"
                    / "posterior.csv"
                )

                posterior = pd.read_csv(path)

                # Calculate from joint posterior draws.
                strength = (
                    posterior["A0"].to_numpy()
                    / posterior["tau"].to_numpy()
                )

                q025, median, q975 = summarize(strength)

                row = {
                    "bird": bird,
                    "session": session,
                    "model": model,
                    "S0_q025": q025,
                    "S0_median": median,
                    "S0_q975": q975,
                }

                if model == "M2":
                    q025, median, q975 = summarize(
                        posterior["pi0"].to_numpy()
                    )

                    row.update({
                        "pi0_q025": q025,
                        "pi0_median": median,
                        "pi0_q975": q975,
                    })

                rows.append(row)

    return pd.DataFrame(rows)


def shade_dynamic(ax, data):
    for session in data.loc[
        data["session_type"] == "dynamic", "session"
    ]:
        ax.axvspan(
            session - 0.5,
            session + 0.5,
            color="orange",
            alpha=0.18,
            linewidth=0,
        )


def draw_parameter(ax, data, parameter, model):
    subset = data[
        data["model"] == model
    ].sort_values("session")

    x = subset["session"].to_numpy()

    ax.fill_between(
        x,
        subset[f"{parameter}_q025"].to_numpy(),
        subset[f"{parameter}_q975"].to_numpy(),
        color=COLORS[model],
        alpha=0.20,
    )

    ax.plot(
        x,
        subset[f"{parameter}_median"].to_numpy(),
        color=COLORS[model],
        linewidth=1.2,
        label=model,
    )

    flagged = subset[
        ~subset["preliminary_ok"].astype(bool)
    ]

    ax.scatter(
        flagged["session"],
        flagged[f"{parameter}_median"],
        marker="x",
        color="red",
        s=35,
        zorder=5,
    )


def main():
    conditions = load_conditions()

    parameters = load_parameters()

    summary = pd.read_csv(
        DRAWS / "parallel_summary.csv"
    )[["bird", "session", "model", "preliminary_ok"]]

    data = parameters.merge(
        summary,
        on=["bird", "session", "model"],
        validate="one_to_one",
    )

    data = data.merge(
        conditions,
        on=["bird", "session"],
        validate="many_to_one",
    )

    assert len(data) == 2160
    assert data["preliminary_ok"].notna().all()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(OUTPUT) as pdf:
        for bird in BIRDS:
            d = data[data["bird"] == bird]

            fig, axes = plt.subplots(
                4, 1,
                figsize=(13, 12),
                sharex=True,
                constrained_layout=True,
            )

            panels = [
                ("S0", "M1", "M1: Initial total strength"),
                ("S0", "M2", "M2: Initial total strength"),
                ("pi0", "M2", "M2: Initial preference"),
            ]

            for ax, (parameter, model, label) in zip(
                axes[:3], panels
            ):
                shade_dynamic(ax, d)
                draw_parameter(ax, d, parameter, model)

                ax.set_ylabel(label)
                ax.grid(alpha=0.2)

                if parameter == "S0":
                    ax.set_yscale("log")
                else:
                    ax.set_ylim(0, 1)
                    ax.axhline(
                        0.5,
                        color="gray",
                        linestyle=":",
                    )

            ax = axes[3]

            shade_dynamic(ax, d)

            for model in ("M1", "M2"):
                draw_parameter(ax, d, "S0", model)

            ax.set_ylabel("Initial strength comparison")
            ax.set_yscale("log")
            ax.grid(alpha=0.2)

            axes[-1].set_xlabel("Session")
            axes[-1].set_xlim(0.5, 180.5)

            legend = [
                Line2D(
                    [0], [0],
                    color=COLORS[model],
                    label=model,
                )
                for model in ("M1", "M2")
            ]

            legend.extend([
                Patch(
                    facecolor="orange",
                    alpha=0.25,
                    label="Dynamic session",
                ),
                Line2D(
                    [0], [0],
                    marker="x",
                    linestyle="none",
                    color="red",
                    label="Flagged fit",
                ),
            ])

            axes[3].legend(
                handles=legend,
                fontsize=9,
                ncol=4,
            )

            fig.suptitle(
                f"{bird} — Initial-condition parameter trajectories",
                fontsize=14,
            )

            pdf.savefig(fig)
            plt.close(fig)

            print(f"{bird}: plotted", flush=True)

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
