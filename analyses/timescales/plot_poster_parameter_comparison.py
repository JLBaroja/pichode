"""Poster figures: slow-component weights and cross-model timescales."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_colors import MODEL_COLORS, BIRD_COLORS
from plot_style import (
    apply_style, style_axes, add_labels,
    add_title, save_figure,
)


ROOT = Path("results/timescales")
EVENTS = Path("data/processed/concurrent/events")
OUTPUT = ROOT / "poster/figures"

BIRDS = ("p004", "p054", "p138", "p510", "p530", "p736")
MODELS = ("M0", "two_timescales", "M1", "M2")

# Poster layout
FIGSIZE = (9.5, 7.0)
DYNAMIC_COLOR = "#888888"
DYNAMIC_ALPHA = 0.16
INTERVAL_ALPHA = 0.13
WEIGHT_INTERVAL_ALPHA = 0.24


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

    assert len(conditions) == 1080
    assert not conditions.duplicated(["bird", "session"]).any()
    assert conditions["session_type"].isin(
        ["stable", "dynamic"]
    ).all()

    return conditions


def load_parameters():
    rows = []

    single = pd.read_csv(
        ROOT / "single_timescale/summary.csv"
    )

    for row in single.itertuples():
        rows.append({
            "bird": row.bird,
            "session": row.session,
            "model": "M0",
            "median": row.tau_median,
            "q025": row.tau_q025,
            "q975": row.tau_q975,
        })

    two = pd.read_csv(
        ROOT / "preliminary/summary.csv"
    )

    for row in two.itertuples():
        rows.append({
            "bird": row.bird,
            "session": row.session,
            "model": "two_timescales",
            "median": row.tau_slow_median,
            "q025": row.tau_slow_q025,
            "q975": row.tau_slow_q975,
        })

    for model in ("M1", "M2"):
        for bird in BIRDS:
            for session in range(1, 181):
                path = (
                    ROOT / "initial_conditions" / model
                    / f"{bird}_s{session:03d}"
                    / "posterior.csv"
                )

                posterior = pd.read_csv(
                    path,
                    usecols=["tau"],
                )

                q025, median, q975 = np.quantile(
                    posterior["tau"].to_numpy(),
                    [0.025, 0.5, 0.975],
                )

                rows.append({
                    "bird": bird,
                    "session": session,
                    "model": model,
                    "median": median,
                    "q025": q025,
                    "q975": q975,
                })

    parameters = pd.DataFrame(rows)

    assert len(parameters) == 4320
    assert not parameters.duplicated(
        ["bird", "session", "model"]
    ).any()

    return parameters


def shade_dynamic(ax, conditions):
    sessions = conditions.loc[
        conditions["session_type"] == "dynamic",
        "session",
    ]

    for session in sessions:
        ax.axvspan(
            session - 0.5,
            session + 0.5,
            color=DYNAMIC_COLOR,
            alpha=DYNAMIC_ALPHA,
            linewidth=0,
            zorder=0,
        )




def make_panels(title, ylabel, ylim):
    # One layout controls the header and all six panels.
    fig = plt.figure(
        figsize=(9.5, 7.0),
        layout="constrained",
    )

    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[0.35, 1, 1, 1],
        hspace=0.06,
        wspace=0.04,
    )

    # Dedicated header: title and legend share the same space.
    header = fig.add_subplot(gs[0, :])
    header.set_axis_off()

    header.text(
        0.5, 0.88,
        title,
        ha="center",
        va="center",
        fontsize=17,
        transform=header.transAxes,
    )

    fig._poster_header = header

    axes = np.empty((3, 2), dtype=object)

    for row in range(3):
        for col in range(2):
            axes[row, col] = fig.add_subplot(
                gs[row + 1, col],
                sharex=axes[0, 0] if row or col else None,
                sharey=axes[0, 0] if row or col else None,
            )

    add_labels(
        fig, axes,
        xlabel="Session",
        ylabel=ylabel,
    )

    for ax, bird in zip(axes.flat, BIRDS):
        style_axes(ax, vertical_y_ticks=True)

        ax.set_xlim(0.5, 180.5)
        ax.set_ylim(*ylim)
        ax.set_xticks(np.arange(20, 181, 20))

        ax.set_title(
            bird,
            color=BIRD_COLORS[bird],
            fontsize=12,
            fontweight="bold",
            pad=5,
        )

        ax.grid(alpha=0.12, linewidth=0.5)

    for ax in axes[:-1, :].flat:
        ax.tick_params(labelbottom=False)

    return fig, axes


def add_legend(fig, handles):
    fig._poster_header.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        bbox_transform=fig._poster_header.transAxes,
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        handlelength=2.2,
        columnspacing=1.7,
        handletextpad=0.6,
    )


def plot_weights(conditions):
    two = pd.read_csv(
        ROOT / "preliminary/summary.csv"
    )

    fig, axes = make_panels(
        "Slow-component weight",
        "Slow-component weight",
        (0, 1),
    )

    color = MODEL_COLORS["two_timescales"]

    for ax, bird in zip(axes.flat, BIRDS):
        d = two.loc[
            two["bird"] == bird
        ].sort_values("session")

        c = conditions.loc[
            conditions["bird"] == bird
        ]

        shade_dynamic(ax, c)

        x = d["session"].to_numpy()

        ax.fill_between(
            x,
            d["w_q025"].to_numpy(),
            d["w_q975"].to_numpy(),
            color=color,
            alpha=WEIGHT_INTERVAL_ALPHA,
            linewidth=0,
            zorder=1,
        )

        ax.plot(
            x,
            d["w_median"].to_numpy(),
            color=color,
            linewidth=1.2,
            zorder=2,
        )

        # Mark every session with a small orange dot.
        ax.scatter(
            x,
            d["w_median"].to_numpy(),
            s=15,
            color=color,
            alpha=0.85,
            edgecolors="none",
            zorder=3,
        )

        # Mark sessions with problematic fit diagnostics.
        flagged = d.loc[
            ~d["preliminary_ok"].astype(str).str.lower().eq("true")
        ]

        ax.scatter(
            flagged["session"],
            flagged["w_median"],
            marker="x",
            s=9,
            linewidths=0.45,
            color="black",
            alpha=0.8,
            zorder=5,
        )

        ax.set_yticks(
            np.arange(0, 1.01, 0.2)
        )

    add_legend(
        fig,
        [
            Line2D(
                [0], [0],
                color=color,
                lw=2,
                label="Posterior median",
            ),
            Patch(
                facecolor=color,
                alpha=WEIGHT_INTERVAL_ALPHA,
                label="95% credible interval",
            ),
            Patch(
                facecolor=DYNAMIC_COLOR,
                alpha=DYNAMIC_ALPHA,
                label="Dynamic session",
            ),
            Line2D(
                [0], [0],
                marker="x",
                linestyle="none",
                color="black",
                markeredgewidth=0.6,
                markersize=5,
                label="Flagged fit",
            ),
        ],
    )

    save_figure(
        fig,
        OUTPUT / "slow_component_weights.pdf",
    )
    plt.close(fig)


def plot_timescales(conditions, parameters):
    fig, axes = make_panels(
        "Timescale estimates across models",
        "Timescale (responses)",
        (0, 5000),
    )

    for ax, bird in zip(axes.flat, BIRDS):
        c = conditions.loc[
            conditions["bird"] == bird
        ]

        shade_dynamic(ax, c)

        bird_parameters = parameters.loc[
            parameters["bird"] == bird
        ]

        # Draw credible intervals underneath trajectories.
        for model in MODELS:
            d = bird_parameters.loc[
                bird_parameters["model"] == model
            ].sort_values("session")

            ax.fill_between(
                d["session"].to_numpy(),
                d["q025"].to_numpy(),
                d["q975"].to_numpy(),
                color=MODEL_COLORS[model],
                alpha=INTERVAL_ALPHA,
                linewidth=0,
                zorder=1,
            )

        # Draw posterior medians on top.
        for model in MODELS:
            d = bird_parameters.loc[
                bird_parameters["model"] == model
            ].sort_values("session")

            ax.plot(
                d["session"].to_numpy(),
                d["median"].to_numpy(),
                color=MODEL_COLORS[model],
                linewidth=1.15,
                alpha=0.95,
                zorder=2,
            )

        ax.set_yticks(
            np.arange(0, 5001, 1000)
        )

    labels = {
        "M0": "M0",
        "two_timescales": "Two (slow)",
        "M1": "M1",
        "M2": "M2",
    }

    handles = [
        Line2D(
            [0], [0],
            color=MODEL_COLORS[model],
            lw=2,
            label=labels[model],
        )
        for model in MODELS
    ]

    handles.append(
        Patch(
            facecolor=DYNAMIC_COLOR,
            alpha=DYNAMIC_ALPHA,
            label="Dynamic session",
        )
    )

    add_legend(fig, handles)

    save_figure(
        fig,
        OUTPUT / "timescales_across_models.pdf",
    )
    plt.close(fig)


def main():
    apply_style()
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    conditions = load_conditions()

    # Figure 4
    plot_weights(conditions)

    # Figure 5
    parameters = load_parameters()
    plot_timescales(conditions, parameters)

    print("\nSaved:")
    print(OUTPUT / "slow_component_weights.pdf")
    print(OUTPUT / "timescales_across_models.pdf")


if __name__ == "__main__":
    main()
