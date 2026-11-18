"""Poster figure: timescale agreement across M0, M1, M2, and two-timescale slow component."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from plot_colors import BIRD_COLORS
from plot_style import apply_style, style_axes, save_figure


ROOT = Path("results/timescales")
OUTPUT = ROOT / "poster/figures/timescale_agreement_four_models_medians.pdf"

BIRDS = ("p004", "p054", "p138", "p510", "p530", "p736")
MODELS = ("M0", "M1", "M2", "fast")

LIMIT = 5000
GRID = np.linspace(0, LIMIT, 400)


def load_data():
    # M0: existing posterior summaries and diagnostic flags.
    m0 = pd.read_csv(
        ROOT / "single_timescale/summary.csv",
        usecols=["bird", "session", "tau_median", "preliminary_ok"],
    ).rename(columns={
        "tau_median": "M0",
        "preliminary_ok": "M0_ok",
    })

    data = m0.copy()

    # Slow component of the two-timescale model.
    slow = pd.read_csv(
        ROOT / "preliminary/summary.csv",
        usecols=[
            "bird", "session",
            "tau_fast_median", "preliminary_ok",
        ],
    ).rename(columns={
        "tau_fast_median": "fast",
        "preliminary_ok": "fast_ok",
    })

    data = data.merge(
        slow,
        on=["bird", "session"],
        validate="one_to_one",
    )

    # M1/M2 diagnostic flags.
    diagnostics = pd.read_csv(
        ROOT / "initial_conditions/parallel_summary.csv",
        usecols=["bird", "session", "model", "preliminary_ok"],
    )

    for model in ("M1", "M2"):
        rows = []

        for bird in BIRDS:
            for session in range(1, 181):
                path = (
                    ROOT / "initial_conditions" / model
                    / f"{bird}_s{session:03d}"
                    / "posterior.csv"
                )

                posterior = pd.read_csv(path, usecols=["tau"])

                rows.append({
                    "bird": bird,
                    "session": session,
                    model: posterior["tau"].median(),
                })

        means = pd.DataFrame(rows)

        flags = diagnostics.loc[
            diagnostics["model"] == model,
            ["bird", "session", "preliminary_ok"],
        ].rename(columns={
            "preliminary_ok": f"{model}_ok",
        })

        data = data.merge(
            means,
            on=["bird", "session"],
            validate="one_to_one",
        ).merge(
            flags,
            on=["bird", "session"],
            validate="one_to_one",
        )

    assert len(data) == 1080
    assert not data.duplicated(["bird", "session"]).any()

    # Handle boolean or string diagnostic values.
    for model in MODELS:
        data[f"{model}_ok"] = (
            data[f"{model}_ok"]
            .astype(str)
            .str.lower()
            .eq("true")
        )

    return data



def plot_matrix(data):
    apply_style()

    # Unified layout: dedicated header + sixteen equal-sized panels.
    fig = plt.figure(figsize=(10.5, 11.5))

    gs = fig.add_gridspec(
        5, 4,
        height_ratios=[0.30, 1, 1, 1, 1],
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.075,
        top=0.985,
        wspace=0.0,
        hspace=0.19,
    )

    # Title and legend occupy the same dedicated header.
    header = fig.add_subplot(gs[0, :])
    header.set_axis_off()

    header.text(
        0.5, 0.82,
        "Agreement in posterior median timescales",
        ha="center",
        va="center",
        fontsize=17,
        transform=header.transAxes,
    )

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            color=BIRD_COLORS[bird],
            markersize=6,
            label=bird,
        )
        for bird in BIRDS
    ]

    handles.append(
        Line2D(
            [0], [0],
            marker="x",
            linestyle="none",
            color="black",
            markeredgewidth=0.6,
            markersize=5,
            label="Either model flagged",
        )
    )

    header.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        bbox_transform=header.transAxes,
        ncol=7,
        frameon=False,
        fontsize=9,
        handlelength=0.8,
        handletextpad=0.25,
        columnspacing=1.4,
    )

    axes = np.empty((4, 4), dtype=object)

    for i, ymodel in enumerate(MODELS):
        for j, xmodel in enumerate(MODELS):

            ax = fig.add_subplot(gs[i + 1, j])
            axes[i, j] = ax

            style_axes(ax, vertical_y_ticks=True)

            # CRITICAL: every panel has the same physical dimensions.
            # This includes density panels.
            ax.set_box_aspect(1)

            ax.set_xlim(0, LIMIT)

            if i == j:
                # Smooth pigeon-specific densities.
                for bird in BIRDS:
                    values = data.loc[
                        data["bird"] == bird, xmodel
                    ].dropna().to_numpy()

                    if len(values) < 2 or np.std(values) == 0:
                        continue

                    kde = gaussian_kde(values)

                    ax.plot(
                        GRID,
                        kde(GRID),
                        color=BIRD_COLORS[bird],
                        linewidth=2.5,
                        alpha=0.9,
                    )

                # Complete box, without density-axis markings.
                ax.tick_params(
                    axis="y",
                    left=False,
                    right=False,
                    labelleft=False,
                    labelright=False,
                )

                ax.set_yticks([])

            else:
                for bird in BIRDS:
                    d = data.loc[data["bird"] == bird]

                    ax.scatter(
                        d[xmodel],
                        d[ymodel],
                        s=13,
                        alpha=0.50,
                        color=BIRD_COLORS[bird],
                        edgecolors="none",
                        rasterized=True,
                    )

                flagged = data.loc[
                    ~data[f"{xmodel}_ok"]
                    | ~data[f"{ymodel}_ok"]
                ]

                ax.scatter(
                    flagged[xmodel],
                    flagged[ymodel],
                    marker="x",
                    s=9,
                    linewidths=0.45,
                    color="black",
                    alpha=0.8,
                    zorder=5,
                    rasterized=True,
                )

                ax.plot(
                    [0, LIMIT],
                    [0, LIMIT],
                    color="black",
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.6,
                    zorder=0,
                )

                ax.set_ylim(0, LIMIT)

            ax.grid(alpha=0.12, linewidth=0.5)

            # Column headings.
            if i == 0:
                ax.set_title(
                    "Two-timescales" if xmodel == "fast" else xmodel,
                    fontsize=12,
                    fontweight="bold",
                    pad=5,
                )

            # Row labels.
            if j == 0 and i != j:
                ax.set_ylabel(r"$\tau_{\mathrm{fast}}$" if ymodel == "fast" else r"$\tau_{" + ymodel + r"}$", fontsize=13)

            # Column labels.
            if i == 3:
                ax.set_xlabel(r"$\tau_{\mathrm{fast}}$" if xmodel == "fast" else r"$\tau_{" + xmodel + r"}$", fontsize=13)

            # Keep the interior panels clean.
            if i < 3:
                ax.tick_params(
                    axis="x",
                    labelbottom=False,
                )

            if j > 0 and i != j:
                ax.tick_params(
                    axis="y",
                    labelleft=False,
                )

    save_figure(fig, OUTPUT)

    print(f"Saved: {OUTPUT}")

    for model in MODELS:
        n = (~data[f"{model}_ok"]).sum()
        print(f"{model}: {n} flagged sessions")


if __name__ == "__main__":
    plot_matrix(load_data())
