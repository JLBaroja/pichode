import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Paths
root = Path("results/timescales")
events_dir = Path("data/processed/concurrent/events")
outdir = root / "comparison/figures"
outdir.mkdir(parents=True, exist_ok=True)

# Load model summaries
single = pd.read_csv(root / "single_timescale/summary.csv")
two = pd.read_csv(root / "preliminary/summary.csv")
initial = pd.read_csv(
    root / "initial_conditions/parallel_summary.csv"
)

m1 = initial[initial["model"] == "M1"].copy()
m2 = initial[initial["model"] == "M2"].copy()

models = {
    "M0": single,
    "M1": m1,
    "M2": m2,
}

colors = {
    "M0": "darkviolet",
    "M1": "forestgreen",
    "M2": "firebrick",
}

# Validate coverage
for name, data in {**models, "Two": two}.items():
    assert len(data) == 1080, name
    assert not data.duplicated(["bird", "session"]).any(), name
    assert data.groupby("bird")["session"].nunique().eq(180).all(), name

# Load session classifications
rows = []

for path in sorted(events_dir.glob("*_japede.csv")):
    d = pd.read_csv(
        path,
        usecols=["bird", "session", "session_type"],
        nrows=1,
    )
    if not d.empty:
        rows.append(d.iloc[0].to_dict())

conditions = pd.DataFrame(rows)

assert not conditions.duplicated(["bird", "session"]).any()
assert conditions["session_type"].isin(["stable", "dynamic"]).all()

# Merge all model summaries
df = conditions.copy()

for name, data in models.items():
    columns = [
        "bird", "session",
        "tau_median", "tau_q025", "tau_q975",
        "preliminary_ok",
    ]

    renamed = data[columns].rename(
        columns={
            col: f"{name}_{col}"
            for col in columns
            if col not in ("bird", "session")
        }
    )

    df = df.merge(
        renamed,
        on=["bird", "session"],
        validate="one_to_one",
    )

columns = [
    "bird", "session",
    "tau_fast_median", "tau_fast_q025", "tau_fast_q975",
    "tau_slow_median", "tau_slow_q025", "tau_slow_q975",
    "w_median", "w_q025", "w_q975",
    "preliminary_ok",
]

renamed = two[columns].rename(
    columns={
        col: f"Two_{col}"
        for col in columns
        if col not in ("bird", "session")
    }
)

df = df.merge(
    renamed,
    on=["bird", "session"],
    validate="one_to_one",
)

assert len(df) == 1080
assert df.notna().all().all()

# Plot helpers
def shade_dynamic(ax, d):
    for session in d.loc[
        d["session_type"] == "dynamic", "session"
    ]:
        ax.axvspan(
            session - 0.5,
            session + 0.5,
            color="orange",
            alpha=0.18,
            linewidth=0,
        )


def draw_trajectory(ax, d, prefix, color):
    x = d["session"].to_numpy()

    ax.fill_between(
        x,
        d[f"{prefix}_q025"].to_numpy(),
        d[f"{prefix}_q975"].to_numpy(),
        color=color,
        alpha=0.20,
    )

    ax.plot(
        x,
        d[f"{prefix}_median"].to_numpy(),
        color=color,
        linewidth=1.2,
    )


def mark_flagged(ax, d, prefix, ok_column):
    flagged = d[
        d[ok_column].astype(str).str.lower() != "true"
    ]

    ax.scatter(
        flagged["session"],
        flagged[f"{prefix}_median"],
        marker="x",
        color="red",
        s=35,
        zorder=5,
    )


# Figure 1: Complete parameter trajectories
parameters = [
    ("M0_tau", "M0: Single timescale", "darkviolet", "M0"),
    ("M1_tau", "M1: Initial strength", "forestgreen", "M1"),
    ("M2_tau", "M2: Initial preference", "firebrick", "M2"),
    ("Two_tau_fast", "Two: Fast timescale", "steelblue", "Two"),
    ("Two_tau_slow", "Two: Slow timescale", "steelblue", "Two"),
    ("Two_w", "Two: Slow-component weight", "steelblue", "Two"),
]

output = outdir / "initial_conditions_parameter_trajectories.pdf"

with PdfPages(output) as pdf:
    for bird in sorted(df["bird"].unique()):
        d = df[df["bird"] == bird].sort_values("session")

        fig, axes = plt.subplots(
            6, 1,
            figsize=(13, 15),
            sharex=True,
            constrained_layout=True,
        )

        for ax, (prefix, label, color, model) in zip(
            axes, parameters
        ):
            shade_dynamic(ax, d)
            draw_trajectory(ax, d, prefix, color)
            mark_flagged(
                ax, d, prefix, f"{model}_preliminary_ok"
            )

            ax.set_ylabel(label)
            ax.set_ylim(0, 1 if prefix == "Two_w" else 5000)
            ax.grid(alpha=0.2)

        axes[-1].set_xlabel("Session")
        axes[-1].set_xlim(0.5, 180.5)

        fig.suptitle(
            f"{bird} — Parameter trajectories",
            fontsize=14,
        )

        pdf.savefig(fig)
        plt.close(fig)

print(f"Saved: {output}")

# Figure 2: Direct tau comparison
output = outdir / "initial_conditions_tau_overlay.pdf"

with PdfPages(output) as pdf:
    for bird in sorted(df["bird"].unique()):
        d = df[df["bird"] == bird].sort_values("session")

        fig, ax = plt.subplots(
            figsize=(13, 5),
            constrained_layout=True,
        )

        shade_dynamic(ax, d)

        for name in ("M0", "M1", "M2"):
            draw_trajectory(
                ax, d, f"{name}_tau", colors[name]
            )

            mark_flagged(
                ax, d,
                f"{name}_tau",
                f"{name}_preliminary_ok",
            )

        ax.set_xlabel("Session")
        ax.set_ylabel("Timescale (responses)")
        ax.set_xlim(0.5, 180.5)
        ax.set_ylim(0, 5000)
        ax.grid(alpha=0.2)

        legend = [
            Line2D([0], [0], color=colors[name], label=name)
            for name in ("M0", "M1", "M2")
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

        ax.legend(handles=legend, ncol=5, fontsize=9)

        fig.suptitle(
            f"{bird} — Effect of initial conditions on tau",
            fontsize=14,
        )

        pdf.savefig(fig)
        plt.close(fig)

print(f"Saved: {output}")
