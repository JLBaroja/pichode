import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --------------------------------------------------
# Paths
# --------------------------------------------------

single_root = Path("results/timescales/single_timescale")
two_root = Path("results/timescales/preliminary")
events_dir = Path("data/processed/concurrent/events")

output = Path(
    "results/timescales/comparison/figures/"
    "comparative_parameter_trajectories.pdf"
)

# --------------------------------------------------
# Load posterior summaries
# --------------------------------------------------

single = pd.read_csv(single_root / "summary.csv")
two = pd.read_csv(two_root / "summary.csv")

for name, df in [("single", single), ("two", two)]:
    assert len(df) == 1080, f"{name}: expected 1080 sessions"
    assert not df.duplicated(["bird", "session"]).any()

# --------------------------------------------------
# Load canonical session classifications
# --------------------------------------------------

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

assert not conditions.empty
assert not conditions.duplicated(["bird", "session"]).any()
assert conditions["session_type"].isin(["stable", "dynamic"]).all()

# --------------------------------------------------
# Combine model summaries
# --------------------------------------------------

single = single.rename(
    columns={
        "preliminary_ok": "single_ok",
    }
)

two = two.rename(
    columns={
        "preliminary_ok": "two_ok",
    }
)

single_columns = [
    "bird", "session",
    "tau_median", "tau_q025", "tau_q975",
    "single_ok",
]

two_columns = [
    "bird", "session",
    "tau_fast_median", "tau_fast_q025", "tau_fast_q975",
    "tau_slow_median", "tau_slow_q025", "tau_slow_q975",
    "w_median", "w_q025", "w_q975",
    "two_ok",
]

df = single[single_columns].merge(
    two[two_columns],
    on=["bird", "session"],
    how="inner",
    validate="one_to_one",
)

df = df.merge(
    conditions,
    on=["bird", "session"],
    how="left",
    validate="one_to_one",
)

assert len(df) == 1080
assert df["session_type"].notna().all()

# --------------------------------------------------
# Plot settings
# --------------------------------------------------

parameters = [
    ("tau", "Single timescale (responses)", "darkviolet", "single_ok"),
    ("tau_fast", "Fast timescale (responses)", "steelblue", "two_ok"),
    ("tau_slow", "Slow timescale (responses)", "steelblue", "two_ok"),
    ("w", "Slow-component weight", "steelblue", "two_ok"),
]

output.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Generate six-page PDF
# --------------------------------------------------

with PdfPages(output) as pdf:

    for bird in sorted(df["bird"].unique()):

        d = df[df["bird"] == bird].sort_values("session")
        x = d["session"].to_numpy()

        fig, axes = plt.subplots(
            4, 1,
            figsize=(13, 11),
            sharex=True,
            constrained_layout=True,
        )

        for ax, (parameter, ylabel, color, ok_column) in zip(
            axes, parameters
        ):

            # Dynamic-session shading
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

            # 95% posterior credible interval
            ax.fill_between(
                x,
                d[f"{parameter}_q025"].to_numpy(),
                d[f"{parameter}_q975"].to_numpy(),
                color=color,
                alpha=0.25,
            )

            # Posterior median
            ax.plot(
                x,
                d[f"{parameter}_median"].to_numpy(),
                color=color,
                linewidth=1.2,
            )

            # Flagged fits
            flagged = d[
                d[ok_column].astype(str).str.lower() != "true"
            ]

            ax.scatter(
                flagged["session"],
                flagged[f"{parameter}_median"],
                marker="x",
                color="red",
                s=35,
                zorder=5,
            )

            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)

            if parameter == "w":
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(0, 5000)

        axes[-1].set_xlabel("Session")
        axes[-1].set_xlim(0.5, 180.5)

        legend = [
            Line2D(
                [0], [0],
                color="black",
                label="Posterior median",
            ),
            Patch(
                facecolor="gray",
                alpha=0.25,
                label="95% credible interval",
            ),
            Patch(
                facecolor="orange",
                alpha=0.25,
                label="Dynamic session",
            ),
            Patch(
                facecolor="white",
                edgecolor="gray",
                label="Stable session",
            ),
            Line2D(
                [0], [0],
                marker="x",
                linestyle="none",
                color="red",
                label="Flagged fit",
            ),
        ]

        axes[0].legend(
            handles=legend,
            loc="upper right",
            fontsize=9,
            ncol=2,
        )

        n_stable = (d["session_type"] == "stable").sum()
        n_dynamic = (d["session_type"] == "dynamic").sum()

        n_single_flagged = (
            d["single_ok"].astype(str).str.lower() != "true"
        ).sum()

        n_two_flagged = (
            d["two_ok"].astype(str).str.lower() != "true"
        ).sum()

        fig.suptitle(
            f"{bird} — Single vs. two-timescale models\n"
            f"{n_stable} stable | {n_dynamic} dynamic sessions | "
            f"Flagged: single={n_single_flagged}, "
            f"two={n_two_flagged}",
            fontsize=14,
        )

        pdf.savefig(fig)
        plt.close(fig)

        print(
            f"{bird}: {n_stable} stable, {n_dynamic} dynamic; "
            f"flagged single={n_single_flagged}, "
            f"two={n_two_flagged}"
        )

print(f"\nSaved: {output}")
