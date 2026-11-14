import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --------------------------------------------------
# Paths
# --------------------------------------------------

root = Path("results/timescales/single_timescale")
events_dir = Path("data/processed/concurrent/events")
output = root / "figures" / "single_timescale_trajectories_by_bird.pdf"

# --------------------------------------------------
# Load posterior summaries
# --------------------------------------------------

df = pd.read_csv(root / "summary.csv")

assert len(df) == 1080, "Expected 1080 fitted sessions"
assert not df.duplicated(["bird", "session"]).any()

# --------------------------------------------------
# Read session classifications
# ONLY canonical *_japede.csv files
# Excludes _bis, _b, and other variants
# --------------------------------------------------

rows = []

for path in sorted(events_dir.glob("*_japede.csv")):

    d = pd.read_csv(
        path,
        usecols=["bird", "session", "session_type"],
        nrows=1,
    )

    if d.empty:
        continue

    rows.append(d.iloc[0].to_dict())

conditions = pd.DataFrame(rows)

assert not conditions.empty, "No canonical event files found"

assert not conditions.duplicated(
    ["bird", "session"]
).any(), "Duplicate canonical sessions"

assert conditions["session_type"].isin(
    ["stable", "dynamic"]
).all(), "Unknown session type"

# --------------------------------------------------
# Merge classifications with fitted sessions
# --------------------------------------------------

df = df.merge(
    conditions,
    on=["bird", "session"],
    how="left",
    validate="one_to_one",
)

assert len(df) == 1080
assert df["session_type"].notna().all(), (
    "Some fitted sessions have no classification"
)

print("\nSession classifications:")
print(df["session_type"].value_counts())

# --------------------------------------------------
# Plot settings
# --------------------------------------------------

parameters = [
    ("tau", "Integration timescale (responses)"),
]

output.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Generate six-page PDF
# --------------------------------------------------

with PdfPages(output) as pdf:

    for bird in sorted(df["bird"].unique()):

        d = df[df["bird"] == bird].sort_values("session")

        fig, axes = plt.subplots(
            1, 1,
            figsize=(13, 5),
            sharex=True,
            constrained_layout=True,
        )

        axes = [axes]

        for ax, (parameter, ylabel) in zip(axes, parameters):

            x = d["session"].to_numpy()

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
                color="steelblue",
                alpha=0.25,
            )

            # Posterior median
            ax.plot(
                x,
                d[f"{parameter}_median"].to_numpy(),
                color="steelblue",
                linewidth=1.2,
            )

            # Flagged fits
            flagged = d[
                d["preliminary_ok"].astype(str).str.lower() != "true"
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

        # Axis limits
        axes[0].set_ylim(0, 5000)

        axes[-1].set_xlabel("Session")
        axes[-1].set_xlim(0.5, 180.5)

        # Legend
        legend = [
            Line2D(
                [0], [0],
                color="steelblue",
                label="Posterior median",
            ),
            Patch(
                facecolor="steelblue",
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

        fig.suptitle(
            f"{bird} — Single reinforcement integration timescale\n"
            f"{n_stable} stable sessions | "
            f"{n_dynamic} dynamic sessions",
            fontsize=15,
        )

        pdf.savefig(fig)
        plt.close(fig)

        print(
            f"{bird}: "
            f"{n_stable} stable, "
            f"{n_dynamic} dynamic"
        )

print(f"\nSaved: {output}")
