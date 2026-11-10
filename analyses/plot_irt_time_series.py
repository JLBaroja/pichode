"""Plot individual IRTs over time within a session."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_DIR = Path("data/processed/concurrent/responses")
OUTPUT_DIR = Path("results/figures")

BIRD = "p004"
SESSION = None  # None = automatically select the first dynamic session

LINEAR_MAX = 10
LOG_MIN = 0.01
LOG_MAX = 1000

COLORS = {
    "left": "tab:blue",
    "right": "tab:orange",
}

POINT_SIZE = 10
POINT_ALPHA = 0.45


# --------------------------------------------------
# Load one session
# --------------------------------------------------

def load_session():
    files = sorted(INPUT_DIR.glob(f"{BIRD}_*.csv"))

    files = [
        f for f in files
        if "_bis" not in f.stem
        and f.stem != "p530_s145_japede_b"
    ]

    if SESSION is not None:
        files = [
            f for f in files
            if int(f.stem.split("_s")[1][:3]) == SESSION
        ]

    for file in files:
        df = pd.read_csv(file)

        if SESSION is not None or (
            df["session_type"].astype(str).str.lower() == "dynamic"
        ).any():
            return file.stem, df

    raise ValueError("No matching session found.")


# --------------------------------------------------
# Identify contingency changes
# --------------------------------------------------

def find_changes(df):
    ordered = df.sort_values("time_sec")

    vi = ordered[["vi_left", "vi_right"]]

    changed = vi.ne(vi.shift()).any(axis=1)
    changed.iloc[0] = False

    return ordered.loc[changed, "time_sec"].to_numpy()


# --------------------------------------------------
# Plot
# --------------------------------------------------

def plot_session(name, df):
    df = df.sort_values("time_sec")

    changes = find_changes(df)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(15, 10),
        sharex=True,
    )

    for ax, scale in zip(axes, ["linear", "log"]):

        for lever in ["left", "right"]:
            color = COLORS[lever]

            for irt_type in ["within", "switch"]:
                subset = df[
                    (df["response"] == lever)
                    & (df["irt_type"] == irt_type)
                    & (df["irt_sec"] > 0)
                ]

                if subset.empty:
                    continue

                ax.scatter(
                    subset["time_sec"],
                    subset["irt_sec"],
                    s=POINT_SIZE if irt_type == "within" else 25,
                    marker="o" if irt_type == "within" else "x",
                    color=color,
                    linewidths=0.7,
                    alpha=POINT_ALPHA,
                    rasterized=True,
                )

        for change_time in changes:
            ax.axvline(
                change_time,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.65,
            )

        ax.set_yscale(scale)
        ax.set_ylabel("IRT (seconds)")
        ax.grid(alpha=0.15)

        if scale == "linear":
            ax.set_ylim(0, LINEAR_MAX)
            ax.set_title("Linear scale")
        else:
            ax.set_ylim(LOG_MIN, LOG_MAX)
            ax.set_title("Logarithmic scale")

    axes[-1].set_xlabel("Session time (seconds)")
    axes[-1].set_xlim(0, df["time_sec"].max())
    
    legend = [
        Line2D(
            [0], [0],
            marker="o" if kind == "within" else "x",
            linestyle="none",
            color=color,
            markersize=7,
            label=f"{lever.capitalize()} — {kind}",
        )
        for lever, color in COLORS.items()
        for kind in ["within", "switch"]
    ]

    if len(changes):
        legend.append(
            Line2D(
                [0], [0],
                color="black",
                linestyle="--",
                label="Contingency change",
            )
        )

    first = df.iloc[0]
    session_type = first["session_type"]

    vi = df[["vi_left", "vi_right"]]
    changed = vi.ne(vi.shift()).any(axis=1)
    
    conditions = " → ".join(
        f"{row.vi_left:g}/{row.vi_right:g}"
        for row in vi.loc[changed].itertuples()
    )

    fig.suptitle(
        f"{name.upper()} — {session_type} — "
        f"VI {conditions}",
        fontsize=16,
        y=0.98,
    )

    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=len(legend),
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    return fig


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    birds = ["p004", "p054", "p138", "p510", "p530", "p736"]

    for bird in birds:
        files = sorted(
            INPUT_DIR.glob(f"{bird}_*.csv"),
            key=lambda f: int(f.stem.split("_s")[1][:3]),
        )

        files = [
            f for f in files
            if "_bis" not in f.stem
            and f.stem != "p530_s145_japede_b"
        ]

        assert len(files) == 180, (
            f"{bird}: expected 180 sessions, found {len(files)}"
        )

        output = OUTPUT_DIR / f"{bird}_irt_time_series.pdf"

        print(f"\nGenerating {output}...")

        with PdfPages(output) as pdf:
            for i, file in enumerate(files, start=1):
                df = pd.read_csv(file)

                fig = plot_session(file.stem, df)

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                print(
                    f"\r{bird.upper()}: {i}/180 sessions",
                    end="",
                    flush=True,
                )

        print(f"\nSaved: {output}")

    print("\nAll six IRT time-series PDFs generated successfully!")


if __name__ == "__main__":
    main()
