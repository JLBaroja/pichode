"""Plot visit behavior over session time for all pigeons."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


# --------------------------------------------------
# SETTINGS — edit these whenever you want
# --------------------------------------------------

VISIT_DIR = Path("data/processed/concurrent/visits")
RESPONSE_DIR = Path("data/processed/concurrent/responses")
OUTPUT_DIR = Path("results/figures")

BIRDS = ["p004", "p054", "p138", "p510", "p530", "p736"]

MAX_VISIT_LENGTH = 15
MAX_VISIT_DURATION = 15
MAX_TIME_SINCE_OTHER_KEY = 15

POINT_SIZE = 12
POINT_ALPHA = 0.5

COLORS = {
    "left": "tab:blue",
    "right": "tab:orange",
}

PANELS = [
    ("n_responses", "Visit length (responses)"),
    ("duration_sec", "Visit duration (seconds)"),
    ("time_since_other_key_sec", "Time since other key (seconds)"),
]


# --------------------------------------------------
# FIND CONTINGENCY CHANGES
# --------------------------------------------------

def get_conditions(response_file):
    """Read VI sequence and change times from response data."""

    df = pd.read_csv(
        response_file,
        usecols=["time_sec", "vi_left", "vi_right"],
    )

    df = df.sort_values("time_sec")

    vi = df[["vi_left", "vi_right"]]
    changed = vi.ne(vi.shift()).any(axis=1)

    # First condition is not a changepoint.
    change_times = df.loc[
        changed & (df.index != df.index[0]),
        "time_sec",
    ].to_numpy()

    conditions = " → ".join(
        f"{row.vi_left:g}/{row.vi_right:g}"
        for row in vi.loc[changed].itertuples()
    )

    return conditions, change_times


# --------------------------------------------------
# PLOT ONE SESSION
# --------------------------------------------------

def plot_session(name, visits, conditions, change_times):
    """Create one page containing three visit time-series panels."""

    visits = visits.sort_values("start_time_sec")

    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 10),
        sharex=True,
    )

    limits = [
        MAX_VISIT_LENGTH,
        MAX_VISIT_DURATION,
        MAX_TIME_SINCE_OTHER_KEY,
    ]

    for ax, (column, ylabel), maximum in zip(
        axes, PANELS, limits
    ):
        for key in ("left", "right"):
            subset = visits[visits["response"] == key]

            ax.scatter(
                subset["start_time_sec"],
                subset[column],
                color=COLORS[key],
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                rasterized=True,
            )

        for change_time in change_times:
            ax.axvline(
                change_time,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.65,
            )

        ax.set_ylabel(ylabel)
        ax.set_ylim(0, maximum)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Session time (seconds)")
    axes[-1].set_xlim(
        0,
        max(visits["end_time_sec"].max(), 1),
    )

    session_type = visits["session_type"].iloc[0]

    fig.suptitle(
        f"{name.upper()} — {session_type} — VI {conditions}",
        fontsize=15,
        y=0.98,
    )

    legend = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            color=color,
            markersize=6,
            label=key.capitalize(),
        )
        for key, color in COLORS.items()
    ]

    if len(change_times):
        legend.append(
            Line2D(
                [0], [0],
                color="black",
                linestyle="--",
                label="Contingency change",
            )
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
# GENERATE ALL PDFs
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for bird in BIRDS:
        files = sorted(
            VISIT_DIR.glob(f"{bird}_*.csv"),
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

        output = OUTPUT_DIR / f"{bird}_visit_time_series.pdf"

        print(f"\nGenerating {output}...")

        with PdfPages(output) as pdf:
            for i, visit_file in enumerate(files, start=1):
                visits = pd.read_csv(visit_file)

                response_file = (
                    RESPONSE_DIR / visit_file.name
                )

                conditions, change_times = get_conditions(
                    response_file
                )

                fig = plot_session(
                    visit_file.stem,
                    visits,
                    conditions,
                    change_times,
                )

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                print(
                    f"\r{bird.upper()}: {i}/180 sessions",
                    end="",
                    flush=True,
                )

        print(f"\nSaved: {output}")

    print("\nAll six visit time-series PDFs generated successfully!")


if __name__ == "__main__":
    main()
