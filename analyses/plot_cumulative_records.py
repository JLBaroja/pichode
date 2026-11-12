"""Cumulative response and reinforcer records for all pigeon sessions."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# --------------------------------------------------
# Settings
# --------------------------------------------------

BASE = Path("data/processed/concurrent/responses")
OUTPUT_DIR = Path("results/figures/cumulative_records")

BIRDS = ("p004", "p054", "p138", "p510", "p530", "p736")

FIGSIZE = (14, 7)
LINEWIDTH = 1.2
PREDICTION_LINEWIDTH = 2

OBSERVED_COLOR = "black"
PREDICTION_COLOR = "tab:green"
REFERENCE_COLOR = "gray"
SWITCH_COLOR = "red"

# None: automatic limits based on the largest session
# for each bird. Otherwise, specify an upper limit.
RESPONSE_LIMIT = None
REINFORCER_LIMIT = None

AXIS_PADDING = 1.05

# Temporal markers
TIME_MARKER_INTERVAL_MIN = 10
TIME_MARKER_COLOR = "tab:blue"
TIME_MARKER_SIZE = 24
TIME_LABEL_FONTSIZE = 7


# --------------------------------------------------
# Session files
# --------------------------------------------------

def session_files(bird):

    return [
        path
        for path in sorted(BASE.glob(f"{bird}_*.csv"))
        if not path.stem.endswith(("_bis", "_b"))
    ]


# --------------------------------------------------
# Prepare cumulative counts and segments
# --------------------------------------------------

def prepare(df):

    df = df.sort_values("time_sec").reset_index(drop=True).copy()

    left = df["response"].eq("left")
    right = df["response"].eq("right")
    reinforced = df["reinforced"].eq(1)

    df["N_left"] = left.cumsum()
    df["N_right"] = right.cumsum()

    df["R_left"] = (left & reinforced).cumsum()
    df["R_right"] = (right & reinforced).cumsum()

    change = (
        df["vi_left"].ne(df["vi_left"].shift())
        | df["vi_right"].ne(df["vi_right"].shift())
    )

    df["segment"] = change.cumsum() - 1

    return df


# --------------------------------------------------
# Segment information
# --------------------------------------------------

def segment_information(df):

    segments = []

    columns = ("N_left", "N_right", "R_left", "R_right")

    for segment_id, group in df.groupby("segment", sort=True):

        first = group.index[0]
        last = group.index[-1]

        if first == 0:
            start = {column: 0 for column in columns}
        else:
            start = df.loc[first - 1, list(columns)].to_dict()

        end = df.loc[last, list(columns)].to_dict()

        segments.append({
            "id": segment_id,
            "start": start,
            "end": end,
            "r_left": end["R_left"] - start["R_left"],
            "r_right": end["R_right"] - start["R_right"],
            "vi_left": group["vi_left"].iloc[0],
            "vi_right": group["vi_right"].iloc[0],
        })

    return segments


# --------------------------------------------------
# Programmed contingency labels
# --------------------------------------------------

def contingency_label(segments):

    return " → ".join(
        f"VI {segment['vi_left']:g}/{segment['vi_right']:g}"
        for segment in segments
    )


# --------------------------------------------------
# Determine shared limits for a pigeon
# --------------------------------------------------

def bird_limits(paths):

    max_responses = 0
    max_reinforcers = 0

    for path in paths:

        df = pd.read_csv(
            path,
            usecols=["response", "reinforced"],
        )

        if df.empty:
            continue

        left = df["response"].eq("left")
        right = df["response"].eq("right")
        reinforced = df["reinforced"].eq(1)

        max_responses = max(
            max_responses,
            int(left.sum()),
            int(right.sum()),
        )

        max_reinforcers = max(
            max_reinforcers,
            int((left & reinforced).sum()),
            int((right & reinforced).sum()),
        )

    response_limit = (
        RESPONSE_LIMIT
        if RESPONSE_LIMIT is not None
        else max(1, max_responses * AXIS_PADDING)
    )

    reinforcer_limit = (
        REINFORCER_LIMIT
        if REINFORCER_LIMIT is not None
        else max(1, max_reinforcers * AXIS_PADDING)
    )

    return response_limit, reinforcer_limit


# --------------------------------------------------
# Plot one cumulative record
# --------------------------------------------------

def plot_record(
    ax,
    df,
    segments,
    xcol,
    ycol,
    limit,
    is_response,
):

    x = np.r_[0, df[xcol].to_numpy()]
    y = np.r_[0, df[ycol].to_numpy()]

    ax.plot(
        x,
        y,
        color=OBSERVED_COLOR,
        linewidth=LINEWIDTH,
        label="Observed",
        zorder=2,
    )

    ax.plot(
        [0, limit],
        [0, limit],
        linestyle="--",
        color=REFERENCE_COLOR,
        linewidth=1,
        label="Equal allocation",
    )

    # Programmed and obtained reinforcement benchmarks
    for i, segment in enumerate(segments):

        start = segment["start"]
        end = segment["end"]

        x0 = start[xcol]
        y0 = start[ycol]

        if is_response:
            total = (
                end["N_left"] - start["N_left"]
                + end["N_right"] - start["N_right"]
            )
        else:
            total = (
                end["R_left"] - start["R_left"]
                + end["R_right"] - start["R_right"]
            )

        # Programmed reinforcement-rate proportions
        rate_left = 1.0 / segment["vi_left"]
        rate_right = 1.0 / segment["vi_right"]
        rate_total = rate_left + rate_right

        # No programmed allocation exists when both rates are zero.
        if rate_total > 0:
            programmed_left = rate_left / rate_total
            programmed_right = rate_right / rate_total
        else:
            programmed_left = None
            programmed_right = None

        # Obtained reinforcement proportions
        r_left = segment["r_left"]
        r_right = segment["r_right"]
        r_total = r_left + r_right

        benchmarks = []

        if programmed_left is not None:
            benchmarks.append(
                (
                    "Programmed",
                    programmed_left,
                    programmed_right,
                    "tab:green",
                )
            )

        if r_total > 0:
            benchmarks.append(
                (
                    "Obtained",
                    r_left / r_total,
                    r_right / r_total,
                    "tab:purple",
                )
            )

        for name, p_left, p_right, color in benchmarks:

            predicted_dx = total * p_left
            predicted_dy = total * p_right

            if is_response:
                label = f"{name} matching prediction"
            else:
                label = f"{name} reinforcement ratio"

            ax.plot(
                [x0, x0 + predicted_dx],
                [y0, y0 + predicted_dy],
                linestyle="--",
                color=color,
                linewidth=PREDICTION_LINEWIDTH,
                label=label if i == 0 else None,
                zorder=3,
            )

    # Temporal markers at fixed intervals since session start.
    interval_sec = TIME_MARKER_INTERVAL_MIN * 60
    times = np.arange(
        interval_sec,
        df["time_sec"].iloc[-1] + 1e-9,
        interval_sec,
    )

    response_times = df["time_sec"].to_numpy()
    indices = np.searchsorted(response_times, times, side="right") - 1

    for minute_time, idx in zip(times, indices):
        if idx >= 0:
            marker_x = df[xcol].iloc[idx]
            marker_y = df[ycol].iloc[idx]
        else:
            marker_x = 0
            marker_y = 0

        minutes = int(round(minute_time / 60))

        ax.scatter(
            marker_x,
            marker_y,
            color=TIME_MARKER_COLOR,
            s=TIME_MARKER_SIZE,
            zorder=6,
        )

        ax.annotate(
            f"{minutes}",
            (marker_x, marker_y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=TIME_LABEL_FONTSIZE,
            color=TIME_MARKER_COLOR,
            zorder=7,
        )

    # Contingency-change markers
    for i, segment in enumerate(segments[1:]):

        switch = segment["start"]

        ax.scatter(
            switch[xcol],
            switch[ycol],
            color=SWITCH_COLOR,
            s=65,
            zorder=5,
            label="Contingency change" if i == 0 else None,
        )

    ax.set(
        xlim=(0, limit),
        ylim=(0, limit),
    )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8)


# --------------------------------------------------
# Plot one session
# --------------------------------------------------

def plot_session(path, response_limit, reinforcer_limit):

    df = pd.read_csv(path)

    if df.empty:
        return None

    df = prepare(df)
    segments = segment_information(df)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    fig.suptitle(
        f"{path.stem} — {contingency_label(segments)}",
        fontsize=14,
    )

    plot_record(
        axes[0],
        df,
        segments,
        "N_left",
        "N_right",
        response_limit,
        is_response=True,
    )

    axes[0].set(
        title="Cumulative responses",
        xlabel="Cumulative left responses",
        ylabel="Cumulative right responses",
    )

    plot_record(
        axes[1],
        df,
        segments,
        "R_left",
        "R_right",
        reinforcer_limit,
        is_response=False,
    )

    axes[1].set(
        title="Cumulative reinforcers",
        xlabel="Cumulative left reinforcers",
        ylabel="Cumulative right reinforcers",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    return fig


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for bird in BIRDS:

        paths = session_files(bird)

        if not paths:
            print(f"{bird}: No sessions found.")
            continue

        response_limit, reinforcer_limit = bird_limits(paths)

        output = OUTPUT_DIR / f"{bird}_cumulative_records.pdf"

        pages = 0

        with PdfPages(output) as pdf:

            for path in paths:

                fig = plot_session(
                    path,
                    response_limit,
                    reinforcer_limit,
                )

                if fig is None:
                    continue

                pdf.savefig(fig)
                plt.close(fig)

                pages += 1

        print(
            f"{bird}: {pages} pages -> {output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
