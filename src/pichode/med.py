from pathlib import Path
import numpy as np
import pandas as pd

def read_med_array(file_name, array):
    """Read session metadata and an array from a MED-PC data file."""

    metadata = {}
    values = []
    reading_array = False

    metadata_fields = {
        "Subject": "subject",
        "Experiment": "experiment_name",
        "Group": "condition",
        "Box": "box",
        "Start Date": "date",
        "Start Time": "session_start",
        "End Time": "session_end",
        "MSN": "experiment_program",
    }

    with open(file_name) as file:
        for line in file:
            line = line.strip()

            if reading_array:
                if not line or not line[0].isdigit():
                    break

                values.extend(line.split()[1:])
                continue

            if line == f"{array}:":
                reading_array = True
                continue

            for field, name in metadata_fields.items():
                if line.startswith(f"{field}:"):
                    metadata[name] = line.split(":", 1)[1].strip()
                    break

    if not reading_array:
        raise ValueError(f"Array {array} not found in {file_name}")

    return {"values": values, **metadata}

def extract_med_events(file_name):
    """Extract timestamps and event codes from MED-PC array A."""
    data = read_med_array(file_name, "A")

    records = []

    for value in data["values"]:
        timestamp, event_code = value.split(".")

        if event_code == "000":
            continue

        records.append({
            "time_sec_100": int(timestamp),
            "event_key": event_code,
        })

    return records

def parse_vi_schedules(msn):
    """Extract left and right VI schedules from an MSN string."""
    parts = msn.split("_")

    left = [float(value) for value in parts[2:5]]
    right = [float(value) for value in parts[6:9]]

    return left, right

def classify_session(msn):
    """Classify a session as stable or dynamic."""
    left, right = parse_vi_schedules(msn)

    if len(set(left)) == 1 and len(set(right)) == 1:
        return "stable"

    return "dynamic"

def concurrent_extractor(file_name):
    """Convert a concurrent MED-PC session into a pandas DataFrame."""

    data = read_med_array(file_name, "A")
    events = extract_med_events(file_name)

    df = pd.DataFrame(events)

    # Convert hundredths of a second to seconds.
    df["session_time_sec"] = df["time_sec_100"] / 100

    # Add session metadata.
    df["bird"] = data["subject"]
    df["date"] = data["date"]
    df["box"] = data["box"]
    df["session_start"] = data["session_start"]
    df["session_end"] = data["session_end"]
    df["med_notation_file"] = data["experiment_program"]
    df["data_file"] = str(file_name)

    # Event keys
    event_names = {
        "010": "session_start",
        "020": "session_end",
        "110": "response_central_key",
        "120": "response_left_key",
        "130": "response_right_key",
        "220": "left_light_on",
        "230": "right_light_on",
        "320": "left_light_off",
        "330": "right_light_off",
        "240": "feeder_on",
        "340": "feeder_off",
        "250": "chamber_light_on",
        "350": "chamber_light_off",
        "540": "feeder_on_left",
        "640": "feeder_on_right",
        "740": "reinforcer_scheduled_left",
        "840": "reinforcer_scheduled_right",
    }
    df["event"] = df["event_key"].map(event_names).fillna(df["event_key"])
    filename = Path(file_name).name
    df["session"] = int(filename.split("_")[1][1:])
    df["data_file"] = filename  

    left = read_med_array(file_name, "V")["values"]
    right = read_med_array(file_name, "W")["values"]

    df["first_cp_left"] = left[0]
    df["second_cp_left"] = left[1]
    df["first_cp_right"] = right[0]
    df["second_cp_right"] = right[1]

    left_vi, right_vi = parse_vi_schedules(data["experiment_program"])
    df["session_type"] = classify_session(data["experiment_program"])

    time = df["session_time_sec"]
    cp_left_1 = float(left[0])
    cp_left_2 = float(left[1])
    cp_right_1 = float(right[0])
    cp_right_2 = float(right[1])

    df["vi_left"] = np.select(
        [time < cp_left_1, time < cp_left_2],
        left_vi[:2],
        default=left_vi[2],
    )

    df["vi_right"] = np.select(
        [time < cp_right_1, time < cp_right_2],
        right_vi[:2],
        default=right_vi[2],
    )
    return df

def extract_concurrent_directory(input_dir, output_dir):
    """Extract all concurrent MED-PC sessions into individual CSV files."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in sorted(input_dir.iterdir()):
        if file.is_file():
            df = concurrent_extractor(file)
            df.to_csv(output_dir / f"{file.name}.csv", index=False)

def extract_responses(df):
    """Extract individual choices and reinforcement outcomes from event data."""

    # Identify feeder activations by timestamp and key.
    reinforced_left = set(
        df.loc[df["event"] == "feeder_on_left", "time_sec_100"]
    )
    reinforced_right = set(
        df.loc[df["event"] == "feeder_on_right", "time_sec_100"]
    )

    # Keep only left/right responses occurring after session initialization.
    responses = df[
        df["event"].isin(["response_left_key", "response_right_key"])
        & (df["time_sec_100"] > 0)
    ].copy()

    # Identify which key was pressed.
    responses["response"] = responses["event"].map({
        "response_left_key": "left",
        "response_right_key": "right",
    })

    # Determine whether each response produced reinforcement.
    responses["reinforced"] = (
        (
            (responses["response"] == "left")
            & responses["time_sec_100"].isin(reinforced_left)
        )
        | (
            (responses["response"] == "right")
            & responses["time_sec_100"].isin(reinforced_right)
        )
    ).astype(int)

    # Retain only the columns needed for modeling.
    responses = responses.rename(columns={"session_time_sec": "time_sec"})

    return responses[
        [
            "bird",
            "session",
            "time_sec",
            "response",
            "reinforced",
            "session_type",
            "vi_left",
            "vi_right",
        ]
    ]

def extract_response_directory(input_dir, output_dir):
    """Convert event-level CSVs into response-level CSVs."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in sorted(input_dir.glob("*.csv")):
        df = pd.read_csv(file)
        responses = extract_responses(df)
        responses.to_csv(output_dir / file.name, index=False, float_format="%.2f")

def extract_visits(df):
    """Summarize consecutive responses on the same key into visits."""

    df = df.copy()

    # A new visit begins whenever the pigeon switches keys.
    df["visit"] = df["response"].ne(df["response"].shift()).cumsum()

    # Summarize each visit.
    visits = df.groupby("visit", sort=True).agg(
        bird=("bird", "first"),
        session=("session", "first"),
        response=("response", "first"),
        start_time_sec=("time_sec", "first"),
        end_time_sec=("time_sec", "last"),
        n_responses=("response", "size"),
        n_reinforcers=("reinforced", "sum"),
        session_type=("session_type", "first"),
        vi_left=("vi_left", "first"),
        vi_right=("vi_right", "first"),
    ).reset_index()

    # Time between first and last responses.
    visits["duration_sec"] = (
        visits["end_time_sec"] - visits["start_time_sec"]
    )

    # Time since the last response on the opposite key.
    visits["time_since_other_key_sec"] = (
        visits["start_time_sec"] - visits["end_time_sec"].shift(1)
    )

    # Time until the next visit begins.
    visits["time_to_switch_sec"] = (
        visits["start_time_sec"].shift(-1) - visits["start_time_sec"]
    )

    return visits[
        [
            "bird",
            "session",
            "visit",
            "response",
            "start_time_sec",
            "end_time_sec",
            "duration_sec",
            "time_since_other_key_sec",
            "time_to_switch_sec",
            "n_responses",
            "n_reinforcers",
            "session_type",
            "vi_left",
            "vi_right",
        ]
    ]

def extract_visit_directory(input_dir, output_dir):
    """Convert response-level CSVs into visit-level CSVs."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in sorted(input_dir.glob("*.csv")):
        df = pd.read_csv(file)
        visits = extract_visits(df)
        visits.to_csv(output_dir / file.name, index=False, float_format="%.2f")

def extract_reinforcers(df):
    """Summarize behavioral history leading up to each obtained reinforcer."""
    df = df.copy().reset_index(drop=True)

    # Identify visits and positions within visits.
    df["visit"] = df["response"].ne(df["response"].shift()).cumsum()
    df["response_in_visit"] = df.groupby("visit").cumcount() + 1

    # Interresponse intervals, measured retrospectively.
    df["iri_sec"] = df["time_sec"].diff().fillna(df["time_sec"])

    # Response-based allocation: interval belongs to its ending response.
    df["time_response_key"] = df["response"]

    # Visit-based allocation: interval belongs to its starting response.
    # The first interval belongs to the first response.
    df["time_visit_key"] = df["response"].shift().fillna(df["response"])

    # Within-visit allocation: exclude intervals crossing a key switch.
    df["time_within_sec"] = df["iri_sec"].where(
        df["response"].eq(df["response"].shift()), 0.0
    )

    # Identify switches (the first response is not a switch).
    df["switch"] = df["response"].ne(df["response"].shift())
    df.loc[0, "switch"] = False

    records = []
    previous_reinforcer_index = -1
    previous_reinforcer_time = 0.0
    previous_reinforcer_key = None

    last_same_key_index = {"left": -1, "right": -1}
    last_same_key_time = {"left": 0.0, "right": 0.0}

    for i in df.index[df["reinforced"] == 1]:
        row = df.loc[i]
        key = row["response"]

        # Responses since the previous reinforcer, including this one.
        interval = df.iloc[previous_reinforcer_index + 1:i + 1]

        # Responses on this key since its previous reinforcer.
        same_key_interval = df.iloc[last_same_key_index[key] + 1:i + 1]

        # Most recent response on the opposite key.
        other_key = "right" if key == "left" else "left"
        previous_other = df.iloc[:i]
        previous_other = previous_other[
            previous_other["response"] == other_key
        ]

        time_since_other = (
            row["time_sec"] - previous_other["time_sec"].iloc[-1]
            if len(previous_other) else float("nan")
        )

        visit_start = df.loc[
            df["visit"] == row["visit"], "time_sec"
        ].iloc[0]

        records.append({
            "bird": row["bird"],
            "session": row["session"],
            "reinforcer": len(records) + 1,
            "time_sec": row["time_sec"],
            "response": key,
            "session_type": row["session_type"],
            "vi_left": row["vi_left"],
            "vi_right": row["vi_right"],
            "visit": row["visit"],
            "response_in_visit": row["response_in_visit"],
            "first_reinforcer": previous_reinforcer_index == -1,
            "cost_responses_total": len(interval),
            "cost_responses_local": (interval["response"] == key).sum(),
            "cost_responses_other": (interval["response"] == other_key).sum(),
            "cost_responses_since_same_key_reinforcer": (
                (same_key_interval["response"] == key).sum()
            ),
            "cost_time_local_response_sec": interval.loc[
                interval["time_response_key"] == key, "iri_sec"
            ].sum(),
            "cost_time_other_response_sec": interval.loc[
                interval["time_response_key"] == other_key, "iri_sec"
            ].sum(),
            "cost_time_local_visit_sec": interval.loc[
                interval["time_visit_key"] == key, "iri_sec"
            ].sum(),
            "cost_time_other_visit_sec": interval.loc[
                interval["time_visit_key"] == other_key, "iri_sec"
            ].sum(),
            "cost_time_local_within_sec": interval.loc[
                interval["response"] == key, "time_within_sec"
            ].sum(),
            "cost_time_other_within_sec": interval.loc[
                interval["response"] == other_key, "time_within_sec"
            ].sum(),
            "cost_time_total_sec": (
                row["time_sec"] - previous_reinforcer_time
            ),
            "cost_time_since_same_key_reinforcer_sec": (
                row["time_sec"] - last_same_key_time[key]
            ),
            "n_switches": int(interval["switch"].sum()),
            "previous_reinforcer_key": previous_reinforcer_key,
            "time_since_other_key_sec": time_since_other,
            "time_in_current_visit_sec": (
                row["time_sec"] - visit_start
            ),
        })

        previous_reinforcer_index = i
        previous_reinforcer_time = row["time_sec"]
        previous_reinforcer_key = key

        last_same_key_index[key] = i
        last_same_key_time[key] = row["time_sec"]

    result = pd.DataFrame(records)

    if result.empty:
        return pd.DataFrame(columns=[
            "bird",
            "session",
            "reinforcer",
            "time_sec",
            "response",
            "session_type",
            "vi_left",
            "vi_right",
            "visit",
            "response_in_visit",
            "first_reinforcer",
            "cost_responses_total",
            "cost_responses_local",
            "cost_responses_other",
            "cost_responses_since_same_key_reinforcer",
            "cost_time_total_sec",
            "cost_time_local_response_sec",
            "cost_time_other_response_sec",
            "cost_time_local_visit_sec",
            "cost_time_other_visit_sec",
            "cost_time_local_within_sec",
            "cost_time_other_within_sec",
            "cost_time_since_same_key_reinforcer_sec",
            "n_switches",
            "previous_reinforcer_key",
            "time_since_other_key_sec",
            "time_in_current_visit_sec",
        ])

    return result

def extract_reinforcer_directory(
    input_dir="data/processed/concurrent/responses",
    output_dir="data/processed/concurrent/reinforcers",
):
    """Extract reinforcer-level data from all response CSV files."""
    from pathlib import Path

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in sorted(input_dir.glob("*.csv")):
        df = pd.read_csv(file)
        reinforcers = extract_reinforcers(df)

        reinforcers.to_csv(
            output_dir / file.name,
            index=False,
            float_format="%.2f",
        )
