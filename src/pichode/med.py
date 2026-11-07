from pathlib import Path
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
    
    return df

