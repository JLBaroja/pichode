from pathlib import Path
import json

import numpy as np
import pandas as pd


# Configuration
BIRD = "p530"
SESSION = 127

DATA_DIR = Path("data/processed/concurrent/responses")
OUTPUT_DIR = Path("results/timescales/fits")


def prepare_session(bird, session):
    path = DATA_DIR / f"{bird}_s{session:03d}_japede.csv"

    df = pd.read_csv(path)
    df = df.sort_values("time_sec", kind="stable").reset_index(drop=True)

    required = ["time_sec", "response", "reinforced"]
    if df[required].isna().any().any():
        raise ValueError("Missing values in required columns.")

    if not df["response"].isin(["left", "right"]).all():
        raise ValueError("Unexpected response labels.")

    if not df["reinforced"].isin([0, 1]).all():
        raise ValueError("Unexpected reinforcement values.")

    left = df["response"].eq("left")
    right = df["response"].eq("right")
    reinforced = df["reinforced"].eq(1)

    data = {
        "N": len(df),
        "y": left.astype(int).tolist(),
        "r_left": (left & reinforced).astype(int).tolist(),
        "r_right": (right & reinforced).astype(int).tolist(),
    }

    return df, data


def main():
    df, data = prepare_session(BIRD, SESSION)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = OUTPUT_DIR / f"{BIRD}_s{SESSION:03d}_stan_data.json"

    with output.open("w") as f:
        json.dump(data, f)

    print(f"Session: {BIRD}_s{SESSION:03d}")
    print(f"Responses: {data['N']}")
    print(f"Left responses: {sum(data['y'])}")
    print(f"Right responses: {data['N'] - sum(data['y'])}")
    print(f"Left reinforcers: {sum(data['r_left'])}")
    print(f"Right reinforcers: {sum(data['r_right'])}")
    print(f"Duration: {df['time_sec'].max() / 60:.1f} minutes")
    print(f"Stan data: {output}")


if __name__ == "__main__":
    main()
