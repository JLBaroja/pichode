from pathlib import Path
from time import perf_counter
import sys

import pandas as pd
from cmdstanpy import CmdStanModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_session import prepare_session

MODEL = "models/timescales/two_timescales.stan"
OUTPUT = Path("results/timescales/benchmark_v2")
OUTPUT.mkdir(parents=True, exist_ok=True)

BIRDS = ["p054", "p736"]
SESSIONS = [30, 150]

model = CmdStanModel(stan_file=MODEL)
records = []

for bird in BIRDS:
    for session in SESSIONS:
        if (bird, session) not in [("p054", 30), ("p736", 150)]:
            continue
        label = f"{bird}_s{session:03d}"
        print(f"\nFITTING {label}", flush=True)

        _, data = prepare_session(bird, session)
        start = perf_counter()

        try:
            fit = model.sample(
                data=data,
                chains=4,
                parallel_chains=4,
                iter_warmup=500,
                iter_sampling=250,
                adapt_delta=0.95,
                max_treedepth=10,
                inits={
                    "tau_fast": 100,
                    "tau_slow": 900,
                    "w": 0.12,
                },
                seed=123,
                output_dir=str(OUTPUT),
                show_progress=False,
            )

            elapsed = perf_counter() - start
            summary = fit.summary()

            divergences = int(
                fit.method_variables()["divergent__"].sum()
            )

            record = {
                "bird": bird,
                "session": session,
                "responses": data["N"],
                "seconds": round(elapsed, 2),
                "divergences": divergences,
                "max_rhat": summary.loc[
                    ["tau_fast", "tau_slow", "w"], "R_hat"
                ].max(),
                "min_ess": summary.loc[
                    ["tau_fast", "tau_slow", "w"], "ESS_bulk"
                ].min(),
                "status": "completed",
            }

        except Exception as e:
            record = {
                "bird": bird,
                "session": session,
                "responses": data["N"],
                "seconds": round(perf_counter() - start, 2),
                "status": "failed",
                "error": str(e),
            }

        records.append(record)

        # Save immediately so interrupted runs retain their results.
        pd.DataFrame(records).to_csv(
            OUTPUT / "benchmark_summary.csv",
            index=False,
        )

        print(record, flush=True)

print("\nBENCHMARK COMPLETE")
print(pd.DataFrame(records).to_string(index=False))
