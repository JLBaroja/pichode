from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
import json
import logging
import sys
import traceback

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_session import prepare_session

MODEL = Path("models/timescales/single_timescale.stan")
OUTPUT = Path("results/timescales/single_timescale")
DRAWS = OUTPUT / "draws"
STAN_OUTPUT = OUTPUT / "stan_csv"
SUMMARY = OUTPUT / "summary.csv"

BIRDS = ["p004", "p054", "p138", "p510", "p530", "p736"]
SESSIONS = range(1, 181)
PARAMETERS = ["tau"]

WORKERS = 2
CHAINS = 4
WARMUP = 500
SAMPLES = 250
ADAPT_DELTA = 0.95
MAX_TREEDEPTH = 10

OUTPUT.mkdir(parents=True, exist_ok=True)
DRAWS.mkdir(parents=True, exist_ok=True)
STAN_OUTPUT.mkdir(parents=True, exist_ok=True)

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def fit_session(task):
    bird, session = task
    label = f"{bird}_s{session:03d}"
    start = perf_counter()

    try:
        _, data = prepare_session(bird, session)

        model = CmdStanModel(
            stan_file=str(MODEL),
        )

        session_output = STAN_OUTPUT / label
        session_output.mkdir(parents=True, exist_ok=True)

        fit = model.sample(
            data=data,
            chains=CHAINS,
            parallel_chains=CHAINS,
            iter_warmup=WARMUP,
            iter_sampling=SAMPLES,
            adapt_delta=ADAPT_DELTA,
            max_treedepth=MAX_TREEDEPTH,
            inits={"tau": 100},
            seed=123 + session,
            output_dir=str(session_output),
            show_progress=False,
        )

        summary = fit.summary()
        diagnostics = fit.method_variables()

        divergences = int(diagnostics["divergent__"].sum())
        treedepth_hits = int(
            (diagnostics["treedepth__"] >= MAX_TREEDEPTH).sum()
        )

        posterior = fit.draws_pd(vars=PARAMETERS)

        # Save posterior draws before declaring the fit complete.
        posterior.to_csv(
            DRAWS / f"{label}.csv",
            index=False,
        )

        record = {
            "bird": bird,
            "session": session,
            "responses": data["N"],
            "reinforcers": sum(data["r_left"]) + sum(data["r_right"]),
            "seconds": round(perf_counter() - start, 2),
            "divergences": divergences,
            "treedepth_hits": treedepth_hits,
            "max_rhat": float(summary.loc[PARAMETERS, "R_hat"].max()),
            "min_ess_bulk": float(
                summary.loc[PARAMETERS, "ESS_bulk"].min()
            ),
            "status": "completed",
        }

        for parameter in PARAMETERS:
            values = posterior[parameter].to_numpy()

            record[f"{parameter}_mean"] = float(np.mean(values))
            record[f"{parameter}_median"] = float(np.median(values))
            record[f"{parameter}_q025"] = float(np.quantile(values, 0.025))
            record[f"{parameter}_q975"] = float(np.quantile(values, 0.975))
            record[f"{parameter}_rhat"] = float(
                summary.loc[parameter, "R_hat"]
            )
            record[f"{parameter}_ess_bulk"] = float(
                summary.loc[parameter, "ESS_bulk"]
            )

        record["preliminary_ok"] = (
            record["max_rhat"] < 1.05
            and divergences == 0
            and treedepth_hits == 0
        )

        return record

    except Exception:
        return {
            "bird": bird,
            "session": session,
            "seconds": round(perf_counter() - start, 2),
            "status": "failed",
            "error": traceback.format_exc(),
            "preliminary_ok": False,
        }


def main():
    # Compile once before launching workers.
    CmdStanModel(stan_file=str(MODEL))

    if SUMMARY.exists():
        previous = pd.read_csv(SUMMARY)
        records = previous.to_dict("records")
    else:
        records = []

    completed = {
        (r["bird"], int(r["session"]))
        for r in records
        if r.get("status") == "completed"
        and (DRAWS / f'{r["bird"]}_s{int(r["session"]):03d}.csv').exists()
    }

    tasks = [
        (bird, session)
        for session in SESSIONS
        for bird in BIRDS
        if (bird, session) not in completed
    ]

    print(f"Already completed: {len(completed)}", flush=True)
    print(f"Remaining sessions: {len(tasks)}", flush=True)
    print(f"Concurrent sessions: {WORKERS}", flush=True)

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(fit_session, task): task
            for task in tasks
        }

        for future in as_completed(futures):
            record = future.result()
            task = (record["bird"], int(record["session"]))

            # Replace previous records for retried sessions.
            records = [
                r for r in records
                if (r["bird"], int(r["session"])) != task
            ]
            records.append(record)

            df = pd.DataFrame(records)
            df = df.sort_values(["bird", "session"])

            # Atomic replacement avoids partially written summaries.
            temporary = SUMMARY.with_suffix(".tmp")
            df.to_csv(temporary, index=False)
            temporary.replace(SUMMARY)

            print(
                f'{record["bird"]}_s{record["session"]:03d} '
                f'{record["status"]} '
                f'{record["seconds"]:.1f}s '
                f'Rhat={record.get("max_rhat", float("nan")):.3f} '
                f'div={record.get("divergences", "-")} '
                f'OK={record.get("preliminary_ok", False)}',
                flush=True,
            )

    print("\nALL SESSIONS PROCESSED", flush=True)


if __name__ == "__main__":
    main()
