from pathlib import Path
from time import perf_counter
import argparse
import logging

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

from prepare_session import prepare_session


MODELS = {
    "M0": {
        "file": "single_timescale.stan",
        "parameters": ["tau"],
    },
    "M1": {
        "file": "initial_strength.stan",
        "parameters": ["tau", "log_relative_strength", "A0"],
    },
    "M2": {
        "file": "initial_preference.stan",
        "parameters": ["tau", "log_relative_strength", "A0", "pi0"],
    },
}

MODEL_DIR = Path("models/timescales")
OUTPUT_DIR = Path("results/timescales/initial_conditions")

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def fit_session(model_name, bird, session):
    config = MODELS[model_name]
    label = f"{bird}_s{session:03d}"

    output = OUTPUT_DIR / model_name / label
    output.mkdir(parents=True, exist_ok=True)

    _, data = prepare_session(bird, session)

    model = CmdStanModel(
        stan_file=str(MODEL_DIR / config["file"])
    )

    start = perf_counter()

    fit = model.sample(
        data=data,
        chains=4,
        parallel_chains=4,
        iter_warmup=500,
        iter_sampling=250,
        adapt_delta=0.95,
        max_treedepth=10,
        seed=123 + session,
        output_dir=str(output),
        show_progress=False,
    )

    summary = fit.summary()
    diagnostics = fit.method_variables()

    parameters = config["parameters"]

    posterior = fit.draws_pd(vars=parameters)
    posterior.to_csv(output / "posterior.csv", index=False)

    records = []

    for parameter in parameters:
        values = posterior[parameter].to_numpy()

        records.append({
            "model": model_name,
            "bird": bird,
            "session": session,
            "parameter": parameter,
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
            "rhat": float(summary.loc[parameter, "R_hat"]),
            "ess_bulk": float(summary.loc[parameter, "ESS_bulk"]),
            "divergences": int(diagnostics["divergent__"].sum()),
            "treedepth_hits": int(
                (diagnostics["treedepth__"] >= 10).sum()
            ),
            "seconds": round(perf_counter() - start, 2),
        })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bird", default="p530")
    parser.add_argument("--session", type=int, default=127)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
    )

    args = parser.parse_args()

    all_results = []

    for model_name in args.models:
        print(
            f"Fitting {model_name}: "
            f"{args.bird}_s{args.session:03d}",
            flush=True,
        )

        results = fit_session(
            model_name,
            args.bird,
            args.session,
        )

        all_results.append(results)

        print(
            results[
                ["parameter", "mean", "rhat", "ess_bulk"]
            ].to_string(index=False)
        )

    combined = pd.concat(all_results, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combined.to_csv(
        OUTPUT_DIR / "pilot_summary.csv",
        index=False,
    )

    print("\nPilot completed.")


if __name__ == "__main__":
    main()
