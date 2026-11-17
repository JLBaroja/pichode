from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
import json
import logging
import traceback
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
from prepare_session import prepare_session

BIRDS = ['p004', 'p054', 'p138', 'p510', 'p530', 'p736']
MODELS = {'M1': ('initial_strength.stan', ['tau', 'log_relative_strength', 'A0']), 'M2': ('initial_preference.stan', ['tau', 'log_relative_strength', 'A0', 'pi0'])}
OUTPUT = Path('results/timescales/initial_conditions')
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)


def run(task):
    name, bird, session = task
    directory = OUTPUT / name / f'{bird}_s{session:03d}'
    directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    try:
        _, data = prepare_session(bird, session)
        stan_file, params = MODELS[name]
        model = CmdStanModel(stan_file=str(Path('models/timescales') / stan_file))
        fit = model.sample(data=data, chains=4, parallel_chains=4, iter_warmup=500, iter_sampling=250, adapt_delta=0.95, max_treedepth=10, seed=123 + session, output_dir=str(directory), show_progress=False)
        summary = fit.summary()
        diagnostics = fit.method_variables()
        draws = fit.draws_pd(vars=params)
        record = {'model': name, 'bird': bird, 'session': session, 'responses': data['N'], 'seconds': round(perf_counter() - started, 2), 'divergences': int(diagnostics['divergent__'].sum()), 'treedepth_hits': int((diagnostics['treedepth__'] >= 10).sum()), 'status': 'completed'}
        for p in params:
            vals = draws[p].to_numpy()
            record.update({f'{p}_mean': float(np.mean(vals)), f'{p}_median': float(np.median(vals)), f'{p}_q025': float(np.quantile(vals, .025)), f'{p}_q975': float(np.quantile(vals, .975)), f'{p}_rhat': float(summary.loc[p, 'R_hat']), f'{p}_ess_bulk': float(summary.loc[p, 'ESS_bulk'])})
        record['preliminary_ok'] = record['divergences'] == 0 and record['treedepth_hits'] == 0 and all(np.isfinite(record[f'{p}_rhat']) and record[f'{p}_rhat'] < 1.05 for p in params)
        tmp = directory / 'posterior.tmp'
        draws.to_csv(tmp, index=False)
        tmp.replace(directory / 'posterior.csv')
        tmp = directory / 'summary.tmp'
        tmp.write_text(json.dumps(record, indent=2))
        tmp.replace(directory / 'summary.json')
        return record
    except Exception:
        record = {'model': name, 'bird': bird, 'session': session, 'seconds': round(perf_counter() - started, 2), 'status': 'failed', 'error': traceback.format_exc(), 'preliminary_ok': False}
        (directory / 'error.txt').write_text(record['error'])
        return record


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT / 'parallel_summary.csv'
    records = {}
    if summary_path.exists():
        try:
            for r in pd.read_csv(summary_path).to_dict('records'):
                records[(r['model'], r['bird'], int(r['session']))] = r
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            print('Existing combined summary is unreadable; recovering individual results.', flush=True)
    for name in MODELS:
        for bird in BIRDS:
            for session in range(1, 181):
                key = (name, bird, session)
                directory = OUTPUT / name / f'{bird}_s{session:03d}'
                summary_file = directory / 'summary.json'
                posterior_file = directory / 'posterior.csv'
                if summary_file.exists() and posterior_file.exists():
                    try:
                        records[key] = json.loads(summary_file.read_text())
                    except (ValueError, OSError):
                        pass
                elif posterior_file.exists() and key not in records:
                    # Recover finished work from a run interrupted before its combined summary was saved.
                    try:
                        df = pd.read_csv(posterior_file)
                        params = MODELS[name][1]
                        r = {'model': name, 'bird': bird, 'session': session, 'status': 'completed', 'preliminary_ok': False, 'recovered': True}
                        for p in params:
                            vals = df[p].to_numpy()
                            r.update({f'{p}_mean': float(np.mean(vals)), f'{p}_median': float(np.median(vals)), f'{p}_q025': float(np.quantile(vals, .025)), f'{p}_q975': float(np.quantile(vals, .975))})
                        records[key] = r
                    except (OSError, ValueError, KeyError, pd.errors.ParserError):
                        pass
    tasks = [(m, b, s) for s in range(1, 181) for b in BIRDS for m in MODELS if not (records.get((m, b, s), {}).get('status') == 'completed' and (OUTPUT / m / f'{b}_s{s:03d}' / 'posterior.csv').exists())]
    print(f'Already completed: {2160 - len(tasks)}; remaining: {len(tasks)}', flush=True)
    if not tasks:
        print('ALL FITS PROCESSED', flush=True)
        return
    for filename, _ in MODELS.values():
        CmdStanModel(stan_file=str(Path('models/timescales') / filename))
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            rec = future.result()
            records[(rec['model'], rec['bird'], int(rec['session']))] = rec
            if i % 20 == 0 or i == len(tasks):
                tmp = summary_path.with_suffix('.tmp')
                pd.DataFrame(list(records.values())).sort_values(['model', 'bird', 'session']).to_csv(tmp, index=False)
                tmp.replace(summary_path)
            eta = (len(tasks) - i) * (perf_counter() - started) / i / 60
            print(f"[{i}/{len(tasks)}] {rec['model']} {rec['bird']}_s{rec['session']:03d} {rec['status']} OK={rec.get('preliminary_ok', False)} ETA={eta:.1f}min", flush=True)
    print('ALL FITS PROCESSED', flush=True)


if __name__ == '__main__':
    main()
