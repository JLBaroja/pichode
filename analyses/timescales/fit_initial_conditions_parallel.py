from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
import logging
import traceback
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
from prepare_session import prepare_session

BIRDS = ['p004', 'p054', 'p138', 'p510', 'p530', 'p736']
MODELS = {'M1': ('initial_strength.stan', ['tau','log_relative_strength','A0']), 'M2': ('initial_preference.stan', ['tau','log_relative_strength','A0','pi0'])}
OUTPUT = Path('results/timescales/initial_conditions')
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

def run(task):
    name, bird, session = task
    label = f'{bird}_s{session:03d}'
    directory = OUTPUT / name / label
    directory.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    try:
        _, data = prepare_session(bird, session)
        stan_file, params = MODELS[name]
        model = CmdStanModel(stan_file=str(Path('models/timescales') / stan_file), compile=True)
        fit = model.sample(data=data, chains=4, parallel_chains=4, iter_warmup=500, iter_sampling=250, adapt_delta=0.95, max_treedepth=10, seed=123+session, output_dir=str(directory), show_progress=False)
        summ = fit.summary()
        diagnostics = fit.method_variables()
        draws = fit.draws_pd(vars=params)
        record = {'model':name,'bird':bird,'session':session,'responses':data['N'],'seconds':round(perf_counter()-t0,2),'divergences':int(diagnostics['divergent__'].sum()),'treedepth_hits':int((diagnostics['treedepth__']>=10).sum()),'status':'completed'}
        for p in params:
            vals = draws[p].to_numpy()
            record.update({f'{p}_mean':float(np.mean(vals)),f'{p}_median':float(np.median(vals)),f'{p}_q025':float(np.quantile(vals,.025)),f'{p}_q975':float(np.quantile(vals,.975)),f'{p}_rhat':float(summ.loc[p,'R_hat']),f'{p}_ess_bulk':float(summ.loc[p,'ESS_bulk'])})
        record['preliminary_ok'] = record['divergences']==0 and record['treedepth_hits']==0 and all(np.isfinite(record[f'{p}_rhat']) and record[f'{p}_rhat']<1.05 for p in params)
        temp = directory / 'posterior.tmp'
        draws.to_csv(temp,index=False)
        temp.replace(directory / 'posterior.csv')
        return record
    except Exception:
        return {'model':name,'bird':bird,'session':session,'seconds':round(perf_counter()-t0,2),'status':'failed','error':traceback.format_exc(),'preliminary_ok':False}

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    summary_path = OUTPUT / 'parallel_summary.csv'
    if summary_path.exists():
        records = pd.read_csv(summary_path).to_dict('records')
    else:
        records = []
    completed = {(r['model'],r['bird'],int(r['session'])) for r in records if r.get('status')=='completed' and (OUTPUT/r['model']/f"{r['bird']}_s{int(r['session']):03d}"/'posterior.csv').exists()}
    tasks = [(m,b,s) for s in range(1,181) for b in BIRDS for m in MODELS if (m,b,s) not in completed]
    print(f'Completed: {len(completed)}; remaining: {len(tasks)}',flush=True)
    for filename,_ in MODELS.values():
        CmdStanModel(stan_file=str(Path('models/timescales')/filename))
    start=perf_counter()
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(run,task):task for task in tasks}
        for i,future in enumerate(as_completed(futures),1):
            rec=future.result()
            key=(rec['model'],rec['bird'],int(rec['session']))
            records=[r for r in records if (r['model'],r['bird'],int(r['session']))!=key]
            records.append(rec)
            df=pd.DataFrame(records).sort_values(['model','bird','session'])
            temp=summary_path.with_suffix('.tmp')
            df.to_csv(temp,index=False)
            temp.replace(summary_path)
            remaining=(len(tasks)-i)*(perf_counter()-start)/i/60
            print(f"[{i}/{len(tasks)}] {rec['model']} {rec['bird']}_s{rec['session']:03d} {rec['status']} OK={rec.get('preliminary_ok',False)} ETA={remaining:.1f}min",flush=True)
    print('ALL FITS PROCESSED',flush=True)

if __name__=='__main__':
    main()
