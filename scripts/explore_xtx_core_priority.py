"""Controlled core-first objective validation; no default parameter changes."""
import _bootstrap
import json,argparse
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from xtx_inputs import load_inputs,core_mask
from sanos_generalized_grid import GridProduct
from run_xtx_generalized_grid import solve_generalized_grid

ROOT=Path(__file__).resolve().parents[1]

def stats(item,fit,core):
    err=np.maximum(item['pure_bid']-fit,0)+np.maximum(fit-item['pure_ask'],0)
    return dict(core_coverage=float(np.mean(err[core]<=1e-9)),coverage=float(np.mean(err<=1e-9)),
        core_outside_loss=float(np.sum(err[core]/(item['pure_ask'][core]-item['pure_bid'][core]))),
        wing_outside_loss=float(np.sum(err[~core]/(item['pure_ask'][~core]-item['pure_bid'][~core]))))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dates',nargs='+',default=['2025-04-01','2025-04-02','2025-06-03'])
    parser.add_argument('--output',type=Path,default=ROOT/'data/outputs/research_exploration/core_priority')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);rows=[]
    def record(row):
        rows.append(row);(args.output/'results.json').write_text(json.dumps(dict(status='provisional',results=rows),indent=2),encoding='utf8')
        pd.DataFrame(rows).to_csv(args.output/'results.csv',index=False);print(json.dumps(row),flush=True)
    for date in args.dates:
        inputs=load_inputs(date,.1);base=solve_generalized_grid(inputs,nx=16385);keys=list(inputs)
        base['per_expiry'].to_csv(args.output/f'{date}_paper.csv',index=False)
        failed=[j for j,r in enumerate(base['per_expiry'].to_dict('records')) if r['n_core_inside']<r['n_core']][:2]
        for j in failed:
            e=keys[j];item=inputs[e];core=core_mask(item['quotes']);prefix=GridProduct(16385)
            if j:prefix.weights=base['density_by_expiry'][keys[j-1]].copy()
            for priority in ['paper','core_first']:
                model=GridProduct(16385);model.weights=prefix.weights.copy()
                try:
                    r=model.fit_step(item['anchor_k'],base['increments_by_expiry'][e],item['market_k'],
                        item['pure_bid'],item['pure_ask'],item['pure_mid'],priority_mask=core if priority=='core_first' else None)
                    fit=model.calls(item['market_k']);calendar=float(np.min(model.calls(model.states)-prefix.calls(model.states)))
                    record(dict(experiment='fixed_prefix',date=date,dte=int(item['dte']),priority=priority,success=True,
                        **stats(item,fit,core),**{k:r[k] for k in ['core_optimal_outside_loss','core_actual_outside_loss','core_loss_limit','selected_stage','core_hard_bid_ask'] if k in r},
                        reprice_error=float(np.max(abs(fit-r['fit']))),calendar_min=calendar))
                except Exception as exc:record(dict(experiment='fixed_prefix',date=date,dte=int(item['dte']),priority=priority,success=False,error=str(exc)))
        for priority in ['paper','core_first']:
            try:
                result=base if priority=='paper' else solve_generalized_grid(inputs,nx=16385,fit_priority=priority)
                frame=result['per_expiry'];frame.to_csv(args.output/f'{date}_{priority}.csv',index=False)
                record(dict(experiment='entire_recursion',date=date,priority=priority,success=True,
                    core_coverage=float(frame.n_core_inside.sum()/frame.n_core.sum()),coverage=float(frame.n_inside.sum()/frame.n_quotes.sum()),
                    max_reprice_error=float(frame.reprice_error.max()),calendar_min=float(frame.calendar_min.min())))
            except Exception as exc:record(dict(experiment='entire_recursion',date=date,priority=priority,success=False,error=str(exc)))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
