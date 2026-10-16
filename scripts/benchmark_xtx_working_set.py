"""Certified local updates versus full LP on the same binding synthetic quote."""
import _bootstrap  # project paths for direct script execution
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEPS=ROOT/'data/outputs'/'native_deps'
if (DEPS/'highspy').exists():sys.path.insert(0,str(DEPS))
from xtx_inputs import load_inputs
from sanos_incremental import PersistentSanosLP,quote_vectors
from sanos_working_set import restricted_update
from diagnose_xtx_active_update import active_quotes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dates',nargs='+',default=['2025-01-02','2025-04-01','2025-06-02'])
    parser.add_argument('--local-budget',type=float,default=.75)
    parser.add_argument('--max-expiries',type=int,default=5)
    parser.add_argument('--integrated',action='store_true',help='Exercise the public update interface, including fallback')
    parser.add_argument('--output',type=Path,default=ROOT/'data/outputs'/'working_set_benchmark')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=[];solutions={}
    for date in args.dates:
        eta=.1 if date=='2025-04-01' else .25
        s=PersistentSanosLP(load_inputs(date,eta,baseline_variant='baseline'),time_limit=30.,working_set=False,objective_mode='legacy')
        initial=s.solve();assert initial['success'],initial['message']
        previous_x=initial['x'].copy();dual=np.asarray(s.h.getSolution().row_dual)
        upper,costs=quote_vectors(s.problem,s.quotes)
        quotes,ex,index=active_quotes(s.problem)
        old_fit=float(s.problem['items'][ex]['G'][index]@previous_x[slice(*s.problem['q_offsets'][ex])])
        old_violation=float(quotes[ex][0][index]-old_fit);assert old_violation>0
        s.update_quotes(quotes)
        if args.integrated:
            s.working_set=True;s.working_set_seconds=args.local_budget;s.working_set_expiries=args.max_expiries
            started=time.perf_counter();result=s.solve();elapsed=time.perf_counter()-started
            assert result['success'] and result['validated'],result['message']
            row=dict(date=date,eta=eta,path=result['optimization_path'],request_seconds=elapsed,
                     objective=result['actual_outside_objective'],gap=result.get('global_objective_gap'),
                     max_constraint_violation=result['max_constraint_violation'],
                     fallback_reason=result.get('working_set_fallback_reason'))
            records.append(row)
            (args.output/'results.json').write_text(json.dumps(records,indent=2),encoding='utf8')
            print(json.dumps(row),flush=True)
            continue
        started=time.perf_counter()
        local=restricted_update(s,previous_x,dual,upper,costs,deadline=started+args.local_budget,max_expiries=args.max_expiries)
        local_seconds=time.perf_counter()-started
        started=time.perf_counter();full=s.solve();full_seconds=time.perf_counter()-started
        row=dict(date=date,eta=eta,local_success=local['success'],local_seconds=local_seconds,
                 local_attempts=local['attempts'],local_reason=local.get('reason'),old_fit_violation=old_violation,
                 full_success=full['success'],full_seconds=full_seconds,full_iterations=full['iterations'],
                 full_objective=full.get('actual_outside_objective'))
        if local['success']:
            row.update(local_audit=local['audit'],global_lower_bound=local['lower_bound'],objective_gap=local['objective_gap'])
            solutions[date+'__local']=local['x']
            if full['success']:
                delta=abs(local['audit']['actual_outside_objective']-full['actual_outside_objective'])
                row.update(objective_difference_to_full=delta,objective_agrees=delta<=1e-6)
        if full['success']:solutions[date+'__full']=full['x']
        records.append(row)
        (args.output/'results.json').write_text(json.dumps(records,indent=2),encoding='utf8')
        import pandas as pd
        pd.DataFrame([{k:v for k,v in r.items() if k not in {'local_attempts','local_audit'}} for r in records]).to_csv(args.output/'summary.csv',index=False)
        print(json.dumps(row),flush=True)
    np.savez(args.output/'solutions.npz',**solutions)


if __name__=='__main__':main()
