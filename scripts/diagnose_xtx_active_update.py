"""Compare reoptimization methods from one identical pre-update simplex basis."""
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
from xtx_inputs import load_inputs,core_mask
from sanos_core import build_convex_order_problem
from sanos_incremental import PersistentSanosLP,compile_native_model

METHODS={
    'retained_dual':{},
    'rebuilt_dual':{},
    'rebuilt_primal':{'simplex_strategy':4},
    'rebuilt_devex':{'simplex_dual_edge_weight_strategy':1},
    'rebuilt_no_perturbation':{'dual_simplex_cost_perturbation_multiplier':0.},
}


def active_quotes(problem):
    quotes={ex:(np.asarray(v['pure_bid']).copy(),np.asarray(v['pure_ask']).copy()) for ex,v in problem['items'].items()}
    ex=min(problem['items'],key=lambda ex:abs(problem['items'][ex]['dte']-30))
    indices=np.flatnonzero(core_mask(problem['items'][ex]['quotes']));index=int(indices[len(indices)//2])
    bid,ask=quotes[ex];width=ask[index]-bid[index]
    bid[index]=ask[index]+.05*width;ask[index]=bid[index]+width
    return quotes,ex,index


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dates',nargs='+',default=['2025-04-01'])
    parser.add_argument('--methods',nargs='+',choices=list(METHODS),default=list(METHODS))
    parser.add_argument('--time-limit',type=float,default=12.)
    parser.add_argument('--output',type=Path,default=ROOT/'data/outputs'/'active_update_methods')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    reference=json.loads((ROOT/'data/outputs'/'numerics_adaptive'/'results.json').read_text())
    results=[];solutions={}
    for date in args.dates:
        eta=.1 if date=='2025-04-01' else .25
        inputs=load_inputs(date,eta,baseline_variant='baseline');problem=build_convex_order_problem(inputs,weight_mode='legacy',mid_penalty=0.);compiled=compile_native_model(problem)
        seed=PersistentSanosLP(inputs,problem=problem,compiled=compiled,time_limit=30.,working_set=False)
        initial=seed.solve()
        if not initial['success']:raise RuntimeError('Initial model failed original constraints')
        basis=seed.h.getBasis();profile=seed._profile_name
        quotes,ex,index=active_quotes(problem)
        previous_fit=float(problem['items'][ex]['G'][index]@initial['x'][slice(*problem['q_offsets'][ex])])
        violation=float(quotes[ex][0][index]-previous_fit)
        assert violation>0,'Active update must invalidate the previous solution'
        control=next(r for r in reference if r['date']==date and r['step']=='cold_active_quote')
        for method in args.methods:
            started=time.perf_counter()
            if method=='retained_dual':session=seed;session.numerics_profile='legacy'
            else:
                session=PersistentSanosLP(inputs,problem=problem,compiled=compiled,time_limit=args.time_limit,numerics_profile='legacy')
                session._create_solver(profile)
                session._check(session.h.setBasis(basis))
            session.time_limit=args.time_limit
            for name,value in METHODS[method].items():session._check(session.h.setOptionValue(name,value))
            changes=session.update_quotes(quotes);setup_seconds=time.perf_counter()-started
            started=time.perf_counter();row=session.solve();request_seconds=time.perf_counter()-started
            x=row.pop('x')
            row.update(date=date,eta=eta,method=method,profile=profile,setup_seconds=setup_seconds,
                       request_seconds=request_seconds,total_seconds=setup_seconds+request_seconds,
                       changed_rows=changes['updated_rows'],changed_costs=changes['updated_costs'],
                       old_fit_violation=violation,active_dte=problem['items'][ex]['dte'])
            if row['success']:
                delta=abs(row['actual_outside_objective']-control['actual_outside_objective'])
                row.update(objective_difference_to_cold=delta,objective_agrees=delta<=1e-6)
                row['success']=row['success'] and row['objective_agrees']
                if row['success']:solutions[date+'__'+method]=x
            results.append(row)
            (args.output/'results.json').write_text(json.dumps(results,indent=2),encoding='utf8')
            import pandas as pd
            pd.DataFrame(results).to_csv(args.output/'results.csv',index=False)
            print(json.dumps(row),flush=True)
    np.savez(args.output/'solutions.npz',**solutions)


if __name__=='__main__':main()
