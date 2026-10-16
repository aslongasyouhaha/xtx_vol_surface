"""Isolate HiGHS scaling effects without changing the SANOS LP or tolerances."""
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
from sanos_core import build_convex_order_problem
from sanos_incremental import PersistentSanosLP,compile_native_model
from benchmark_xtx_incremental import frames,core_mask

CASES={
    'adaptive':{},
    'default':{'simplex_scale_strategy':2},
    'no_internal_scaling':{'simplex_scale_strategy':0},
    'limited_scaling':{'simplex_scale_strategy':2,'allowed_matrix_scale_factor':5},
    'max_value_scaling':{'simplex_scale_strategy':4},
    'direct_unscaled':{'simplex_scale_strategy':2,'simplex_unscaled_solution_strategy':2},
    'default_then_unscaled':{'simplex_scale_strategy':2},
    'ipm_then_unscaled':{'simplex_scale_strategy':0,'solver':'ipm'},
}
TRANSITIONS={'default_then_unscaled':{'simplex_scale_strategy':0},
             'ipm_then_unscaled':{'solver':'simplex'}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dates',nargs='+',default=['2025-04-01'])
    parser.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    parser.add_argument('--time-limit',type=float,default=15.)
    parser.add_argument('--output',type=Path,default=ROOT/'data/outputs'/'numerics_scaling')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=[];solutions={}
    for date in args.dates:
        eta=.1 if date=='2025-04-01' else .25
        inputs=load_inputs(date,eta,baseline_variant='baseline');problem=build_convex_order_problem(inputs,weight_mode='legacy',mid_penalty=0.)
        compiled=compile_native_model(problem);updates=frames(problem,'sparse')
        active={ex:(np.asarray(v['pure_bid']).copy(),np.asarray(v['pure_ask']).copy()) for ex,v in problem['items'].items()}
        active_ex=min(problem['items'],key=lambda ex:abs(problem['items'][ex]['dte']-30))
        indices=np.flatnonzero(core_mask(problem['items'][active_ex]['quotes']));active_index=int(indices[len(indices)//2])
        bid,ask=active[active_ex];width=ask[active_index]-bid[active_index]
        bid[active_index]=ask[active_index]+.05*width;ask[active_index]=bid[active_index]+width
        for name in args.cases:
            mode='adaptive' if name=='adaptive' else 'legacy'
            session=PersistentSanosLP(inputs,problem=problem,compiled=compiled,time_limit=args.time_limit,numerics_profile=mode,working_set=False)
            for key,value in CASES[name].items():session._check(session.h.setOptionValue(key,value))
            active_warm=None
            for step in ['initial','identical','one_quote','active_quote','cold_active_quote']:
                if step=='identical':
                    for key,value in TRANSITIONS.get(name,{}).items():session._check(session.h.setOptionValue(key,value))
                if step=='one_quote':session.update_quotes(updates[1][0])
                if step=='active_quote':
                    previous_x=x.copy();session.update_quotes(active)
                    v=problem['items'][active_ex]
                    previous_fit=float(v['G'][active_index]@previous_x[slice(*problem['q_offsets'][active_ex])])
                    forced_violation=float(active[active_ex][0][active_index]-previous_fit)
                if step=='cold_active_quote':
                    session=PersistentSanosLP(inputs,problem=problem,compiled=compiled,time_limit=args.time_limit,numerics_profile=mode,working_set=False)
                    for key,value in CASES[name].items():session._check(session.h.setOptionValue(key,value))
                    session.update_quotes(active)
                started=time.perf_counter();row=session.solve();total=time.perf_counter()-started
                x=row.pop('x')
                row.update(date=date,eta=eta,case=name,step=step,total_seconds=total,
                           options=CASES[name],transition_options=TRANSITIONS.get(name,{}),version=session.h.version())
                if step=='active_quote':
                    row.update(old_fit_violation=forced_violation,active_dte=problem['items'][active_ex]['dte'],active_strike=float(problem['items'][active_ex]['market_k'][active_index]))
                    if row['success']:active_warm=row['actual_outside_objective']
                if step=='cold_active_quote' and row['success'] and active_warm is not None:
                    row['warm_cold_objective_difference']=abs(row['actual_outside_objective']-active_warm)
                    row['objective_agrees']=row['warm_cold_objective_difference']<=1e-6
                    row['success']=row['success'] and row['objective_agrees']
                if row['success']:solutions[date+'__'+name+'__'+step]=x
                records.append(row)
                (args.output/'results.json').write_text(json.dumps(records,indent=2),encoding='utf8')
                import pandas as pd
                pd.DataFrame([{k:v for k,v in r.items() if k!='options'} for r in records]).to_csv(args.output/'results.csv',index=False)
                print(json.dumps(row),flush=True)
                if not row['success']:break
    np.savez(args.output/'solutions.npz',**solutions)


if __name__=='__main__':main()
