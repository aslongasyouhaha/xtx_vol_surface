"""Controlled identical-basis latency comparison for the current paper LP.

No geometry, weights, midpoint penalty, quote targets or audit tolerance change.
Model construction is timed separately from solve requests.
"""
import _bootstrap  # project paths for direct script execution
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
DEPS=ROOT/'data/outputs/native_deps'
if (DEPS/'highspy').exists():sys.path.insert(0,str(DEPS))
from xtx_inputs import load_inputs,core_mask
from sanos_incremental import PersistentSanosLP


def summarize(result):
    keys=['success','request_seconds','solve_seconds','validation_seconds','iterations',
          'optimization_path','working_set_seconds','working_set_fallback_reason',
          'actual_objective','max_constraint_violation','numerics_attempts','working_set_attempts',
          'global_objective_gap']
    return {k:result[k] for k in keys if k in result}


def clone_session(inputs,original,basis,*,working_set,budget=.75):
    s=PersistentSanosLP(inputs,problem=original.problem,compiled=original.compiled,
        working_set=working_set,working_set_seconds=budget,time_limit=30)
    if original._profile_name!=s._profile_name:s._create_solver(original._profile_name)
    if s._profile_name=='ipm_seed':s._check(s.h.setOptionValue('solver','simplex'))
    s._check(s.h.setBasis(basis))
    for key in ['_accepted_x','_accepted_dual','_accepted_upper','_accepted_costs']:
        setattr(s,key,getattr(original,key).copy())
    return s


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dates',nargs='+',default=['2025-01-02','2025-04-01','2025-06-02'])
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    out=ROOT/'data/outputs/paper_latency';out.mkdir(parents=True,exist_ok=True)
    records=[]
    for date in args.dates:
        eta=.1 if date=='2025-04-01' else .25
        inputs=load_inputs(date,eta)
        from sanos_core import build_convex_order_problem
        problem=build_convex_order_problem(inputs,calendar_grid='union')
        start=time.perf_counter();original=PersistentSanosLP(inputs,problem=problem,working_set=False,time_limit=30)
        construction=time.perf_counter()-start;initial=original.solve()
        if not initial['success']:raise RuntimeError(initial['message'])
        basis=original.h.getBasis();p=original.problem;matrix=original.matrix
        q=np.concatenate([initial['x'][slice(*v)] for v in p['q_offsets'].values()])
        row=dict(date=date,eta=eta,construction_seconds=construction,initial=summarize(initial),
            rows=matrix.shape[0],columns=matrix.shape[1],nnz=matrix.nnz,
            coefficient_abs_min=float(abs(matrix.data).min()),coefficient_abs_max=float(abs(matrix.data).max()),
            fraction_coefficients_below_1e_minus9=float(np.mean(abs(matrix.data)<1e-9)),
            positive_cost_min=float(original.costs[original.costs>0].min()),positive_cost_max=float(original.costs.max()),
            q_fraction_below_1e_minus12=float(np.mean(q<1e-12)),cases=[])
        ex=min(p['items'],key=lambda e:abs(p['items'][e]['dte']-30));item=p['items'][ex]
        indices=np.flatnonzero(core_mask(item['quotes']));idx=int(indices[len(indices)//2])
        quotes={e:(b.copy(),a.copy()) for e,(b,a) in original.quotes.items()}
        bid,ask=quotes[ex];width=ask[idx]-bid[idx]
        oldfit=float(item['G'][idx]@initial['x'][slice(*p['q_offsets'][ex])])
        bid[idx]=max(ask[idx],oldfit)+.05*width;ask[idx]=bid[idx]+width
        row['changed_dte']=int(item['dte']);row['old_fit_violation']=float(bid[idx]-oldfit)
        for repeat in range(args.repeats):
            # Alternate execution order to reduce systematic first-run bias.
            cases=[('direct_full',False,.75),('local_750ms',True,.75),('local_50ms',True,.05)]
            if repeat%2:cases=cases[::-1]
            for name,working_set,budget in cases:
                session=clone_session(inputs,original,basis,working_set=working_set,budget=budget)
                # A copied basis alone does not copy its numerical factorization.
                # Prepare it with unchanged quotes BEFORE timing the update.
                prepared=session.solve()
                if not prepared['success']:raise RuntimeError('Basis prewarm failed')
                update_start=time.perf_counter();session.update_quotes(quotes)
                update_seconds=time.perf_counter()-update_start
                result=session.solve()
                record=dict(case=name,repeat=repeat,quote_update_seconds=update_seconds,
                    prewarm_iterations=prepared['iterations'],prewarm_profile=prepared['numerics_profile'],
                    **summarize(result))
                if not result['success']:raise RuntimeError(f'{date} {name}: {result["message"]}')
                row['cases'].append(record)
        objectives=[r['actual_objective'] for r in row['cases']]
        row['objective_range']=max(objectives)-min(objectives)
        row['all_objectives_agree_1e6']=row['objective_range']<=1e-6
        row['medians']={name:float(np.median([r['request_seconds'] for r in row['cases'] if r['case']==name]))
                        for name in ['direct_full','local_750ms','local_50ms']}
        records.append(row)
        (out/'results.json').write_text(json.dumps(records,indent=2),encoding='utf8')
        print(json.dumps({k:row[k] for k in ['date','construction_seconds','rows','columns','nnz','medians',
            'objective_range','all_objectives_agree_1e6']}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
