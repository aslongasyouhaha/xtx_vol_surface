"""Joint paper-LP marginals plus explicit latent martingale transitions.

Independent lognormal smoothing with nondecreasing total variance can then
be added to this latent chain, reproducing the original price parameterization.
"""
import _bootstrap  # project paths for direct script execution
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
from xtx_inputs import load_inputs,core_mask
from sanos_core import build_convex_order_problem,call_kernel
from sanos_martingale_transport import couple_marginals


def strict_joint(inputs):
    problem=build_convex_order_problem(inputs,weight_mode='paper',mid_penalty=1e-8)
    start=time.perf_counter()
    bounds=[(0,None)]*len(problem['c'])
    for method,presolve in [('highs-ds',False),('highs-ipm',True),('highs-ipm',False)]:
        for refinement in range(4):
            remaining=60-(time.perf_counter()-start)
            if remaining<=0:raise RuntimeError('Joint LP time budget exhausted')
            r=linprog(problem['c'],A_ub=problem['A'],b_ub=problem['b'],A_eq=problem['E'],
                b_eq=np.ones(problem['E'].shape[0]),bounds=bounds,method=method,
                options={'presolve':presolve,'time_limit':remaining,
                         'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
            if not r.success:break
            negative=[a+i for a,b in problem['q_offsets'].values() for i in np.flatnonzero(r.x[a:b]<0)]
            if not negative:break
            for index in negative:bounds[index]=(0,0)
        if not r.success:continue
        q={e:r.x[a:b].copy() for e,(a,b) in problem['q_offsets'].items()}
        if min(v.min() for v in q.values())<0:continue
        error=max(float(np.max(abs(problem['E']@r.x-1))),float(np.max(problem['A']@r.x-problem['b'])),0.)
        if error<=2e-7:return problem,q,error
    raise RuntimeError('Joint LP did not return independently validated nonnegative marginals')


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dates',nargs='+',default=['2025-01-02','2025-04-01','2025-06-02'])
    args=parser.parse_args()
    root=Path('data/outputs/generalized_grid/joint_transport');root.mkdir(parents=True,exist_ok=True)
    results=[]
    for date,eta in [('2025-01-02',.25),('2025-04-01',.1),('2025-06-02',.25)]:
        if date not in args.dates:continue
        start=time.perf_counter();out=root/date;out.mkdir(exist_ok=True)
        try:
            problem,qs,error=strict_joint(load_inputs(date,eta));rows=[]
            states=np.array([1.]);p=np.array([1.]);variance=0.
            for j,(e,it) in enumerate(problem['items'].items()):
                w=float(it['smooth_variance'])
                if w<variance-1e-15:raise ValueError('Decreasing lognormal smoothing variance')
                for tolerance in [0.,1e-10,1e-9]:
                    try:
                        r=couple_marginals(states,p,it['anchor_k'],qs[e],time_limit=15,marginal_tolerance=tolerance)
                        break
                    except RuntimeError:
                        if tolerance==1e-9:raise
                if not r['validated']:raise ValueError(f'Coupling audit failed: {r["errors"]}')
                fit=call_kernel(r['target_states'],it['market_k'],w)@r['target_weights']
                reference=it['G']@qs[e]
                inside=(fit>=it['pure_bid']-1e-9)&(fit<=it['pure_ask']+1e-9)
                original=(reference>=it['pure_bid']-1e-9)&(reference<=it['pure_ask']+1e-9)
                core=core_mask(it['quotes'])
                rows.append(dict(exdate=str(e),dte=int(it['dte']),n_quotes=len(fit),n_inside=int(inside.sum()),
                    n_reference_inside=int(original.sum()),n_core=int(core.sum()),n_core_inside=int(inside[core].sum()),
                    max_reprice_difference=float(np.max(abs(fit-reference))),
                    max_transition_error=max(r['errors'].values()),smoothing_variance=w,
                    marginal_tolerance=r['marginal_tolerance'],marginal_l1_error=r['marginal_l1_error'],
                    smoothing_increment=w-variance,source_active=len(r['source_states']),target_active=len(r['target_states'])))
                np.savez_compressed(out/f'transition_{j}.npz',source_states=r['source_states'],
                    target_states=r['target_states'],source_weights=r['source_weights'],
                    target_weights=r['target_weights'],Q=r['Q'],
                    requested_target=r['requested_target'],marginal_tolerance=r['marginal_tolerance'],
                    marginal_l1_error=r['marginal_l1_error'],
                    smoothing_variance=w,smoothing_increment=w-variance,
                    market_k=it['market_k'],fit=fit,reference_fit=reference)
                pd.DataFrame(rows).to_csv(out/'per_expiry.csv',index=False)
                states,p,variance=r['target_states'],r['target_weights'],w
            f=pd.DataFrame(rows)
            result=dict(date=date,eta=eta,validated=True,expiries=len(f),joint_lp_error=error,
                coverage=float(f.n_inside.sum()/f.n_quotes.sum()),
                reference_coverage=float(f.n_reference_inside.sum()/f.n_quotes.sum()),
                core_coverage=float(f.n_core_inside.sum()/f.n_core.sum()),
                max_reprice_difference=float(f.max_reprice_difference.max()),
                max_transition_error=float(f.max_transition_error.max()),
                max_marginal_tolerance=float(f.marginal_tolerance.max()),max_marginal_l1_error=float(f.marginal_l1_error.max()))
        except Exception as exc:result=dict(date=date,eta=eta,validated=False,error=str(exc))
        result['seconds']=time.perf_counter()-start;results.append(result);print(json.dumps(result),flush=True)
        # Preserve results for dates omitted by --dates.
        old=json.loads((root/'summary.json').read_text(encoding='utf8')) if (root/'summary.json').exists() else []
        by_date={x['date']:x for x in old}
        by_date.update({x['date']:x for x in results})
        (root/'summary.json').write_text(json.dumps(list(by_date.values()),indent=2),encoding='utf8')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
