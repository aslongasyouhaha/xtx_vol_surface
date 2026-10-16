"""State-dependent transport controls: joint marginals, sequential chain and fixed prefix."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from sanos_core import build_convex_order_problem,call_kernel
from sanos_martingale_transport import couple_marginals,fit_state_dependent_transition,calls
from sanos_generalized_grid import GridProduct
from xtx_inputs import load_inputs,core_mask
from xtx_results import BLOCKS
from xtx_generalized_research import solve_generalized_grid

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


def fit_chain(items,output=None):
    ordered=sorted(items.items(),key=lambda x:x[1]['dte'])
    # Same broad boundaries used by the previously audited generalized grid.
    reference=GridProduct(16385).states
    states=np.array([reference[0],1.,reference[-1]]);p=np.array([0.,1.,0.])
    previous_strikes=[];rows=[]
    for step,(e,it) in enumerate(ordered):
        start=time.perf_counter()
        prior=calls(states,p,it['market_k'])
        jensen=np.maximum(prior-it['pure_ask'],0)
        r=fit_state_dependent_transition(states,p,it,preserve_strikes=previous_strikes,time_limit=30)
        if not r['validated']:raise ValueError(f'Transition audit failed at {e}: {r["errors"]}')
        core=core_mask(it['quotes']) if 'quotes' in it else np.ones(len(prior),bool)
        outside=np.maximum.reduce([it['pure_bid']-r['fit'],r['fit']-it['pure_ask'],np.zeros(len(prior))])
        inside=outside<=1e-9
        row=dict(exdate=str(e),dte=int(it['dte']),n_quotes=len(prior),n_inside=int(inside.sum()),
            n_core=int(core.sum()),n_core_inside=int(inside[core].sum()),
            coverage=float(inside.mean()),core_coverage=float(inside[core].mean()) if core.any() else None,
            outside_objective=r['outside_objective'],objective=r['objective'],
            jensen_blocked=int((jensen>1e-9).sum()),core_jensen_blocked=int(((jensen>1e-9)&core).sum()),
            jensen_loss_lower_bound=float(np.sum(jensen/(it['pure_ask']-it['pure_bid']))),
            max_audit_error=max(r['errors'].values()),seconds=time.perf_counter()-start)
        rows.append(row)
        if output:
            output.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(output/f'transition_{step}.npz',original_states=states,original_weights=p,
                grid=r['grid'],source=r['source'],target=r['target'],Q=r['Q'],
                market_k=it['market_k'],bid=it['pure_bid'],ask=it['pure_ask'],fit=r['fit'])
            sparse.save_npz(output/f'projection_{step}.npz',r['projection'])
            pd.DataFrame(rows).to_csv(output/'per_expiry.csv',index=False)
            (output/f'audit_{step}.json').write_text(json.dumps(r['errors'],indent=2),encoding='utf8')
        states,p=r['grid'],r['target'];previous_strikes=it['market_k']
    f=pd.DataFrame(rows)
    return dict(validated=True,fit_target_met=bool(f.n_inside.sum()==f.n_quotes.sum()),
        expiries=len(f),coverage=float(f.n_inside.sum()/f.n_quotes.sum()),
        core_coverage=float(f.n_core_inside.sum()/f.n_core.sum()),
        objective=float(f.objective.sum()),max_audit_error=float(f.max_audit_error.max()),
        jensen_blocked=int(f.jensen_blocked.sum()),core_jensen_blocked=int(f.core_jensen_blocked.sum()))


def diagnostic_eta(date):
    for eta,dates in BLOCKS.values():
        if date in dates:return eta
    raise ValueError('Specify --eta for dates outside the validated diagnostic cases')


def run_joint(dates,output,eta=None):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    results=[]
    requested_eta=eta
    for date in dates:
        eta=diagnostic_eta(date) if requested_eta is None else requested_eta
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


def run_chain(dates,output,eta=.1):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    results=[]
    for date in dates:
        start=time.perf_counter()
        try:r=dict(date=date,**fit_chain(load_inputs(date,eta),out/date))
        except Exception as exc:r=dict(date=date,validated=False,error=str(exc))
        r['seconds']=time.perf_counter()-start;results.append(r);print(json.dumps(r),flush=True)
        (out/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf8')


def run_fixed_prefix(date,output,eta=.1):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    inputs=load_inputs(date,eta);items=list(inputs.values())
    baseline=solve_generalized_grid(inputs,nx=16385)
    saved={'states':baseline['states'],'density_-1':GridProduct(16385).weights}
    for j,e in enumerate(inputs):saved[f'density_{j}']=baseline['density_by_expiry'][e]
    rows=[]
    for j,it in enumerate(items):
        if it['dte'] not in [3,6,7,10]:continue
        start=time.perf_counter()
        try:
            result=fit_state_dependent_transition(saved['states'],saved[f'density_{j-1}'],it,
                preserve_strikes=items[j-1]['market_k'],time_limit=40)
            core=core_mask(it['quotes']);fit=result['fit'];inside=(fit>=it['pure_bid']-1e-9)&(fit<=it['pure_ask']+1e-9)
            row=dict(dte=int(it['dte']),validated=result['validated'],
                coverage=result['coverage'],core_coverage=float(inside[core].mean()),
                objective=result['objective'],outside_objective=result['outside_objective'],
                errors=result['errors'],grid_size=len(result['grid']))
            idx=np.searchsorted(np.cumsum(result['source']),[.1,.5,.9])
            row['conditional_samples']=[dict(state=float(result['grid'][i]),
                relative_variance=float(result['conditional_relative_variance'][i])) for i in idx]
            prefix=out/f'dte_{it["dte"]}'
            np.savez_compressed(str(prefix)+'.npz',grid=result['grid'],source=result['source'],
                target=result['target'],Q=result['Q'],original_states=saved['states'],
                original_weights=saved[f'density_{j-1}'],market_k=it['market_k'],
                bid=it['pure_bid'],ask=it['pure_ask'],mid=it['pure_mid'],fit=fit)
            from scipy import sparse
            sparse.save_npz(str(prefix)+'_projection.npz',result['projection'])
        except Exception as exc:row=dict(dte=int(it['dte']),validated=False,error=str(exc))
        row['seconds']=time.perf_counter()-start;rows.append(row);print(json.dumps(row),flush=True)
        (out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
