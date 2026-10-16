"""Provisional controlled experiments; does not modify defaults or conclusions."""
import _bootstrap
import argparse,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEPS=ROOT/'data/outputs/native_deps'
if (DEPS/'highspy').exists():sys.path.insert(0,str(DEPS))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from xtx_inputs import load_inputs,core_mask
from sanos_incremental import PersistentSanosLP,recompute_loss_variables
from sanos_core import build_convex_order_problem
from sanos_daily import snapshot,daily_references,stability_qp,active_support_qp
from sanos_generalized_grid import GridProduct
from sanos_surface import SanosSurface,audit_surface
from diagnose_xtx_generalized_april import relaxed_marginal
from run_xtx_generalized_grid import solve_generalized_grid

ROOT=Path(__file__).resolve().parents[1]

def fit_stats(p,x):
    total=inside=nc=ic=0;loss=0.
    for e,v in p['items'].items():
        fit=v['G']@x[slice(*p['q_offsets'][e])]
        outside=np.maximum(v['pure_bid']-fit,0)+np.maximum(fit-v['pure_ask'],0)
        c=core_mask(v['quotes']);total+=len(fit);inside+=int(np.sum(outside<=1e-9))
        nc+=int(c.sum());ic+=int(np.sum(outside[c]<=1e-9));loss+=float(np.sum(outside/(v['pure_ask']-v['pure_bid'])))
    surface=SanosSurface([v.get('T',v['dte']/365) for v in p['items'].values()],
        [v['anchor_k'] for v in p['items'].values()],[x[slice(*p['q_offsets'][e])] for e in p['items']],
        [v['smooth_variance'] for v in p['items'].values()])
    return dict(coverage=inside/total,core_coverage=ic/nc,outside_loss=loss,surface_audit=audit_surface(surface))

def saved_lp(date,block,eta):
    p=build_convex_order_problem(load_inputs(date,eta));x=np.zeros(p['cursor'])
    with np.load(ROOT/f'data/outputs/daily_validation/{block}/{date}/lp/model.npz') as model:
        for e,v in p['items'].items():
            key=str(e.date());np.testing.assert_array_equal(v['anchor_k'],model[key+'__anchors'])
            x[slice(*p['q_offsets'][e])]=model[key+'__q']
    quotes={e:(v['pure_bid'],v['pure_ask']) for e,v in p['items'].items()}
    recompute_loss_variables(p,x,quotes)
    return p,x,quotes

def eta_cases(record,budget):
    for date in ['2025-04-01','2025-04-02','2025-06-03']:
        inputs=load_inputs(date,.25)
        for boundary in ['default','left_020']:
            for eta in [.1,.25]:
                changed={e:dict(v,anchor_k=(np.unique(np.r_[.2,v['anchor_k']]) if boundary=='left_020' else v['anchor_k']),
                    smooth_variance=eta*float(v['quotes'].atm_total_variance.iloc[0])) for e,v in inputs.items()}
                t=time.perf_counter();session=PersistentSanosLP(changed,time_limit=budget);result=session.solve()
                row=dict(experiment='eta_boundary',date=date,boundary=boundary,eta=eta,success=result['success'],
                    seconds=time.perf_counter()-t,message=result['message'])
                if result['success']:row.update(fit_stats(session.problem,result['x']))
                record(row)

def generalized_cases(record):
    date='2025-04-01';inputs=load_inputs(date,.1)
    baseline=solve_generalized_grid(inputs,eta_s=.1,nx=16385)
    record(dict(experiment='generalized_fresh_baseline',date=date,success=True,
        core_coverage=float(baseline['per_expiry'].n_core_inside.sum()/baseline['per_expiry'].n_core.sum()),
        coverage=float(baseline['per_expiry'].n_inside.sum()/baseline['per_expiry'].n_quotes.sum())))
    ordered=list(inputs)
    for j,(e,v) in enumerate(inputs.items()):
        if v['dte'] not in [3,6]:continue
        prefix=GridProduct(16385)
        if j:prefix.weights=baseline['density_by_expiry'][ordered[j-1]].copy()
        a=v['anchor_k'];dv=baseline['increments_by_expiry'][e];assert len(a)==len(dv)
        c=core_mask(v['quotes']);floor=prefix.calls(v['market_k'])
        blocked=floor>v['pure_ask']+1e-9
        common=dict(experiment='generalized_fixed_prefix',date=date,dte=int(v['dte']),
            jensen_blocked=int(blocked.sum()),core_jensen_blocked=int(blocked[c].sum()))
        for factor in [0.,.25,1.,4.]:
            model=GridProduct(16385);model.weights=prefix.weights.copy()
            t=time.perf_counter()
            try:
                r=model.fit_step(a,dv*factor,v['market_k'],v['pure_bid'],v['pure_ask'],v['pure_mid'])
                fit=model.calls(v['market_k']);out=np.maximum(v['pure_bid']-fit,0)+np.maximum(fit-v['pure_ask'],0)
                audit=dict(reprice_error=float(np.max(abs(fit-r['fit']))),mass_error=float(abs(model.weights.sum()-1)),
                    mean_error=float(abs(model.weights@model.states-1)),calendar_min=float(np.min(model.calls(model.states)-prefix.calls(model.states))))
                valid=audit['reprice_error']<=2e-7 and audit['mass_error']<=2e-7 and audit['mean_error']<=2e-7 and audit['calendar_min']>=-2e-7
                record(dict(**common,factor=factor,success=valid,coverage=float(np.mean(out<=1e-9)),
                    core_coverage=float(np.mean(out[c]<=1e-9)),outside_loss=float(np.sum(out/(v['pure_ask']-v['pure_bid']))),
                    seconds=time.perf_counter()-t,**audit))
            except Exception as exc:record(dict(**common,factor=factor,success=False,error=str(exc)))
        record(dict(experiment='generalized_free_marginal',date=date,dte=int(v['dte']),**relaxed_marginal(prefix,v)))

def qp_cases(record,budget):
    # Both support controls use exactly the same prior LP reference and guards.
    # This is an isolated support comparison, not a sequential QP deployment replay.
    for block,old,date,eta in [('april','2025-04-01','2025-04-02',.1),('june','2025-06-03','2025-06-04',.25)]:
        pp,px,_=saved_lp(old,block,eta);p,x,quotes=saved_lp(date,block,eta)
        refs=daily_references(snapshot(old,pp,px),p,date)
        for support,allowance in [('restricted',0.),('restricted',.01),('full',.01)]:
            if support=='restricted':r=active_support_qp(p,x,quotes,refs,loss_allowance=allowance,time_limit=budget)
            else:r=stability_qp(p,x,quotes,refs,loss_allowance=allowance,time_limit=budget,backend_order=('clarabel',))
            row=dict(experiment='daily_qp_support',date=date,source_date=old,support=support,loss_allowance=allowance,
                budget_seconds=budget,success=r['success'],reason=r.get('reason'),seconds=r['seconds'],attempts=r['attempts'])
            if r['success']:
                row.update(fit_stats(p,r['x']),lp_reference_rms=r['lp_movement_rms'],qp_reference_rms=r['qp_movement_rms'],
                    feasible_step_alpha=r['feasible_step_alpha'],selected_point_is_optimizer_result=r['selected_point_is_optimizer_result'])
            record(row)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments',nargs='+',choices=['eta','generalized','qp'],default=['eta','generalized','qp'])
    parser.add_argument('--budget',type=float,default=15.)
    parser.add_argument('--output',type=Path,default=ROOT/'data/outputs/research_exploration')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);rows=[]
    def record(row):
        rows.append(row)
        (args.output/'results.json').write_text(json.dumps(dict(status='provisional',results=rows),indent=2),encoding='utf8')
        pd.DataFrame([{k:v for k,v in r.items() if k not in ['attempts','surface_audit']} for r in rows]).to_csv(args.output/'results.csv',index=False)
        print(json.dumps({k:v for k,v in row.items() if k not in ['attempts','surface_audit']}),flush=True)
    for name in args.experiments:
        try:
            if name=='eta':eta_cases(record,args.budget)
            elif name=='generalized':generalized_cases(record)
            else:qp_cases(record,args.budget)
        except Exception as exc:record(dict(experiment=name,success=False,error=str(exc)))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
