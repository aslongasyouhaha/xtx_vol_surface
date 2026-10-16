"""Provisional increment-dictionary and fit-preserving daily-QP controls."""
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
from sanos_daily import snapshot,daily_references
from sanos_generalized_grid import GridProduct
from run_xtx_generalized_grid import solve_generalized_grid
from explore_xtx_research import saved_lp,fit_stats
from explore_xtx_fit_preserving_qp import fit_preserving_qp

def increments(record):
    for date in ['2025-04-01','2025-04-02']:
        inputs=load_inputs(date,.1);baseline=solve_generalized_grid(inputs,eta_s=.1,nx=16385)
        keys=list(inputs)
        # Test the second and third expiry, with identical current-model prefix.
        for j in [1,2]:
            e=keys[j];v=inputs[e];original=v['anchor_k'];dv=baseline['increments_by_expiry'][e]
            wide=np.unique(np.r_[original,np.geomspace(.2,3,121),np.linspace(.7,1.3,121),1.])
            cases=[('baseline',original,dv),('wide_zero_variance',wide,np.zeros(len(wide))),
                ('same_states_variance_mixture',np.tile(original,3),np.concatenate([dv*0,dv*.25,dv]))]
            for name,a,w in cases:
                if len(a)>1024:raise ValueError('Increment dictionary exceeds 1024-column experiment cap')
                model=GridProduct(16385);model.weights=baseline['density_by_expiry'][keys[j-1]].copy()
                previous=model.calls(model.states);t=time.perf_counter()
                row=dict(experiment='independent_increment_dictionary',date=date,dte=int(v['dte']),
                    dictionary=name,columns=len(a),prefix='fresh canonical independent-increment fit',independence_preserved=True)
                try:
                    r=model.fit_step(a,w,v['market_k'],v['pure_bid'],v['pure_ask'],v['pure_mid'])
                    fit=model.calls(v['market_k']);err=np.maximum(v['pure_bid']-fit,0)+np.maximum(fit-v['pure_ask'],0)
                    c=core_mask(v['quotes']);audit=dict(mass_error=float(abs(model.weights.sum()-1)),
                        mean_error=float(abs(model.weights@model.states-1)),reprice_error=float(np.max(abs(fit-r['fit']))),
                        calendar_min=float(np.min(model.calls(model.states)-previous)))
                    valid=audit['mass_error']<=2e-7 and audit['mean_error']<=2e-7 and audit['reprice_error']<=2e-7 and audit['calendar_min']>=-2e-7
                    row.update(success=valid,coverage=float(np.mean(err<=1e-9)),core_coverage=float(np.mean(err[c]<=1e-9)),
                        outside_loss=float(np.sum(err/(v['pure_ask']-v['pure_bid']))),seconds=time.perf_counter()-t,**audit)
                except Exception as exc:row.update(success=False,error=str(exc),seconds=time.perf_counter()-t)
                record(row)

def daily(record,budget,out):
    for block,eta,dates in [('april',.1,['2025-04-01','2025-04-02','2025-04-03']),
            ('june',.25,['2025-06-02','2025-06-03','2025-06-04'])]:
        for old,date in zip(dates[:-1],dates[1:]):
            pp,px,_=saved_lp(old,block,eta);p,x,quotes=saved_lp(date,block,eta)
            refs=daily_references(snapshot(old,pp,px),p,date)
            r=fit_preserving_qp(p,x,quotes,refs,time_limit=budget)
            row=dict(experiment='pointwise_fit_preserving_QP',date=date,source_date=old,
                reference='previous daily LP; isolated comparison, not sequential QP replay',
                **{k:v for k,v in r.items() if k not in ['x','audit']})
            if r['success']:
                row.update(fit_stats(p,r['x']))
                np.savez_compressed(out/(date+'_qp.npz'),x=r['x'])
            record(row)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiments',nargs='+',choices=['generalized','qp'],default=['generalized','qp'])
    p.add_argument('--budget',type=float,default=20.)
    p.add_argument('--output',type=Path,default=ROOT/'data/outputs/research_exploration/flexibility')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);rows=[]
    def record(row):
        rows.append(row);(args.output/'results.json').write_text(json.dumps(dict(status='provisional',results=rows),indent=2),encoding='utf8')
        pd.DataFrame([{k:v for k,v in r.items() if k!='surface_audit'} for r in rows]).to_csv(args.output/'results.csv',index=False)
        print(json.dumps({k:v for k,v in row.items() if k!='surface_audit'}),flush=True)
    for name in args.experiments:
        try:
            if name=='generalized':increments(record)
            else:daily(record,args.budget,args.output)
        except Exception as exc:record(dict(experiment=name,success=False,error=str(exc)))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
