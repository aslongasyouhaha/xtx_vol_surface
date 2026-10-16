"""Local two-expiry weight lookback with protected earlier quote errors."""
import _bootstrap
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
from sanos_generalized_grid import GridProduct
from xtx_inputs import load_inputs,core_mask
from run_xtx_generalized_grid import solve_generalized_grid

ROOT=Path(__file__).resolve().parents[1]

def model_with(weights):
    m=GridProduct(len(weights));m.weights=weights.copy();return m

def quote_stats(v,fit):
    err=np.maximum(v['pure_bid']-fit,0)+np.maximum(fit-v['pure_ask'],0)
    core=core_mask(v['quotes'])
    return dict(coverage=float(np.mean(err<=1e-9)),core_coverage=float(np.mean(err[core]<=1e-9)),
        outside_loss=float(np.sum(err/(v['pure_ask']-v['pure_bid'])))),err

def protected_first(A1,A2,a,first,second,baseline_error):
    A=np.vstack([A1,A2]);m=A.shape[1];n=A.shape[0]
    bid=np.r_[first['pure_bid'],second['pure_bid']];ask=np.r_[first['pure_ask'],second['pure_ask']]
    mid=np.r_[first['pure_mid'],second['pure_mid']];I=sparse.eye(n);Z=sparse.csr_matrix((n,n))
    padded=lambda B:sparse.hstack([B,sparse.csr_matrix((B.shape[0],2*n))])
    inequalities=sparse.vstack([sparse.hstack([A,-I,Z]),sparse.hstack([-A,-I,Z]),
        sparse.hstack([A,Z,-I]),sparse.hstack([-A,Z,-I]),padded(A1),padded(-A1)],format='csr')
    rhs=np.r_[ask,-bid,mid,-mid,first['pure_ask']+baseline_error,-first['pure_bid']+baseline_error]
    E=np.c_[np.array([np.ones(m),a]),np.zeros((2,2*n))]
    result=linprog(np.r_[np.zeros(m),1/(ask-bid),1e-8/(ask-bid)],A_ub=inequalities,b_ub=rhs,
        A_eq=E,b_eq=[1,1],bounds=(0,None),method='highs',
        options={'time_limit':10,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:raise RuntimeError(result.message)
    residual=max(0,float(np.max(inequalities@result.x-rhs)),float(np.max(abs(E@result.x-1))))
    if residual>2e-7:raise ValueError('Lookback LP original-unit audit failed')
    return result.x[:m]

def pair(inputs,baseline,j,iterations):
    keys=list(inputs);e1,e2=keys[j:j+2];first,second=inputs[e1],inputs[e2]
    prefix=baseline['density_by_expiry'][keys[j-1]].copy()
    q1=baseline['q_by_expiry'][e1].copy();q2=baseline['q_by_expiry'][e2].copy()
    a1,a2=first['anchor_k'],second['anchor_k'];v1=baseline['increments_by_expiry'][e1];v2=baseline['increments_by_expiry'][e2]
    A1=model_with(prefix).kernel(a1,v1,first['market_k']);basefit=A1@q1
    _,base_error=quote_stats(first,basefit);trace=[]
    for iteration in range(iterations+1):
        earlier=model_with(prefix);earlier.advance(a1,v1,q1)
        later=model_with(earlier.weights);later.advance(a2,v2,q2)
        fit1=earlier.calls(first['market_k']);fit2=later.calls(second['market_k'])
        s1,err1=quote_stats(first,fit1);s2,_=quote_stats(second,fit2)
        valid=(np.max(err1-base_error)<=1e-9 and np.all((err1<=1e-9)[base_error<=1e-9])
            and min(np.min(earlier.calls(earlier.states)-model_with(prefix).calls(earlier.states)),
                np.min(later.calls(later.states)-earlier.calls(later.states)))>=-2e-7)
        trace.append(dict(iteration=iteration,earlier=s1,later=s2,
            earlier_max_error_increase=float(np.max(err1-base_error)),protection_and_calendar_passed=bool(valid),
            combined_outside_loss=s1['outside_loss']+s2['outside_loss']))
        if not valid:break
        if iteration==iterations:break
        # For fixed q2, independent multiplicative increments commute.
        swapped=model_with(prefix);swapped.advance(a2,v2,q2)
        A2=swapped.kernel(a1,v1,second['market_k'])
        q1=protected_first(A1,A2,a1,first,second,base_error)
        earlier=model_with(prefix);earlier.advance(a1,v1,q1)
        later=model_with(earlier.weights)
        q2=later.fit_step(a2,v2,second['market_k'],second['pure_bid'],second['pure_ask'],second['pure_mid'])['q']
    return dict(earlier_dte=int(first['dte']),later_dte=int(second['dte']),trace=trace,
        success=bool(all(r['protection_and_calendar_passed'] for r in trace)),
        scope='two adjacent maturities on one observation date; not a global fit or daily update'),q1,q2

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dates',nargs='+',default=['2025-04-01','2025-04-02'])
    p.add_argument('--iterations',type=int,default=5)
    p.add_argument('--output',type=Path,default=ROOT/'data/outputs/research_exploration/prefix_lookback')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);rows=[]
    for date in args.dates:
        inputs=load_inputs(date,.1);baseline=solve_generalized_grid(inputs,eta_s=.1,nx=16385)
        for j in [1,2]:
            try:
                row,q1,q2=pair(inputs,baseline,j,args.iterations);row['date']=date
                np.savez_compressed(args.output/f'{date}_pair{j}.npz',q1=q1,q2=q2)
                print(json.dumps(dict(date=date,earlier_dte=row['earlier_dte'],later_dte=row['later_dte'],
                    success=row['success'],initial=row['trace'][0],final=row['trace'][-1])),flush=True)
            except Exception as exc:row=dict(date=date,pair_index=j,success=False,error=str(exc));print(json.dumps(row),flush=True)
            rows.append(row);(args.output/'results.json').write_text(json.dumps(dict(status='provisional',results=rows),indent=2),encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
