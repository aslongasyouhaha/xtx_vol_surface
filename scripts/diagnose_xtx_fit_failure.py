"""Attribute generalized fit failure; provisional evidence, not default changes.

Compare all/core quote LPs, a fresh prefix and arbitrary independent increments.
The latter has a dual lower-bound check over ALL nonnegative multiplier values,
using an event sweep of a finite prefix distribution's call-price kinks.
No quadratic state-transition array is constructed.
"""
import _bootstrap
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
from xtx_inputs import load_inputs,core_mask
from sanos_generalized_grid import GridProduct
from run_xtx_generalized_grid import solve_generalized_grid
from diagnose_xtx_generalized_april import relaxed_marginal

ROOT=Path(__file__).resolve().parents[1]

def independent_kernel(prefix,a,k):
    a=np.asarray(a,float);k=np.asarray(k,float)
    result=np.zeros((len(k),len(a)));positive=a>0
    result[:,positive]=a[None,positive]*prefix.calls(k[:,None]/a[None,positive])
    return result

def minimum_reduced_cost(states,weights,strikes,t,lam0,lam1):
    """Minimum of sum(t_l E[(a X-k_l)+])-lam0-lam1*a for a>=0.

    With a finite positive prefix, this is piecewise linear with kinks k_l/X_i.
    Check every kink and zero; first make the asymptotic slope nonnegative.
    Returned safety margin is a conservative floating-point error allowance,
    not a formal interval-arithmetic proof.
    """
    s=np.asarray(states);p=np.asarray(weights);k=np.asarray(strikes);t=np.asarray(t)
    keep=p>0;s=s[keep];p=p[keep]
    if np.any(p<0) or np.any(s<=0):raise ValueError('Invalid prefix distribution')
    tail=float(t.sum()*(p@s));correction=max(0.,lam1-tail);lam1-=correction
    if len(s)*len(k)>5_000_000:raise ValueError('Event-sweep budget exceeded')
    knots=(k[:,None]/s[None,:]).ravel()
    jumps=(t[:,None]*(p*s)[None,:]).ravel();order=np.argsort(knots)
    knots=knots[order];jumps=jumps[order]
    sums=np.cumsum(jumps);intercepts=np.cumsum(jumps*knots)
    values=knots*(sums-lam1)-intercepts-lam0
    index=int(np.argmin(values));argmin=float(knots[index]);minimum=float(values[index])
    if -lam0<minimum:minimum=-lam0;argmin=0.
    eps=np.finfo(float).eps;gamma=(len(knots)+10)*eps/(1-(len(knots)+10)*eps)
    safety=8*gamma*(float(knots[-1])*(float(np.sum(abs(jumps)))+abs(lam1))+float(np.sum(abs(jumps*knots)))+abs(lam0)+1)
    return dict(minimum=minimum,argmin=argmin,lam1=lam1,tail_correction=correction,
        floating_safety=max(1e-7,float(safety)),events=len(knots))

def price_lp(A,a,bid,ask,mid=None):
    n,m=A.shape;I=sparse.eye(n);Z=sparse.csr_matrix((n,n))
    # No midpoint term: the optimum directly measures unavoidable outside loss.
    rows=sparse.vstack([sparse.hstack([A,-I]),sparse.hstack([-A,-I])],format='csr')
    E=sparse.csr_matrix(np.c_[np.array([np.ones(m),a]),np.zeros((2,n))])
    r=linprog(np.r_[np.zeros(m),1/(ask-bid)],A_ub=rows,b_ub=np.r_[ask,-bid],
        A_eq=E,b_eq=[1,1],bounds=(0,None),method='highs',
        options={'time_limit':20,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not r.success:return dict(success=False,status=int(r.status),message=r.message),None
    fit=A@r.x[:m];error=np.maximum(bid-fit,0)+np.maximum(fit-ask,0)
    residual=max(0,float(np.max(rows@r.x-np.r_[ask,-bid])),float(np.max(abs(E@r.x-1))),float(-r.x.min()))
    return dict(success=residual<=2e-7,status=int(r.status),message=r.message,objective=float(r.fun),
        outside_loss=float(np.sum(error/(ask-bid))),coverage=float(np.mean(error<=1e-9)),
        max_residual=residual),r

def independent_certificate(prefix,v,mask):
    k=v['market_k'][mask];bid=v['pure_bid'][mask];ask=v['pure_ask'][mask]
    a=np.unique(np.r_[0.,1.,np.geomspace(.05,5,301),v['anchor_k']])
    A=independent_kernel(prefix,a,k);stats,r=price_lp(A,a,bid,ask)
    if r is None:return stats
    n=len(k);up=r.ineqlin.marginals[:n];low=r.ineqlin.marginals[n:]
    if np.any(up>0) or np.any(low>0):return dict(**stats,certificate_valid=False,reason='dual_sign_error')
    scale=1/max(1.,float(np.max(-(up+low)*(ask-bid))))
    up=up*scale;low=low*scale;lam0,lam1=r.eqlin.marginals*scale
    reduced=minimum_reduced_cost(prefix.states,prefix.weights,k,low-up,float(lam0),float(lam1))
    bound=float(ask@up-bid@low+lam0+lam1-reduced['tail_correction']+reduced['minimum']-reduced['floating_safety'])
    # Re-evaluate the sweep's minimizer independently using exact finite-prefix calls.
    amin=reduced['argmin'];direct=float((low-up)@independent_kernel(prefix,[amin],k)[:,0]-lam0-reduced['lam1']*amin)
    agreement=abs(direct-reduced['minimum'])
    used=r.x[:len(a)]>1e-10
    positive_used=used&(a>0)
    return dict(**stats,all_multiplier_outside_loss_lower_bound=bound,
        certificate_valid=bool(agreement<=reduced['floating_safety']),sweep_direct_agreement=agreement,
        solution_mass_at_zero=float(r.x[0]) if a[0]==0 else 0.,
        min_positive_used_multiplier=float(a[positive_used].min()) if positive_used.any() else None,
        max_used_multiplier=float(a[used].max()) if used.any() else None,
        min_multiplier=float(amin),floating_safety=reduced['floating_safety'],events=reduced['events'],
        scope='fixed finite-grid prefix; arbitrary independent nonnegative unit-mean increment; floating-point certificate')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dates',nargs='+',default=['2025-04-01','2025-04-02'])
    parser.add_argument('--output',type=Path,default=ROOT/'data/outputs/research_exploration/failure_attribution')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);rows=[]
    for date in args.dates:
        inputs=load_inputs(date,.1);base=solve_generalized_grid(inputs,eta_s=.1,nx=16385);keys=list(inputs)
        failures=[j for j,r in enumerate(base['per_expiry'].to_dict('records')) if r['n_core_inside']<r['n_core']][:2]
        for j in failures:
            e=keys[j];v=inputs[e];mask=core_mask(v['quotes']);a=v['anchor_k'];dv=base['increments_by_expiry'][e]
            previous=GridProduct(16385)
            if j:previous.weights=base['density_by_expiry'][keys[j-1]].copy()
            kernel=previous.kernel(a,dv,v['market_k'])
            row=dict(date=date,dte=int(v['dte']),core_quotes=int(mask.sum()),prefix_mass_error=float(abs(previous.weights.sum()-1)),
                prefix_mean_error=float(abs(previous.weights@previous.states-1)))
            row['all_quote_lp'],all_result=price_lp(kernel,a,v['pure_bid'],v['pure_ask'])
            row['core_only_lp'],core_result=price_lp(kernel[mask],a,v['pure_bid'][mask],v['pure_ask'][mask])
            if all_result is not None and core_result is not None:
                f=v['quotes'].copy();f['diagnostic_core']=mask
                for name,result in [('all_quote_lp',all_result),('core_only_lp',core_result)]:
                    fit=kernel@result.x[:len(a)]
                    error=np.maximum(v['pure_bid']-fit,0)+np.maximum(fit-v['pure_ask'],0)
                    f[name+'_fit']=fit;f[name+'_outside_spreads']=error/(v['pure_ask']-v['pure_bid'])
                    row[name]['core_coverage']=float(np.mean(error[mask]<=1e-9))
                    row[name]['core_outside_loss']=float(np.sum(error[mask]/(v['pure_ask'][mask]-v['pure_bid'][mask])))
                f.to_csv(args.output/f'{date}_{v["dte"]}D_quotes.csv',index=False)
            fresh=GridProduct(16385);freshkernel=fresh.kernel(a,dv,v['market_k'][mask])
            row['no_prefix_core_lp'],_=price_lp(freshkernel,a,v['pure_bid'][mask],v['pure_ask'][mask])
            floor=previous.calls(v['market_k'][mask])
            row['core_jensen_blocked']=int(np.sum(floor>v['pure_ask'][mask]+1e-9))
            row['arbitrary_independent_core']=independent_certificate(previous,v,mask)
            coreitem=dict(v,market_k=v['market_k'][mask],pure_bid=v['pure_bid'][mask],
                pure_ask=v['pure_ask'][mask],pure_mid=v['pure_mid'][mask])
            row['free_marginal_core']=relaxed_marginal(previous,coreitem)
            rows.append(row);(args.output/'results.json').write_text(json.dumps(dict(status='provisional',results=rows),indent=2),encoding='utf8')
            print(json.dumps(row),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
