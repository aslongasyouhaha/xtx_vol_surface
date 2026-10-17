"""Generalized-model calibration, controlled experiments and failure certificates.

These experiments do not modify the paper LP defaults. Certificates apply only
to their fixed finite-grid prefix and retain their floating-point safeguards."""

import json
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog, brentq
from sanos_core import call_kernel
from sanos_generalized import GridProduct
from xtx_io import load_inputs, core_mask

def smile(item):
    pairs=[]
    for k,c in zip(item['market_k'],item['pure_mid']):
        try:
            w=brentq(lambda w:call_kernel([1.],[k],w)[0,0]-c,1e-10,9.)
            pairs.append((k,w))
        except ValueError:pass
    return np.array(sorted(pairs)).T if pairs else np.empty((2,0))


def solve_generalized_grid(inputs,eta_s=.1,nx=16385,quadrature=128,*,fit_priority='paper',core_loss_tolerance=1e-5):
    if fit_priority not in ['paper','core_first']:raise ValueError('Unknown fit priority')
    model=GridProduct(nx,quadrature);prev=None;rows=[];qs={};dens={};increments={}
    # All knots certify calendar order of the piecewise-linear grid prices;
    # a sparse sample alone would miss inter-strike violations.
    grid=np.unique(np.r_[model.states,np.geomspace(.01,10,2001),1.])
    previous=model.calls(grid)
    for e,it in sorted(inputs.items(),key=lambda x:x[1]['dte']):
        a=np.asarray(it['anchor_k']);sw=smile(it)
        if sw.shape[1]>=3:
            wn=np.interp(a,*sw)
            dv=np.maximum(eta_s*(wn if prev is None else wn-np.interp(a,*prev)),1e-7)
            prev=sw
        else:dv=np.full(len(a),max(it['smooth_variance'],1e-7))
        g=it.get('quotes');core=np.ones(len(it['market_k']),bool)
        if g is not None:
            from xtx_io import core_mask
            core=core_mask(g)
        result=model.fit_step(a,dv,it['market_k'],it['pure_bid'],it['pure_ask'],it['pure_mid'],
            priority_mask=core if fit_priority=='core_first' else None,core_loss_tolerance=core_loss_tolerance)
        fit=result['fit'];current=model.calls(grid)
        reprice=float(np.max(abs(fit-model.calls(it['market_k']))))
        calendar=float(np.min(current-previous))
        if reprice>2e-7 or calendar < -2e-7:raise ValueError('Price/calendar audit failed')
        inside=(fit>=it['pure_bid']-1e-9)&(fit<=it['pure_ask']+1e-9)
        rows.append(dict(exdate=str(e),dte=int(it['dte']),coverage=float(inside.mean()),
            n_quotes=len(fit),n_inside=int(inside.sum()),n_core=int(core.sum()),n_core_inside=int(inside[core].sum()),
            core_coverage=float(inside[core].mean()) if core.any() else None,
            mass_error=float(abs(model.weights.sum()-1)),mean_error=float(abs(model.weights@model.states-1)),
            reprice_error=reprice,calendar_min=calendar,objective=result['objective'],fit_priority=fit_priority,
            **{k:result[k] for k in ['core_optimal_outside_loss','core_actual_outside_loss','core_loss_limit',
                'core_hard_bid_ask','selected_stage','second_stage_accepted'] if k in result}))
        qs[e]=result['q'];dens[e]=model.weights.copy();increments[e]=dv
        previous=current
    return dict(success=True,numerically_validated=True,
        fit_target_met=all(r['n_inside']==r['n_quotes'] for r in rows),
        message='grid audits passed; approximation, not exact tensor',
        per_expiry=pd.DataFrame(rows),q_by_expiry=qs,density_by_expiry=dens,
        increments_by_expiry=increments,exdates=list(qs),states=model.states,
        eta_s=eta_s,fit_priority=fit_priority,calendar_min_global=min(r['calendar_min'] for r in rows))


def relaxed_marginal(previous,item):
    """Free next marginal on same grid, constrained to dominate fixed previous.

    Call-value variables make full-grid convex-order constraints sparse.
    This relaxes independent-increment restrictions, not calendar order.
    """
    s=previous.states;n=len(s);k=np.asarray(item['market_k']);m=len(k)
    d=np.diff(s)
    slope=sparse.diags([-1/d,1/d],[0,1],shape=(n-1,n),format='csr')
    convex=slope[:-1]-slope[1:]
    j=np.searchsorted(s,k)-1;t=(k-s[j])/(s[j+1]-s[j])
    G=sparse.coo_matrix((np.r_[1-t,t],(np.r_[np.arange(m),np.arange(m)],np.r_[j,j+1])),shape=(m,n)).tocsr()
    I=sparse.eye(m);Z=sparse.csr_matrix((m,m))
    def padded(a):return sparse.hstack([a,sparse.csr_matrix((a.shape[0],2*m))])
    A=sparse.vstack([padded(slope),padded(-slope),padded(convex),
        sparse.hstack([G,-I,Z]),sparse.hstack([-G,-I,Z]),
        sparse.hstack([G,Z,-I]),sparse.hstack([-G,Z,-I])],format='csr')
    bid,ask,mid=[np.asarray(item[x]) for x in ['pure_bid','pure_ask','pure_mid']]
    b=np.r_[np.zeros(n-1),np.ones(n-1),np.zeros(n-2),ask,-bid,mid,-mid]
    lower=previous.calls(s)
    bounds=[(max(0,float(x)),None) for x in lower]+[(0,None)]*(2*m)
    bounds[0]=(1-s[0],1-s[0]);bounds[n-1]=(0,0)
    c=np.r_[np.zeros(n),1/(ask-bid),1e-8/(ask-bid)]
    result=linprog(c,A_ub=A,b_ub=b,bounds=bounds,method='highs',
        options={'time_limit':30,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:return dict(success=False,message=result.message)
    fit=G@result.x[:n]
    residual=max(float(np.max(A@result.x-b)),float(np.max(lower-result.x[:n])),0.)
    return dict(success=residual<=2e-7,objective=float(result.fun),
        actual_outside_objective=float(np.sum(np.maximum.reduce([bid-fit,fit-ask,np.zeros(m)])/(ask-bid))),
        max_residual=residual,coverage=float(np.mean((fit>=bid-1e-9)&(fit<=ask+1e-9))))


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
    n,m=A.shape;I=sparse.eye(n)
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


def run_failure_cases(dates,output):
    output.mkdir(parents=True,exist_ok=True);rows=[]
    for date in dates:
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
                f.to_csv(output/f'{date}_{v["dte"]}D_quotes.csv',index=False)
            fresh=GridProduct(16385);freshkernel=fresh.kernel(a,dv,v['market_k'][mask])
            row['no_prefix_core_lp'],_=price_lp(freshkernel,a,v['pure_bid'][mask],v['pure_ask'][mask])
            floor=previous.calls(v['market_k'][mask])
            row['core_jensen_blocked']=int(np.sum(floor>v['pure_ask'][mask]+1e-9))
            row['arbitrary_independent_core']=independent_certificate(previous,v,mask)
            coreitem=dict(v,market_k=v['market_k'][mask],pure_bid=v['pure_bid'][mask],
                pure_ask=v['pure_ask'][mask],pure_mid=v['pure_mid'][mask])
            row['free_marginal_core']=relaxed_marginal(previous,coreitem)
            rows.append(row);(output/'results.json').write_text(json.dumps(dict(status='provisional',results=rows),indent=2),encoding='utf8')
            print(json.dumps(row),flush=True)


def stats(item,fit,core):
    err=np.maximum(item['pure_bid']-fit,0)+np.maximum(fit-item['pure_ask'],0)
    return dict(core_coverage=float(np.mean(err[core]<=1e-9)),coverage=float(np.mean(err<=1e-9)),
        core_outside_loss=float(np.sum(err[core]/(item['pure_ask'][core]-item['pure_bid'][core]))),
        wing_outside_loss=float(np.sum(err[~core]/(item['pure_ask'][~core]-item['pure_bid'][~core]))))


def run_core_priority(dates,output):
    output.mkdir(parents=True,exist_ok=True);rows=[]
    def record(row):
        rows.append(row);(output/'results.json').write_text(json.dumps(dict(status='provisional',results=rows),indent=2),encoding='utf8')
        pd.DataFrame(rows).to_csv(output/'results.csv',index=False);print(json.dumps(row),flush=True)
    for date in dates:
        inputs=load_inputs(date,.1);base=solve_generalized_grid(inputs,nx=16385);keys=list(inputs)
        base['per_expiry'].to_csv(output/f'{date}_paper.csv',index=False)
        failed=[j for j,r in enumerate(base['per_expiry'].to_dict('records')) if r['n_core_inside']<r['n_core']][:2]
        for j in failed:
            e=keys[j];item=inputs[e];core=core_mask(item['quotes']);prefix=GridProduct(16385)
            if j:prefix.weights=base['density_by_expiry'][keys[j-1]].copy()
            for priority in ['paper','core_first']:
                model=GridProduct(16385);model.weights=prefix.weights.copy()
                try:
                    r=model.fit_step(item['anchor_k'],base['increments_by_expiry'][e],item['market_k'],
                        item['pure_bid'],item['pure_ask'],item['pure_mid'],priority_mask=core if priority=='core_first' else None)
                    fit=model.calls(item['market_k']);calendar=float(np.min(model.calls(model.states)-prefix.calls(model.states)))
                    record(dict(experiment='fixed_prefix',date=date,dte=int(item['dte']),priority=priority,success=True,
                        **stats(item,fit,core),**{k:r[k] for k in ['core_optimal_outside_loss','core_actual_outside_loss','core_loss_limit','selected_stage','core_hard_bid_ask'] if k in r},
                        reprice_error=float(np.max(abs(fit-r['fit']))),calendar_min=calendar))
                except Exception as exc:record(dict(experiment='fixed_prefix',date=date,dte=int(item['dte']),priority=priority,success=False,error=str(exc)))
        for priority in ['paper','core_first']:
            try:
                result=base if priority=='paper' else solve_generalized_grid(inputs,nx=16385,fit_priority=priority)
                frame=result['per_expiry'];frame.to_csv(output/f'{date}_{priority}.csv',index=False)
                record(dict(experiment='entire_recursion',date=date,priority=priority,success=True,
                    core_coverage=float(frame.n_core_inside.sum()/frame.n_core.sum()),coverage=float(frame.n_inside.sum()/frame.n_quotes.sum()),
                    max_reprice_error=float(frame.reprice_error.max()),calendar_min=float(frame.calendar_min.min())))
            except Exception as exc:record(dict(experiment='entire_recursion',date=date,priority=priority,success=False,error=str(exc)))


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


def run_lookback(dates,output,iterations=5):
    output.mkdir(parents=True,exist_ok=True);rows=[]
    for date in dates:
        inputs=load_inputs(date,.1);baseline=solve_generalized_grid(inputs,eta_s=.1,nx=16385)
        for j in [1,2]:
            try:
                row,q1,q2=pair(inputs,baseline,j,iterations);row['date']=date
                np.savez_compressed(output/f'{date}_pair{j}.npz',q1=q1,q2=q2)
                print(json.dumps(dict(date=date,earlier_dte=row['earlier_dte'],later_dte=row['later_dte'],
                    success=row['success'],initial=row['trace'][0],final=row['trace'][-1])),flush=True)
            except Exception as exc:row=dict(date=date,pair_index=j,success=False,error=str(exc));print(json.dumps(row),flush=True)
            rows.append(row);(output/'results.json').write_text(json.dumps(dict(status='provisional',results=rows),indent=2),encoding='utf8')


def increment_cases(record,dates):
    for date in dates:
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


def run_generalized(args):
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for date in args.dates:
        for nx in args.grids:
            start=time.perf_counter()
            try:
                result=solve_generalized_grid(load_inputs(date,.1),nx=nx,eta_s=args.eta_s,
                    fit_priority=args.fit_priority,core_loss_tolerance=args.core_loss_tolerance)
                frame=result['per_expiry'];frame.to_csv(out/f'{date}_{nx}.csv',index=False)
                arrays={'states':result['states']}
                for j,e in enumerate(result['exdates']):
                    arrays[f'q_{j}']=result['q_by_expiry'][e]
                    arrays[f'density_{j}']=result['density_by_expiry'][e]
                    arrays[f'increment_{j}']=result['increments_by_expiry'][e]
                np.savez_compressed(out/f'{date}_{nx}.npz',**arrays)
                r=dict(date=date,nx=nx,success=True,expiries=len(frame),fit_priority=args.fit_priority,
                    fit_target_met=result['fit_target_met'],
                    coverage=float(frame.n_inside.sum()/frame.n_quotes.sum()),
                    core_coverage=float(frame.n_core_inside.sum()/frame.n_core.sum()),
                    min_coverage=float(frame.coverage.min()),min_core_coverage=float(frame.core_coverage.min()),
                    max_mass_error=float(frame.mass_error.max()),max_mean_error=float(frame.mean_error.max()),
                    max_reprice_error=float(frame.reprice_error.max()),calendar_min=float(frame.calendar_min.min()))
            except Exception as exc:r=dict(date=date,nx=nx,success=False,fit_priority=args.fit_priority,error=str(exc))
            r['seconds']=time.perf_counter()-start;summaries.append(r);print(json.dumps(r),flush=True)
            (out/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf8')
