"""Diagnostics on the same input selection and models as the main pipeline."""
import numpy as np
import pandas as pd
from sanos_core import solve_convex_order_lp
from sanos_surface import SanosSurface,audit_surface
from sanos_generalized import ExactProduct
from sanos_generalized_grid import GridProduct
from xtx_inputs import core_mask


def market_calendar(quotes,*,points=301,tolerance=1e-9):
    """Adjacent-expiry interpolated quote compatibility, not an execution test.

    ask_next < bid_prev rules out ordered prices inside the interpolated bands.
    bid_next < ask_prev alone does not. Interpolation can introduce artifacts.
    """
    groups=[]
    for expiry,g in quotes.groupby('exdate',sort=True):
        g=g.groupby('k',as_index=False)[['pure_bid','pure_mid','pure_ask']].median().sort_values('k')
        if len(g)>=2:groups.append((expiry,g))
    rows=[];curves=[]
    for (prev,a),(nxt,b) in zip(groups[:-1],groups[1:]):
        lo=max(a.k.min(),b.k.min());hi=min(a.k.max(),b.k.max())
        if lo>=hi:
            rows.append(dict(previous=str(prev),next=str(nxt),status='no_common_strike_range'));continue
        k=np.linspace(lo,hi,points)
        interp=lambda f,c:np.interp(k,f.k,f[c])
        mid=interp(b,'pure_mid')-interp(a,'pure_mid')
        margin=interp(b,'pure_ask')-interp(a,'pure_bid')
        rows.append(dict(previous=str(prev),next=str(nxt),status='checked',min_k=float(lo),max_k=float(hi),
            min_mid_difference=float(mid.min()),min_compatible_order_margin=float(margin.min()),
            mid_inversion_points=int(np.sum(mid < -tolerance)),
            interpolated_band_conflict_points=int(np.sum(margin < -tolerance))))
        curves.append(dict(previous=str(prev),next=str(nxt),k=k,mid_difference=mid,compatible_order_margin=margin))
    return rows,curves


def eta_control(inputs,eta,*,time_limit=45):
    """Full joint LP on unchanged observations/states; no automatic eta choice."""
    if not np.isfinite(eta) or not 0<=eta<1:raise ValueError('eta must be in [0,1)')
    items={e:dict(v,smooth_variance=float(eta)*float(v['quotes'].atm_total_variance.iloc[0])) for e,v in inputs.items()}
    sol=solve_convex_order_lp(items,time_limit=time_limit)
    row=dict(eta=float(eta),success=bool(sol['success']),message=sol['message'],solve_seconds=sol['solve_seconds'])
    if not sol['success']:return row
    n=inside=nc=ic=0;outside_total=0.
    for e,v in sol['inputs'].items():
        fit=sol['fit_by_expiry'][e];outside=np.maximum(v['pure_bid']-fit,0)+np.maximum(fit-v['pure_ask'],0)
        core=core_mask(v['quotes']);n+=len(fit);inside+=int(np.sum(outside<=1e-9))
        nc+=int(core.sum());ic+=int(np.sum(outside[core]<=1e-9))
        outside_total+=float(np.sum(outside/(v['pure_ask']-v['pure_bid'])))
    audit=audit_surface(SanosSurface.from_solution(sol))
    row.update(success=bool(audit['passed']),quotes=n,expiries=len(items),coverage=inside/n,
        core_coverage=ic/nc if nc else None,outside_loss=outside_total,
        actual_objective=sol['actual_objective'],surface_audit=audit)
    return row


def generalized_accuracy(grids=(4097,16385),*,steps=3):
    """Identical multipliers/variances/weights: analytic reference vs grid model."""
    k=np.linspace(.6,1.4,201);rows=[]
    for nx in grids:
        model=GridProduct(nx);exact=ExactProduct();previous=model.calls(k)
        for step in range(steps):
            a=[.9,1,1.1];v=[.001,.002,.003];q=np.array([.2,.6,.2])
            fitted=model.kernel(a,v,k)@q;model.advance(a,v,q);exact.advance(a,v,q)
            prices=model.calls(k)
            rows.append(dict(nx=int(nx),step=step+1,max_price_error=float(np.max(abs(prices-exact.calls(k)))),
                reprice_error=float(np.max(abs(prices-fitted))),mass_error=float(abs(model.weights.sum()-1)),
                mean_error=float(abs(model.weights@model.states-1)),calendar_min=float(np.min(prices-previous))))
            previous=prices
    return rows
