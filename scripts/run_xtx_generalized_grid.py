"""Validated conservative-grid generalized model; experimental, not default LP."""
import _bootstrap  # project paths for direct script execution
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sanos_core import call_kernel
from sanos_generalized_grid import GridProduct


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
            from xtx_inputs import core_mask
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


def main():
    import time,argparse
    from xtx_inputs import load_inputs
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dates',nargs='+',default=['2025-01-02','2025-04-01','2025-06-02'])
    parser.add_argument('--grids',nargs='+',type=int,default=[4097,16385])
    parser.add_argument('--fit-priority',choices=['paper','core_first'],default='paper')
    parser.add_argument('--eta-s',type=float,default=.1)
    parser.add_argument('--core-loss-tolerance',type=float,default=1e-5,
        help='Numerical allowance in TOTAL spread-normalized core outside loss.')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    out=args.output or Path('data/outputs/generalized_grid' if args.fit_priority=='paper' else 'data/outputs/generalized_grid_core_first')
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


if __name__=='__main__':main()
