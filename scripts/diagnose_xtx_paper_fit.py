"""Controlled paper-LP diagnostics; ETA variants are not production selection."""
import _bootstrap  # project paths for direct script execution
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from xtx_inputs import load_inputs,core_mask
from sanos_core import solve_convex_order_lp,call_kernel
from sanos_working_set import mean_one_minimum

ROOT=Path(__file__).resolve().parents[1]


def metrics(sol):
    row=dict(success=sol['success'],solve_seconds=sol['solve_seconds'])
    if not sol['success']:return row
    inside=core_inside=total=core_total=large=0
    for ex,v in sol['inputs'].items():
        fit=sol['fit_by_expiry'][ex];bid,ask=v['pure_bid'],v['pure_ask'];mask=core_mask(v['quotes'])
        outside=np.maximum(bid-fit,0)+np.maximum(fit-ask,0)
        inside+=int((outside<=1e-9).sum());total+=len(fit)
        core_inside+=int((outside[mask]<=1e-9).sum());core_total+=int(mask.sum())
        large+=int((outside[mask]>.1*(ask-bid)[mask]).sum())
    row.update(coverage=inside/total,core_coverage=core_inside/core_total,core_misses=core_total-core_inside,
               core_outside_gt_01_spread=large,actual_objective=sol['actual_objective'])
    return row


def main():
    out=ROOT/'data/outputs'/'paper_fit_diagnosis';out.mkdir(parents=True,exist_ok=True)
    inputs=load_inputs('2025-04-01',.25,baseline_variant='baseline_paper');records=[];point_frames=[];singles=[]
    for ex,v in inputs.items():
        G=call_kernel(v['anchor_k'],v['market_k'],v['smooth_variance'])
        minimum=np.array([mean_one_minimum(v['anchor_k'],r) for r in G])
        maximum=-np.array([mean_one_minimum(v['anchor_k'],-r) for r in G])
        g=v['quotes'].copy();g['attainable_min']=minimum;g['attainable_max']=maximum
        g['individually_impossible']=(minimum>v['pure_ask']+1e-9)|(maximum<v['pure_bid']-1e-9)
        g['core']=core_mask(g);point_frames.append(g)
        sol=solve_convex_order_lp({ex:v},time_limit=15.)
        row=dict(exdate=str(ex.date()),dte=v['dte'],**metrics(sol));singles.append(row)
    pd.concat(point_frames).to_csv(out/'pointwise_reachability.csv',index=False)
    pd.DataFrame(singles).to_csv(out/'independent_expiries.csv',index=False)
    print('SINGLE_EXPIRIES '+json.dumps(singles),flush=True)
    for eta in [.25,.1,0.]:
        items=copy.deepcopy(inputs)
        for v in items.values():v['smooth_variance']=eta*float(v['quotes'].atm_total_variance.iloc[0])
        sol=solve_convex_order_lp(items,time_limit=30.)
        row=dict(eta=eta,**metrics(sol));records.append(row)
        print(json.dumps(row),flush=True)
        if sol['success']:sol['per_expiry'].to_csv(out/f'eta_{eta}_expiries.csv',index=False)
        (out/'results.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    dense=copy.deepcopy(inputs)
    for v in dense.values():
        a=v['anchor_k'];v['anchor_k']=np.unique(np.r_[a,(a[:-1]+a[1:])/2])
    dense_singles=[]
    for ex,v in dense.items():
        sol=solve_convex_order_lp({ex:v},time_limit=15.)
        dense_singles.append(dict(exdate=str(ex.date()),dte=v['dte'],**metrics(sol)))
    pd.DataFrame(dense_singles).to_csv(out/'dense_independent_expiries.csv',index=False)
    sol=solve_convex_order_lp(dense,time_limit=30.)
    row=dict(eta=.25,anchor_grid='one_midpoint_between_each_original_pair',**metrics(sol))
    records.append(row);print(json.dumps(row),flush=True)
    (out/'results.json').write_text(json.dumps(records,indent=2),encoding='utf8')


if __name__=='__main__':main()
