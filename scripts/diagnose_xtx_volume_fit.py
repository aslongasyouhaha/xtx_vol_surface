"""Separate quote filtering from anchor-support and calendar effects."""
import _bootstrap  # project paths for direct script execution
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
from xtx_inputs import load_inputs, core_mask
from sanos_core import solve_convex_order_lp

ROOT=Path(__file__).resolve().parents[1]


def main():
    date='2025-01-02';inputs=load_inputs(date,.25,baseline_variant='baseline_volume100_filtered_anchors')
    old=np.load(ROOT/'data/outputs'/'baseline'/date/'model.npz')
    out=ROOT/'data/outputs'/'volume_fit_diagnosis';out.mkdir(parents=True,exist_ok=True)
    records=[]
    scenarios={'filtered_grid':copy.deepcopy(inputs),'old_boundaries':copy.deepcopy(inputs),
               'old_grids':copy.deepcopy(inputs)}
    for ex,v in scenarios['old_boundaries'].items():
        old_a=old[str(ex.date())+'__anchors'];v['anchor_k']=np.unique(np.r_[v['anchor_k'],old_a[0],old_a[-1]])
    for ex,v in scenarios['old_grids'].items():v['anchor_k']=old[str(ex.date())+'__anchors']
    for name,items in scenarios.items():
        sol=solve_convex_order_lp(items,time_limit=30.,mid_penalty=0.,weight_mode='legacy')
        row=dict(scenario=name,success=sol['success'],solve_seconds=sol['solve_seconds'])
        if sol['success']:
            frames=[]
            for ex,v in sol['inputs'].items():
                g=v['quotes'].copy();fit=sol['fit_by_expiry'][ex]
                g['outside']=np.maximum(v['pure_bid']-fit,0)+np.maximum(fit-v['pure_ask'],0)
                g['core']=core_mask(g);frames.append(g)
            q=pd.concat(frames);q['inside']=q.outside.le(1e-9)
            row.update(coverage=float(q.inside.mean()),core_coverage=float(q.loc[q.core,'inside'].mean()),objective=sol['objective'])
            q.to_csv(out/(name+'_quotes.csv'),index=False)
            sol['per_expiry'].to_csv(out/(name+'_expiries.csv'),index=False)
        records.append(row);print(json.dumps(row),flush=True)
        (out/'results.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    singles=[]
    for ex,v in inputs.items():
        sol=solve_convex_order_lp({ex:v},time_limit=10.,mid_penalty=0.,weight_mode='legacy')
        row=dict(exdate=str(ex.date()),dte=v['dte'],success=sol['success'])
        if sol['success']:
            fit=sol['fit_by_expiry'][ex];mask=core_mask(v['quotes'])
            outside=np.maximum(v['pure_bid']-fit,0)+np.maximum(fit-v['pure_ask'],0)
            row.update(coverage=float(np.mean(outside<=1e-9)),core_coverage=float(np.mean(outside[mask]<=1e-9)))
        singles.append(row)
    pd.DataFrame(singles).to_csv(out/'independent_expiries.csv',index=False)


if __name__=='__main__':main()
