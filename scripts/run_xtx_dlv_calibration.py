"""Daily market calibration of nonnegative DLVs; nonlinear local experiments."""
import _bootstrap  # project paths for direct script execution
import argparse
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
DEPS=ROOT/'data/outputs'/'native_deps'
if (DEPS/'highspy').exists():sys.path.insert(0,str(DEPS))
from xtx_inputs import load_inputs,core_mask
from sanos_dlv_calibration import DLVCalibrator,calibration_grid
from sanos_core import solve_convex_order_lp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dates',nargs='+',default=['2025-01-02','2025-04-01','2025-06-02'])
    p.add_argument('--time-limit',type=float,default=30.)
    p.add_argument('--max-iterations',type=int,default=1500)
    p.add_argument('--initialization',choices=['lp_interior','atm'],default='lp_interior')
    p.add_argument('--seed-loss-allowance',type=float,default=.1)
    p.add_argument('--central-points',type=int,default=121)
    p.add_argument('--wing-points',type=int,default=31)
    p.add_argument('--output-dir',type=Path,default=ROOT/'data/outputs'/'dlv_calibration')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);rows=[]
    with threadpool_limits(limits=1):
        for date in args.dates:
            eta=.1 if date=='2025-04-01' else .25
            inputs=load_inputs(date,eta)
            grid=calibration_grid(inputs,central_points=args.central_points,wing_points=args.wing_points)
            c=DLVCalibrator(inputs,grid)
            print(json.dumps(dict(date=date,event='start',eta=eta,states=len(grid),parameters=int(np.prod(c.shape)))),flush=True)
            started=time.perf_counter();initial=None;initialization=dict(method='atm')
            if args.initialization=='lp_interior':
                try:
                    initial,initialization=c.interior_lp_initialization(loss_allowance=args.seed_loss_allowance,
                        time_limit=min(20.,args.time_limit*.6))
                    initial=c.rescale_parameters(initial)
                except ValueError as error:
                    initialization=dict(method='atm_fallback',lp_seed_error=str(error),seconds=time.perf_counter()-started)
            remaining=max(.001,args.time_limit-(time.perf_counter()-started))
            result=c.calibrate(time_limit=remaining,max_iterations=args.max_iterations,initial_parameters=initial)
            calibration_seconds=time.perf_counter()-started
            out=args.output_dir/date;out.mkdir(parents=True,exist_ok=True)
            frames=[]
            for (ex,v),fit in zip(c.items.items(),result['fits']):
                g=v['quotes'][['exdate','dte','T','k','source_cp_flag','volume','pure_bid','pure_mid','pure_ask','source_otm_iv']].copy()
                g['fit']=fit;g['outside']=np.maximum(g.pure_bid-fit,0)+np.maximum(fit-g.pure_ask,0)
                g['outside_spreads']=g.outside/(g.pure_ask-g.pure_bid);g['inside']=g.outside.le(1e-9)
                g['core']=core_mask(v['quotes']);frames.append(g)
            quotes=pd.concat(frames,ignore_index=True);quotes.to_csv(out/'quotes.csv',index=False)
            per=quotes.groupby(['exdate','dte']).agg(n_quotes=('inside','size'),coverage=('inside','mean'),
                                                   max_outside_spreads=('outside_spreads','max')).reset_index()
            per.to_csv(out/'per_expiry.csv',index=False)
            model=result['model']
            np.savez(out/'model.npz',anchors=model.anchors,times=model.times,smoothing_variances=model.smoothing_variances,
                     local_volatilities=c.volatilities(result['parameters']),
                     weights=model.weights,initial_weights=model.initial_weights)
            control=copy.deepcopy(inputs)
            for v in control.values():v['anchor_k']=grid.copy()
            lp=solve_convex_order_lp(control,time_limit=30.)
            row={k:v for k,v in result.items() if k not in {'model','parameters','fits'}}
            row.update(initialization=initialization,calibration_seconds=calibration_seconds)
            row.update(date=date,eta=eta,states=len(grid),parameters=int(np.prod(c.shape)),quotes=len(quotes),
                       core_quotes=int(quotes.core.sum()),core_coverage=float(quotes.loc[quotes.core,'inside'].mean()),
                       core_outside_gt_01_spread=int((quotes.core & quotes.outside_spreads.gt(.1)).sum()),
                       max_local_volatility=float(c.volatilities(result['parameters']).max()),
                       same_grid_lp_success=lp['success'],same_grid_lp_seconds=lp['solve_seconds'])
            if lp['success']:
                inside=total=0
                for ex,v in lp['inputs'].items():
                    mask=core_mask(v['quotes']);fit=lp['fit_by_expiry'][ex]
                    outside=np.maximum(v['pure_bid']-fit,0)+np.maximum(fit-v['pure_ask'],0)
                    total+=int(mask.sum());inside+=int((outside[mask]<=1e-9).sum())
                row.update(same_grid_lp_core_coverage=inside/total,same_grid_lp_objective=lp['actual_objective'])
            (out/'summary.json').write_text(json.dumps(row,indent=2),encoding='utf8')
            rows.append(row);(args.output_dir/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
            pd.DataFrame([{k:v for k,v in r.items() if k not in {'attempts','audit','initialization'}} for r in rows]).to_csv(args.output_dir/'summary.csv',index=False)
            print(json.dumps(row),flush=True)


if __name__=='__main__':main()
