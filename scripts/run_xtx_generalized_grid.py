"""Validated conservative-grid generalized model; experimental, not default LP."""
import _bootstrap  # project paths for direct script execution
import json
from pathlib import Path
import numpy as np
from xtx_generalized_research import solve_generalized_grid






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
