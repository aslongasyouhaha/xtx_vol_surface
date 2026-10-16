"""Validate interpolated surfaces against saved paper-baseline fits."""
import _bootstrap  # project paths for direct script execution
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sanos_surface import SanosSurface,audit_surface

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir',type=Path,default=ROOT/'data/outputs'/'baseline_paper_gap001')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'data/outputs'/'surface_validation_gap001')
    args=parser.parse_args()
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    results=[]
    for date in ['2025-01-02','2025-04-01','2025-06-02']:
        base=args.baseline_dir/date
        quotes=pd.read_csv(base/'quotes.csv')
        for mode in ['atm_pchip','linear']:
            surface=SanosSurface.from_directory(base,time_interpolation=mode)
            report=audit_surface(surface)
            fit=surface.pure_call_days(quotes.dte.to_numpy(),quotes.k.to_numpy())
            error=float(np.max(abs(fit-quotes.fit.to_numpy())))
            report.update(date=date,max_saved_fit_difference=error)
            assert report['passed'] and error<1e-12,report
            # Check every interior whole day in addition to interval subdivisions.
            days=np.unique(np.r_[surface.times*365,np.arange(np.ceil(surface.times[0]*365),np.floor(surface.times[-1]*365)+1)])
            k=np.linspace(.5,2.,301)
            values=surface.pure_call_days(days[:,None],k[None,:])
            violation=max(0.,float(np.max(-np.diff(values,axis=0))))
            report['daily_grid_calendar_violation']=violation
            assert violation<=2e-7,report
            results.append(report);print(json.dumps(report),flush=True)
            np.savez(output/f'{date}_{mode}.npz',dte=days,k=k,pure_call=values)
    (output/'results.json').write_text(json.dumps(results,indent=2),encoding='utf8')


if __name__=='__main__':main()
