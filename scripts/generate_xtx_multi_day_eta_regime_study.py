"""Fixed-grid, full joint-LP eta sensitivity using shared production inputs."""
import _bootstrap
import argparse,json
from pathlib import Path
import pandas as pd
from threadpoolctl import threadpool_limits
from xtx_inputs import load_inputs
from xtx_research import eta_control

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dates',nargs='+',default=['2025-01-02','2025-04-01','2025-06-02'])
    p.add_argument('--etas',nargs='+',type=float,default=[.1,.25])
    p.add_argument('--time-limit',type=float,default=45.)
    p.add_argument('--output',type=Path,default=Path('data/outputs/eta_sensitivity'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);rows=[]
    if any(not 0<=eta<1 for eta in args.etas):p.error('Every eta must be in [0,1)')
    for date in args.dates:
        inputs=load_inputs(date,.25)
        for eta in args.etas:
            row=dict(date=date,method='joint_paper_lp_fixed_grid',**eta_control(inputs,eta,time_limit=args.time_limit))
            rows.append(row);print(json.dumps(row),flush=True)
            (args.output/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
            pd.DataFrame([{k:v for k,v in r.items() if k!='surface_audit'} for r in rows]).to_csv(args.output/'summary.csv',index=False)
    if not all(r['success'] for r in rows):raise SystemExit(1)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
