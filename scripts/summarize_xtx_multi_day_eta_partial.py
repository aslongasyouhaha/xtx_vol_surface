"""Summarize joint-LP eta sensitivity without heuristic parameter selection."""
import _bootstrap
import argparse,json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('data/outputs/eta_sensitivity/results.json'))
    args=p.parse_args();rows=json.loads(args.input.read_text(encoding='utf8'))
    frame=pd.DataFrame([{k:v for k,v in r.items() if k!='surface_audit'} for r in rows])
    if frame.empty:p.error('No sensitivity results')
    if not frame.method.eq('joint_paper_lp_fixed_grid').all():p.error('Input must come from the shared joint LP sensitivity runner')
    frame=frame.sort_values(['date','eta']);frame.to_csv(args.input.parent/'summary.csv',index=False)
    print(frame.to_string(index=False))
if __name__=='__main__':main()
