"""Validate the maintained generalized model against its exact path reference."""
import _bootstrap
import argparse,json
from pathlib import Path
from threadpoolctl import threadpool_limits
from xtx_research import generalized_accuracy

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grids',nargs='+',type=int,default=[4097,16385])
    p.add_argument('--output',type=Path,default=Path('data/outputs/generalized_accuracy'))
    args=p.parse_args();rows=generalized_accuracy(args.grids)
    passed=all(r['mass_error']<=1e-10 and r['mean_error']<=1e-10 and r['reprice_error']<=1e-12 and r['calendar_min']>=-1e-10 for r in rows)
    passed=passed and all(r['max_price_error']<=2e-5 for r in rows if r['nx']==max(args.grids))
    report=dict(passed=bool(passed),reference='ExactProduct, identical fixed parameters',rows=rows)
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))
    if not passed:raise SystemExit(1)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
