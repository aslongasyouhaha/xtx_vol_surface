"""Run state-dependent martingale transport with an explicit choice of control."""
import _bootstrap
import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits
from xtx_transport_research import run_joint,run_chain,run_fixed_prefix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['joint','chain','prefix'],
        help='Joint LP then couple marginals; sequential state-dependent fits; fixed generalized-prefix control.')
    p.add_argument('--dates',nargs='+')
    p.add_argument('--eta',type=float,
        help='Joint mode defaults to the recorded validation cases; other modes default to 0.1.')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.eta is not None and not 0<=args.eta<1:p.error('--eta must be in [0,1)')
    defaults={'joint':'joint_transport','chain':'transport_chain','prefix':'state_dependent_transport'}
    output=args.output or Path('data/outputs/generalized_grid')/defaults[args.mode]
    dates=args.dates or (['2025-04-01'] if args.mode=='prefix' else ['2025-01-02','2025-04-01','2025-06-02'])
    if args.mode=='joint':return run_joint(dates,output,eta=args.eta)
    eta=.1 if args.eta is None else args.eta
    if args.mode=='chain':return run_chain(dates,output,eta=eta)
    for date in dates:run_fixed_prefix(date,output/date,eta=eta)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
