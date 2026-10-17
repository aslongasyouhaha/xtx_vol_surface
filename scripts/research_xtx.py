"""Run controlled SANOS research through one command with named experiments."""
import _bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
from threadpoolctl import threadpool_limits
from xtx_io import load_inputs,TESTCASES
from xtx_research import market_calendar,eta_control,generalized_accuracy,run_wing_scan,run_wing_boundary,run_testcases

ROOT=Path(__file__).resolve().parents[1]
OUTPUTS=ROOT/'data/outputs'
DATES=['2025-01-02','2025-04-01','2025-06-02']
APRIL=['2025-04-01','2025-04-02']


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='experiment',required=True)
    def command(name,help,output):
        c=commands.add_parser(name,help=help)
        c.add_argument('--output',type=Path,default=OUTPUTS/output if output else None)
        return c
    c=command('plot','Interactive IV comparison of saved calibration observations','calibration_quotes/2025-06-02.html')
    c.add_argument('--model-dir',type=Path,default=OUTPUTS/'daily_validation/june/2025-06-02/lp')
    c=command('generalized','Calibrate independent product increments on a conservative grid',None)
    c.add_argument('--dates',nargs='+',default=DATES)
    c.add_argument('--grids',nargs='+',type=int,default=[4097,16385])
    c.add_argument('--fit-priority',choices=['paper','core_first'],default='paper')
    c.add_argument('--eta-s',type=float,default=.1)
    c.add_argument('--core-loss-tolerance',type=float,default=1e-5,
        help='Numerical allowance in TOTAL spread-normalized core outside loss.')
    c=command('transport','State-dependent martingale transport controls',None)
    c.add_argument('mode',choices=['joint','chain','prefix'])
    c.add_argument('--dates',nargs='+')
    c.add_argument('--eta',type=float,
        help='Joint: recorded validation eta; chain/prefix: 0.1 unless specified.')
    c=command('dlv-recovery','Validate DLV generation, recovery and saved LP encoding','dlv_validation')
    c.add_argument('--dates',nargs='+',default=DATES)
    c.add_argument('--synthetic-only',action='store_true')
    c.add_argument('--baseline-dir',type=Path,default=OUTPUTS/'baseline_paper_gap001')
    c=command('dlv-calibrate','Nonlinear calibration of nonnegative DLV parameters','dlv_calibration')
    c.add_argument('--dates',nargs='+',default=DATES)
    c.add_argument('--time-limit',type=float,default=30.)
    c.add_argument('--max-iterations',type=int,default=1500)
    c.add_argument('--initialization',choices=['lp_interior','atm'],default='lp_interior')
    c.add_argument('--seed-loss-allowance',type=float,default=.1)
    c.add_argument('--central-points',type=int,default=121)
    c.add_argument('--wing-points',type=int,default=31)
    c=command('daily-validation','Consecutive observation dates: joint LP and full-state QP controls','daily_validation')
    c.add_argument('--blocks',nargs='+',choices=['january','april','june'],default=['january','april','june'])
    c.add_argument('--generalized',action='store_true')
    c.add_argument('--qp-seconds',type=float,default=12.)
    c=command('daily-update','Sequential daily QP using previously accepted fits','daily_validation/active_support_qp')
    c.add_argument('--blocks',nargs='+',choices=['january','april','june'],default=['january','april','june'])
    c.add_argument('--baseline-dir',type=Path,default=OUTPUTS/'daily_validation')
    c.add_argument('--loss-allowance',type=float,default=.01)
    c.add_argument('--time-limit',type=float,default=20.)
    c=command('benchmark','Fixed-geometry paper LP updates or Python/C++ comparison',None)
    c.add_argument('--mode',choices=['paper','language'],default='paper')
    c.add_argument('--dates',nargs='+')
    c.add_argument('--eta',type=float,default=.25,help='Paper mode smoothing coefficient.')
    c.add_argument('--time-limit',type=float,default=60.)
    c.add_argument('--scenario',choices=['broad','sparse'],default='broad',help='Language mode quote updates.')
    c.add_argument('--skip-scipy',action='store_true')
    c.add_argument('--cpp-exe',type=Path,default=OUTPUTS/'native_deps/incremental_highs.exe')
    c.add_argument('--dll',type=Path,default=OUTPUTS/'native_deps/highs-1.15.1/bin/highs.dll')
    c=command('testcases','Run public synthetic fixtures without licensed market data','testcases')
    c.add_argument('--cases',nargs='+',choices=TESTCASES,default=list(TESTCASES))
    c.add_argument('--data-dir',type=Path,default=ROOT/'data/testcases')
    c=command('calendar','Adjacent-expiry market quote compatibility','market_calendar')
    c.add_argument('--date',default='2025-06-02')
    c=command('eta','Full joint LP on fixed observations and states','eta_sensitivity')
    c.add_argument('--dates',nargs='+',default=DATES)
    c.add_argument('--etas',nargs='+',type=float,default=[.1,.25])
    c.add_argument('--time-limit',type=float,default=45.)
    c=command('wings','Scan saved LP slices for wing reversals','wing_turns')
    c.add_argument('--baseline-dir',type=Path,default=OUTPUTS/'daily_validation')
    command('wing-boundary','Fixed-quote shared-left-state comparison','wing_boundary')
    c=command('surface','Audit saved fits and interpolated surfaces','surface_validation_gap001')
    c.add_argument('--baseline-dir',type=Path,default=OUTPUTS/'baseline_paper_gap001')
    c.add_argument('--dates',nargs='+',default=DATES)
    c=command('accuracy','Conservative grid against exact finite paths','generalized_accuracy')
    c.add_argument('--grids',nargs='+',type=int,default=[4097,16385])
    c=command('refine-grid','Reprice fixed saved parameters on a finer grid','generalized_grid')
    c.add_argument('--model-dir',type=Path,default=OUTPUTS/'generalized_grid')
    c.add_argument('--dates',nargs='+',default=DATES)
    c.add_argument('--grid',type=int,default=65537)
    c=command('core-priority','Compare paper and core-first generalized objectives','research_exploration/core_priority')
    c.add_argument('--dates',nargs='+',default=[*APRIL,'2025-06-03'])
    c=command('fit-failure','Attribute generalized fit errors and check dual bounds','research_exploration/failure_attribution')
    c.add_argument('--dates',nargs='+',default=APRIL)
    c=command('increment-flexibility','Controlled independent-increment dictionaries','research_exploration/flexibility')
    c.add_argument('--dates',nargs='+',default=APRIL)
    c=command('prefix-lookback','Two-expiry adjustment with earlier quote protection','research_exploration/prefix_lookback')
    c.add_argument('--dates',nargs='+',default=APRIL)
    c.add_argument('--iterations',type=int,default=5)
    c=command('daily-qp','Pointwise fit-preserving QP; isolated prior-LP references','research_exploration/fit_preserving_qp')
    c.add_argument('--blocks',nargs='+',choices=['january','april','june'],default=['april','june'])
    c.add_argument('--coordinates',choices=['absolute','delta','moments'],default='absolute')
    c.add_argument('--time-limit',type=float,default=20.)
    return p


def calendar(args):
    import matplotlib.pyplot as plt
    items=load_inputs(args.date,.25)
    quotes=pd.concat([v['quotes'] for v in items.values()],ignore_index=True)
    rows,curves=market_calendar(quotes);out=args.output/args.date;out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'pairs.csv',index=False)
    metadata=dict(date=args.date,expiries=len(items),pairs=len(curves),
        interpretation='Interpolated quote-band compatibility; not executable arbitrage.')
    (out/'summary.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    if curves:
        fig,axs=plt.subplots(len(curves),1,figsize=(10,2.5*len(curves)),squeeze=False)
        for ax,c in zip(axs.flat,curves):
            ax.plot(c['k'],c['mid_difference'],label='mid_next - mid_previous')
            ax.plot(c['k'],c['compatible_order_margin'],label='ask_next - bid_previous')
            ax.axhline(0,color='gray',ls='--');ax.set_title(c['previous']+' -> '+c['next']);ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/'calendar.png',dpi=120);plt.close(fig)
    print(json.dumps(metadata))


def eta(args):
    if any(not np.isfinite(x) or not 0<=x<1 for x in args.etas):
        raise ValueError('Every eta must be finite and in [0,1)')
    rows=[]
    for date in args.dates:
        inputs=load_inputs(date,.25)
        for value in args.etas:
            row=dict(date=date,method='joint_paper_lp_fixed_grid',**eta_control(inputs,value,time_limit=args.time_limit))
            rows.append(row);print(json.dumps(row),flush=True)
            (args.output/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
            pd.DataFrame([{k:v for k,v in r.items() if k!='surface_audit'} for r in rows]).to_csv(args.output/'summary.csv',index=False)
    if not all(r['success'] for r in rows):raise SystemExit(1)


def surface(args):
    from sanos_surface import SanosSurface,audit_surface
    rows=[]
    for date in args.dates:
        base=args.baseline_dir/date;quotes=pd.read_csv(base/'quotes.csv')
        for mode in ['atm_pchip','linear']:
            model=SanosSurface.from_directory(base,time_interpolation=mode);report=audit_surface(model)
            fit=model.pure_call_days(quotes.dte.to_numpy(),quotes.k.to_numpy())
            error=float(np.max(abs(fit-quotes.fit.to_numpy())));report.update(date=date,max_saved_fit_difference=error)
            if not report['passed'] or error>=1e-12:raise ValueError(report)
            days=np.unique(np.r_[model.times*365,np.arange(np.ceil(model.times[0]*365),np.floor(model.times[-1]*365)+1)])
            k=np.linspace(.5,2.,301);values=model.pure_call_days(days[:,None],k[None,:])
            report['daily_grid_calendar_violation']=max(0.,float(np.max(-np.diff(values,axis=0))))
            if report['daily_grid_calendar_violation']>2e-7:raise ValueError(report)
            rows.append(report);print(json.dumps(report),flush=True)
            np.savez(args.output/f'{date}_{mode}.npz',dte=days,k=k,pure_call=values)
    (args.output/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')


def accuracy(args):
    rows=generalized_accuracy(args.grids)
    passed=all(r['mass_error']<=1e-10 and r['mean_error']<=1e-10 and r['reprice_error']<=1e-12
        and r['calendar_min']>=-1e-10 for r in rows)
    passed=passed and all(r['max_price_error']<=2e-5 for r in rows if r['nx']==max(args.grids))
    report=dict(passed=bool(passed),reference='ExactProduct, identical fixed parameters',rows=rows)
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))
    if not passed:raise SystemExit(1)


def refine_grid(args):
    from sanos_generalized import GridProduct
    rows=[]
    for date in args.dates:
        with np.load(args.model_dir/f'{date}_16385.npz') as saved:
            model=GridProduct(args.grid);items=load_inputs(date,.1);expiries=[]
            for j,it in enumerate(items.values()):
                model.advance(it['anchor_k'],saved[f'increment_{j}'],saved[f'q_{j}'])
                original=GridProduct(16385);original.weights=saved[f'density_{j}']
                difference=abs(model.calls(it['market_k'])-original.calls(it['market_k']))
                expiries.append(dict(dte=int(it['dte']),max_price_difference=float(difference.max()),
                    max_difference_spreads=float(np.max(difference/(it['pure_ask']-it['pure_bid'])))))
            rows.append(dict(date=date,comparison=f'fixed q: 16385 versus {args.grid} grid points',rows=expiries))
    (args.output/'accuracy_controls.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    print(json.dumps(rows,indent=2))


def main():
    matplotlib.use('Agg')
    p=parser();args=p.parse_args()
    if args.experiment=='plot':
        from xtx_plotting import run_plot
        return run_plot(args)
    if args.experiment=='generalized' and args.output is None:
        args.output=OUTPUTS/('generalized_grid' if args.fit_priority=='paper' else 'generalized_grid_core_first')
    if args.experiment=='transport':
        if args.eta is not None and not 0<=args.eta<1:p.error('--eta must be in [0,1)')
        defaults={'joint':'joint_transport','chain':'transport_chain','prefix':'state_dependent_transport'}
        args.output=args.output or OUTPUTS/'generalized_grid'/defaults[args.mode]
        args.dates=args.dates or (['2025-04-01'] if args.mode=='prefix' else DATES)
    if args.experiment=='benchmark':
        args.dates=args.dates or (DATES if args.mode=='paper' else ['2025-04-01','2025-06-02'])
        directory='paper_incremental' if args.mode=='paper' else 'incremental_benchmark'+('_sparse' if args.scenario=='sparse' else '')
        args.output=args.output or OUTPUTS/directory
    args.output.mkdir(parents=True,exist_ok=True)
    if args.experiment=='generalized':
        from xtx_generalized_research import run_generalized
        return run_generalized(args)
    if args.experiment=='transport':
        from xtx_transport_research import run_joint,run_chain,run_fixed_prefix
        if args.mode=='joint':return run_joint(args.dates,args.output,eta=args.eta)
        eta=.1 if args.eta is None else args.eta
        if args.mode=='chain':return run_chain(args.dates,args.output,eta=eta)
        for date in args.dates:run_fixed_prefix(date,args.output/date,eta=eta)
        return
    if args.experiment=='dlv-recovery':
        from xtx_research import run_dlv_recovery
        return run_dlv_recovery(args)
    if args.experiment=='dlv-calibrate':
        from xtx_research import run_dlv_calibration
        return run_dlv_calibration(args)
    if args.experiment=='daily-validation':
        from xtx_daily_research import run_daily_validation
        return run_daily_validation(args)
    if args.experiment=='daily-update':
        from xtx_daily_research import run_daily_update
        return run_daily_update(args)
    if args.experiment=='benchmark':
        from sanos_solver_research import run_paper_benchmark,run_language_benchmark
        return run_paper_benchmark(args) if args.mode=='paper' else run_language_benchmark(args)
    if args.experiment=='testcases':return run_testcases(args.cases,args.output,args.data_dir)
    basic={'calendar':calendar,'eta':eta,'surface':surface,'accuracy':accuracy,'refine-grid':refine_grid}
    if args.experiment in basic:return basic[args.experiment](args)
    if args.experiment=='wings':return run_wing_scan(args.output,args.baseline_dir)
    if args.experiment=='wing-boundary':return run_wing_boundary(args.output)
    if args.experiment=='daily-qp':
        from xtx_daily_research import run_fit_preserving
        return run_fit_preserving(args.blocks,args.output,time_limit=args.time_limit,coordinates=args.coordinates)
    from xtx_generalized_research import run_core_priority,run_failure_cases,increment_cases,run_lookback
    if args.experiment=='core-priority':return run_core_priority(args.dates,args.output)
    if args.experiment=='fit-failure':return run_failure_cases(args.dates,args.output)
    if args.experiment=='prefix-lookback':return run_lookback(args.dates,args.output,iterations=args.iterations)
    rows=[]
    def record(row):
        rows.append(row)
        (args.output/'results.json').write_text(json.dumps(dict(status='provisional',results=rows),indent=2),encoding='utf8')
        pd.DataFrame(rows).to_csv(args.output/'results.csv',index=False);print(json.dumps(row),flush=True)
    return increment_cases(record,args.dates)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
