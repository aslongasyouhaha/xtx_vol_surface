"""Consecutive observation dates: paper LP, rolled prior reference QP, optional GNR.

Within-day calendar arbitrage is constrained; cross-day price monotonicity is
NOT imposed. Geometry is rebuilt each day. Prior information never uses a
future observation and old bases are not reused across changed dimensions.
"""
import _bootstrap  # project paths for direct script execution
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
DEPS=ROOT/'data/outputs/native_deps'
if (DEPS/'highspy').exists():sys.path.insert(0,str(DEPS))
from xtx_inputs import load_inputs,core_mask
from sanos_incremental import PersistentSanosLP
from sanos_daily import snapshot,daily_references,seed_previous_marginals,stability_qp
from sanos_surface import SanosSurface,audit_surface

BLOCKS={'january':(.25,['2025-01-02','2025-01-03','2025-01-06']),
        'april':(.1,['2025-04-01','2025-04-02','2025-04-03']),
        'june':(.25,['2025-06-02','2025-06-03','2025-06-04'])}


def save_fit(directory,date,p,x,metadata):
    directory.mkdir(parents=True,exist_ok=True);arrays={};frames=[]
    for e,v in p['items'].items():
        prefix=str(e.date());q=x[slice(*p['q_offsets'][e])]
        g=v['quotes'].copy();fit=v['G']@q
        g['fit']=fit;g['outside']=np.maximum(g.pure_bid-fit,0)+np.maximum(fit-g.pure_ask,0)
        g['inside']=g.outside<=1e-9;g['core']=core_mask(g)
        g['outside_spreads']=g.outside/(g.pure_ask-g.pure_bid);frames.append(g)
        arrays[prefix+'__q']=q;arrays[prefix+'__anchors']=v['anchor_k']
        arrays[prefix+'__smooth_variance']=np.asarray(v['smooth_variance'])
        arrays[prefix+'__dte']=np.asarray(v['dte']);arrays[prefix+'__T']=np.asarray(v.get('T',v['dte']/365))
    f=pd.concat(frames,ignore_index=True);core=f[f.core]
    surface=SanosSurface([float(v.get('T',v['dte']/365)) for v in p['items'].values()],
        [v['anchor_k'] for v in p['items'].values()],[x[slice(*p['q_offsets'][e])] for e in p['items']],
        [v['smooth_variance'] for v in p['items'].values()])
    audit=audit_surface(surface)
    row=dict(date=date,success=audit['passed'],quotes=len(f),expiries=len(p['items']),
        coverage=float(f.inside.mean()),core_coverage=float(core.inside.mean()),
        core_misses=int((~core.inside).sum()),outside_loss=float(f.outside_spreads.sum()),surface_audit=audit,**metadata)
    f.to_csv(directory/'quotes.csv',index=False);np.savez_compressed(directory/'model.npz',**arrays)
    (directory/'summary.json').write_text(json.dumps(row,indent=2),encoding='utf8')
    if not audit['passed']:raise ValueError('Saved fit failed independent surface audit')
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks',nargs='+',choices=list(BLOCKS),default=list(BLOCKS))
    parser.add_argument('--generalized',action='store_true')
    parser.add_argument('--qp-seconds',type=float,default=12.)
    parser.add_argument('--output',type=Path,default=ROOT/'data/outputs/daily_validation')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);rows=[]
    for block in args.blocks:
        eta,dates=BLOCKS[block];prior_lp=None;prior_deployed=None
        for j,date in enumerate(dates):
            start=time.perf_counter();print(f'BUILD {block} {date} eta={eta}',flush=True)
            try:
                inputs=load_inputs(date,eta);session=PersistentSanosLP(inputs,time_limit=45)
                seed=seed_previous_marginals(session,prior_lp,date) if prior_lp else dict(expiries_seeded=0,old_basis_reused=False)
                result=session.solve()
                if not result['success']:raise ValueError(result['message'])
                p=session.problem;x=result['x'];day=args.output/block/date
                lpmeta=dict(block=block,eta=eta,method='paper_lp',data_timing='end_of_day',
                    source_date=prior_lp['date'] if prior_lp else None,seed=seed,
                    solve_seconds=result['request_seconds'],max_constraint_violation=result['max_constraint_violation'])
                if j==1:
                    cold=PersistentSanosLP(inputs,time_limit=45);check=cold.solve()
                    lpmeta['cold_control_success']=check['success']
                    if check['success']:
                        lpmeta['warm_cold_objective_difference']=abs(result['actual_objective']-check['actual_objective'])
                        lpmeta['warm_cold_agrees_1e6']=lpmeta['warm_cold_objective_difference']<=1e-6
                        if check['actual_objective']<result['actual_objective']:
                            p=cold.problem;x=check['x'];session=cold;result=check
                            lpmeta['selected_lower_loss_cold_candidate']=True
                lprow=save_fit(day/'lp',date,p,x,lpmeta);rows.append(lprow)
                current_lp=snapshot(date,p,x)
                if prior_deployed:
                    refs=daily_references(prior_deployed,p,date)
                    refsave={}
                    for e,(target,mask) in refs.items():
                        refsave[str(e.date())+'__target']=target;refsave[str(e.date())+'__mask']=mask
                    np.savez_compressed(day/'prior_references.npz',source_date=prior_deployed['date'],**refsave)
                    qp=stability_qp(p,x,session.quotes,refs,time_limit=args.qp_seconds)
                    qpmeta=dict(block=block,eta=eta,method='daily_stability_qp',source_date=prior_deployed['date'],
                        accepted=qp['success'],attempts=qp['attempts'],seconds=qp.get('seconds'),
                        reference_rule='same expiry, same forward moneyness, prior annualized IV rolled to current remaining time')
                    if qp['success']:
                        qpmeta.update(reference_quotes=qp['reference_quotes'],lp_reference_rms_spreads=qp['lp_movement_rms'],
                            qp_reference_rms_spreads=qp['qp_movement_rms'],loss_allowance=qp['loss_allowance'],
                            lp_outside_loss=qp['lp_outside_loss'])
                        qprow=save_fit(day/'qp',date,p,qp['x'],qpmeta);rows.append(qprow)
                        prior_deployed=snapshot(date,p,qp['x'])
                    else:
                        qpmeta.update(date=date,success=False,reason=qp['reason'],deployed_fallback='current_validated_LP')
                        (day/'qp_rejected.json').write_text(json.dumps(qpmeta,indent=2),encoding='utf8');rows.append(qpmeta)
                        prior_deployed=current_lp
                else:prior_deployed=current_lp
                prior_lp=current_lp
                if args.generalized:
                    from run_xtx_generalized_grid import solve_generalized_grid
                    try:
                        g=solve_generalized_grid(inputs,eta_s=.1,nx=16385);f=g['per_expiry']
                        f.to_csv(day/'generalized_per_expiry.csv',index=False)
                        rows.append(dict(date=date,block=block,method='generalized_expiry_recursion',
                            success=True,eta_s=.1,fit_target_met=g['fit_target_met'],
                            coverage=float(f.n_inside.sum()/f.n_quotes.sum()),core_coverage=float(f.n_core_inside.sum()/f.n_core.sum()),
                            note='within-date expiry recursion; NOT an across-observation-date martingale'))
                    except Exception as exc:rows.append(dict(date=date,block=block,method='generalized_expiry_recursion',success=False,error=str(exc)))
                print(json.dumps(dict(date=date,lp_core=lprow['core_coverage'],lp_all=lprow['coverage'],
                    recent=[{k:r.get(k) for k in ['method','success','core_coverage','lp_reference_rms_spreads','qp_reference_rms_spreads','reason']} for r in rows[-2:]],
                    elapsed=time.perf_counter()-start)),flush=True)
            except Exception as exc:
                row=dict(date=date,block=block,method='daily_pipeline',success=False,error=f'{type(exc).__name__}: {exc}')
                rows.append(row);print(json.dumps(row),flush=True)
            (args.output/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
            pd.DataFrame([{k:v for k,v in r.items() if k not in {'surface_audit','attempts','seed'}} for r in rows]).to_csv(args.output/'summary.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
