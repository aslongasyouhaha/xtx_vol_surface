"""Fit-preserving daily QP experiments; original-unit guards remain mandatory."""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from scipy.special import ndtr
from sanos_core import build_convex_order_problem
from sanos_incremental import recompute_loss_variables, quote_vectors, validate_frame, PersistentSanosLP
from sanos_daily import snapshot, daily_references, seed_previous_marginals, stability_qp, active_support_qp
from xtx_io import BLOCKS, load_saved_lp, fit_stats, load_inputs, save_fit

def run_fit_preserving(blocks,output,*,time_limit=20.,coordinates='absolute'):
    """Isolated previous-LP reference controls, with no sequential deployment claim."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rows=[]
    for block in blocks:
        eta,dates=BLOCKS[block]
        for old,date in zip(dates[:-1],dates[1:]):
            pp,px,_=load_saved_lp(old,block,eta);p,x,quotes=load_saved_lp(date,block,eta)
            refs=daily_references(snapshot(old,pp,px),p,date)
            result=fit_preserving_qp(p,x,quotes,refs,time_limit=time_limit,coordinates=coordinates)
            row=dict(experiment='pointwise_fit_preserving_QP',date=date,source_date=old,
                reference='previous daily LP; isolated comparison, not sequential QP replay',
                **{k:v for k,v in result.items() if k not in ['x','audit']})
            if result['success']:
                row.update(fit_stats(p,result['x']))
                np.savez_compressed(output/f'{date}_qp.npz',x=result['x'])
            rows.append(row)
            (output/'results.json').write_text(json.dumps(dict(status='provisional',results=rows),indent=2),encoding='utf8')
            pd.DataFrame([{k:v for k,v in r.items() if k!='surface_audit'} for r in rows]).to_csv(output/'results.csv',index=False)
            print(json.dumps({k:v for k,v in row.items() if k!='surface_audit'}),flush=True)


def fit_preserving_qp(problem,lp_x,quotes,references,*,time_limit=20,coordinates='absolute'):
    import clarabel
    start=time.perf_counter();selection={};inputs={}
    for e,v in problem['items'].items():
        a=v['anchor_k'];q=lp_x[slice(*problem['q_offsets'][e])];nz=np.flatnonzero(q!=0)
        keep=np.unique(np.r_[0,len(q)-1,np.argmin(abs(a-1)),nz,np.maximum(nz-1,0),np.minimum(nz+1,len(q)-1)]).astype(int)
        selection[e]=keep;inputs[e]=dict(v,anchor_k=a[keep])
    p=build_convex_order_problem(inputs)
    qcols=np.concatenate([np.arange(a,b) for a,b in p['q_offsets'].values()]);nq=len(qcols)
    initial=np.concatenate([lp_x[slice(*problem['q_offsets'][e])][selection[e]] for e in p['items']])
    blocks=[];lower=[];upper=[];spreads=[];refrows=[];targets=[];offset=0
    for e,v in p['items'].items():
        k=v['market_k'][:,None];a=v['anchor_k'][None,:];w=v['smooth_variance']
        if w>0:
            s=np.sqrt(w);d1=np.log(a/k)/s+s/2;d2=d1-s
            kernel=np.where(k<1,k*ndtr(-d2)-a*ndtr(-d1),a*ndtr(d1)-k*ndtr(d2))
        else:kernel=np.where(k<1,np.maximum(k-a,0),np.maximum(a-k,0))
        bid,ask=quotes[e];spread=ask-bid;intrinsic=np.maximum(1-v['market_k'],0)
        original=problem['items'][e]['G']@lp_x[slice(*problem['q_offsets'][e])]
        lower.extend((np.minimum(bid,original)-intrinsic)/spread)
        upper.extend((np.maximum(ask,original)-intrinsic)/spread);spreads.extend(spread)
        blocks.append(sparse.csr_matrix(kernel/spread[:,None]))
        if e in references:
            target,mask=references[e];refrows.extend((offset+np.flatnonzero(mask)).tolist())
            targets.extend(((target[mask]-intrinsic[mask])/spread[mask]).tolist())
        offset+=len(bid)
    if not refrows:return dict(success=False,reason='no_shared_reference')
    G=sparse.block_diag(blocks,format='csc');R=G[refrows];targets=np.asarray(targets);nr=len(targets)
    E=p['E'][:,qcols];ne=E.shape[0];ca,cb=p['calendar_rows'];C=p['A'][ca-ne:cb-ne,:][:,qcols]
    norms=np.asarray(abs(C).max(axis=1).toarray()).ravel();C=sparse.diags(1/np.maximum(norms,1e-30))@C
    pad=lambda a:sparse.hstack([a,sparse.csr_matrix((a.shape[0],nr))])
    eq=sparse.vstack([pad(E),sparse.hstack([R,-sparse.eye(nr)])],format='csc')
    inequality=sparse.vstack([pad(C),pad(G),pad(-G),pad(-sparse.eye(nq))],format='csc')
    A=sparse.vstack([eq,inequality],format='csc')
    b=np.r_[np.ones(ne),targets,np.zeros(C.shape[0]),upper,-np.asarray(lower),np.zeros(nq)]
    H=sparse.diags(np.r_[np.zeros(nq),np.full(nr,2/nr)],format='csc')
    if coordinates not in ['absolute','delta','moments']:raise ValueError('Unknown coordinates')
    origin=np.r_[initial,R@initial-targets] if coordinates=='delta' else np.zeros(nq+nr)
    basis=sparse.eye(nq+nr,format='csc');neq=ne+nr
    if coordinates=='moments':
        # Eliminate each expiry's mass/mean equations algebraically using its
        # two boundary weights. This parameterizes exactly the same q polytope.
        transforms=[];particular=[]
        for v in p['items'].values():
            a=v['anchor_k'];n=len(a);free=np.arange(1,n-1);width=a[-1]-a[0]
            rr=np.r_[free,np.zeros(len(free),int),np.full(len(free),n-1)]
            cc=np.tile(np.arange(len(free)),3)
            values=np.r_[np.ones(len(free)),(a[free]-a[-1])/width,(a[0]-a[free])/width]
            transforms.append(sparse.coo_matrix((values,(rr,cc)),shape=(n,n-2)).tocsc())
            particular.extend(np.r_[(a[-1]-1)/width,np.zeros(n-2),(1-a[0])/width])
        T=sparse.block_diag(transforms,format='csc');particular=np.asarray(particular)
        basis=sparse.block_diag([T,sparse.eye(nr)],format='csc')
        origin=np.r_[particular,R@particular-targets]
        A=A[ne:];b=b[ne:];neq=nr
    # An exact affine change of variables, preserving every original constraint.
    b=b-A@origin
    linear=np.asarray(basis.T@(H@origin)).ravel()
    A=(A@basis).tocsc();objective=(basis.T@H@basis).tocsc()
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
    settings.time_limit=max(.001,time_limit-(time.perf_counter()-start));settings.max_iter=200
    settings.tol_feas=1e-10;settings.tol_gap_abs=1e-8;settings.tol_gap_rel=1e-9
    raw=clarabel.DefaultSolver(objective,linear,A,b,
        [clarabel.ZeroConeT(neq),clarabel.NonnegativeConeT(inequality.shape[0])],settings).solve()
    if str(raw.status) not in ['Solved','AlmostSolved']:
        return dict(success=False,reason=str(raw.status),seconds=time.perf_counter()-start)
    q=(basis@np.asarray(raw.x)+origin)[:nq];candidate=np.zeros(problem['cursor'])
    at=0
    for e,keep in selection.items():
        lo,_=problem['q_offsets'][e];candidate[lo+keep]=q[at:at+len(keep)];at+=len(keep)
    base=[];fits=[];bids=[];asks=[];widths=[]
    for e,v in problem['items'].items():
        base.extend(v['G']@lp_x[slice(*problem['q_offsets'][e])]);fits.extend(v['G']@candidate[slice(*problem['q_offsets'][e])])
        bid,ask=quotes[e];bids.extend(bid);asks.extend(ask);widths.extend(ask-bid)
    base,fits,bids,asks,widths=map(np.asarray,[base,fits,bids,asks,widths])
    outside=lambda f:np.maximum(bids-f,0)+np.maximum(f-asks,0)
    base_error=outside(base);base_loss=float(np.sum(base_error/widths))
    raw_loss=float(np.sum(outside(fits)/widths))
    raw_quote_error_increase=float(np.max(outside(fits)-base_error))
    def guards(t):
        err=outside(base+t*(fits-base))
        return np.max(err-base_error)<=1e-9 and np.sum(err/widths)<=base_loss+1e-8
    alpha=1.
    if not guards(1):
        lo,hi=0.,1.
        for _ in range(50):
            mid=(lo+hi)/2
            if guards(mid):lo=mid
            else:hi=mid
        alpha=lo*(1-1e-5)
    x=(1-alpha)*lp_x+alpha*candidate;recompute_loss_variables(problem,x,quotes)
    upper0,cost=quote_vectors(problem,quotes);audit=validate_frame(problem,upper0,cost,x,quotes)
    initial_rms=float(np.sqrt(np.mean((R@initial-targets)**2)))
    final_rms=float(np.sqrt(np.mean((R@((1-alpha)*initial+alpha*q)-targets)**2)))
    valid=bool(audit['validated'] and guards(alpha) and final_rms<initial_rms-1e-8)
    return dict(success=valid,reason=None if valid else 'no_valid_fit_preserving_improvement',x=x,audit=audit,
        seconds=time.perf_counter()-start,lp_reference_rms=initial_rms,qp_reference_rms=final_rms,
        feasible_step_alpha=alpha,lp_outside_loss=base_loss,qp_outside_loss=float(np.sum(outside(base+alpha*(fits-base))/widths)),
        raw_qp_outside_loss=raw_loss,raw_quote_error_increase=raw_quote_error_increase,
        coordinates=coordinates,solver_status=str(raw.status),solver_primal_residual=float(raw.r_prim),
        raw_moment_error=float(np.max(abs(E@q-1))),
        raw_scaled_quote_violation=float(max(0,np.max(G@q-np.asarray(upper)),np.max(np.asarray(lower)-G@q))),
        relative_reference_improvement=(initial_rms-final_rms)/initial_rms if initial_rms else 0.,
        max_quote_error_increase=float(np.max(outside(base+alpha*(fits-base))-base_error)),
        selected_point_is_optimizer_result=alpha==1,allowed_states=nq)


def run_daily_validation(args):
    args.output.mkdir(parents=True,exist_ok=True);rows=[]
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
                    from xtx_generalized_research import solve_generalized_grid
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


def run_daily_update(args):
    root=args.baseline_dir;out=args.output;out.mkdir(parents=True,exist_ok=True);rows=[]
    for block in args.blocks:
        eta,dates=BLOCKS[block];previous=None
        for date in dates:
            path=root/block/date/'lp/model.npz'
            if not path.exists():raise ValueError(f'Run daily LP validation first: {date}')
            p,x,quotes=load_saved_lp(date,block,eta,baseline_dir=root)
            if previous is None:
                previous=snapshot(date,p,x);continue
            refs=daily_references(previous,p,date)
            result=active_support_qp(p,x,quotes,refs,time_limit=args.time_limit,loss_allowance=args.loss_allowance)
            metadata=dict(block=block,eta=eta,method='cross_day_active_support_QP',source_date=previous['date'],
                support_mode=result['support_mode'],full_state_count=result['full_state_count'],
                allowed_state_count=result['allowed_state_count'],attempts=result['attempts'],seconds=result['seconds'])
            if result['success']:
                metadata.update(reference_quotes=result['reference_quotes'],lp_reference_rms_spreads=result['lp_movement_rms'],
                    qp_reference_rms_spreads=result['qp_movement_rms'],loss_allowance=result['loss_allowance'],
                    lp_outside_loss=result['lp_outside_loss'],feasible_step_alpha=result['feasible_step_alpha'],
                    selected_point_is_optimizer_result=result['selected_point_is_optimizer_result'])
                row=save_fit(out/block/date,date,p,result['x'],metadata)
                previous=snapshot(date,p,result['x'])
            else:
                row=dict(date=date,success=False,reason=result['reason'],deployed_fallback='current_validated_LP',**metadata)
                previous=snapshot(date,p,x)
            rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
            pd.DataFrame([{k:v for k,v in r.items() if k not in {'surface_audit','attempts'}} for r in rows]).to_csv(out/'summary.csv',index=False)
            print(json.dumps({k:row.get(k) for k in ['date','success','core_coverage','lp_reference_rms_spreads','qp_reference_rms_spreads','reason','allowed_state_count']}),flush=True)
