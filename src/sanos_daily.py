"""Cross-observation-date calibration references and a fit-budget stability QP.

This is not a martingale transition between daily conditional risk-neutral laws.
References match absolute expiry and forward moneyness; prior annualized IV is
rolled to today's remaining time. They are a modelling choice, not a forecast.
"""
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import ndtr
from sanos_core import call_kernel
from sanos_incremental import quote_vectors,validate_frame,recompute_loss_variables


def unit_call(k,w):
    k,w=np.broadcast_arrays(np.asarray(k,float),np.asarray(w,float));s=np.sqrt(np.maximum(w,1e-30))
    d1=-np.log(k)/s+.5*s;d2=d1-s
    return np.where(k<1,1-k+k*ndtr(-d2)-ndtr(-d1),ndtr(d1)-k*ndtr(d2))


def implied_total_variance(k,price):
    k,p=np.broadcast_arrays(np.asarray(k,float),np.asarray(price,float))
    if np.any(p<np.maximum(1-k,0)-2e-7) or np.any(p>=1):raise ValueError('Invalid reference prices')
    low=np.zeros(k.shape);high=np.full(k.shape,16.)
    if np.any(unit_call(k,high)<p):raise ValueError('Reference variance exceeds inversion bracket')
    for _ in range(60):
        mid=(low+high)/2;below=unit_call(k,mid)<p
        low=np.where(below,mid,low);high=np.where(below,high,mid)
    return (low+high)/2


def snapshot(date,problem,x):
    return dict(date=str(pd.Timestamp(date).date()),problem=problem,x=np.asarray(x).copy())


def daily_references(previous,problem,date):
    if pd.Timestamp(previous['date'])>=pd.Timestamp(date):raise ValueError('Reference must be strictly earlier')
    from xtx_io import core_mask
    refs={};old=previous['problem'];daystep=(pd.Timestamp(date)-pd.Timestamp(previous['date'])).days
    for e,v in problem['items'].items():
        if e not in old['items']:continue
        ov=old['items'][e];k=np.asarray(v['market_k']);mask=core_mask(v['quotes'])
        mask &= (k>=np.min(ov['market_k']))&(k<=np.max(ov['market_k']))
        if not np.any(mask):continue
        oldT=float(ov.get('T',ov['dte']/365));newT=float(v.get('T',v['dte']/365))
        if not newT<oldT or abs((oldT-newT)*365-daystep)>1e-6:raise ValueError('Expiry roll inconsistent with observation dates')
        q=previous['x'][slice(*old['q_offsets'][e])]
        oldprice=call_kernel(ov['anchor_k'],k[mask],ov['smooth_variance'])@q
        w=implied_total_variance(k[mask],oldprice)
        target=np.zeros(len(k));target[mask]=unit_call(k[mask],w*newT/oldT)
        refs[e]=(target,mask)
    return refs


def seed_previous_marginals(session,previous,date):
    """Partial primal guess only; no old basis or cross-date constraint reuse."""
    if pd.Timestamp(previous['date'])>=pd.Timestamp(date):raise ValueError('Primal seed must be strictly earlier')
    old=previous['problem'];indices=[];values=[];used=0;skipped={}
    for e,v in session.problem['items'].items():
        if e not in old['items']:continue
        ov=old['items'][e];p=previous['x'][slice(*old['q_offsets'][e])]
        if p.min()<0:skipped[str(e)]='prior numerical negative weight';continue
        nonzero=p>0;s=np.asarray(ov['anchor_k'])[nonzero];p=p[nonzero];a=np.asarray(v['anchor_k'])
        if s.min()<a[0] or s.max()>a[-1]:skipped[str(e)]='prior support outside new model grid';continue
        j=np.clip(np.searchsorted(a,s,side='right')-1,0,len(a)-2);t=(s-a[j])/(a[j+1]-a[j])
        q=np.zeros(len(a));np.add.at(q,j,p*(1-t));np.add.at(q,j+1,p*t)
        lo,hi=session.problem['q_offsets'][e];indices.extend(range(lo,hi));values.extend(q);used+=1
    if indices:session._check(session.h.setSolution(len(indices),np.asarray(indices,np.int32),np.asarray(values,float)))
    return dict(expiries_seeded=used,skipped=skipped,old_basis_reused=False)


def stability_qp(problem,lp_x,quotes,references,*,loss_allowance=1e-4,time_limit=12,backend_order=('osqp','clarabel')):
    """QP: minimize core reference movement with original daily no-arbitrage.

    Total spread-normalized outside loss may exceed today's LP by at most the
    explicitly reported allowance. Core points fitted by LP remain hard bid/ask
    constraints. Full original-unit audit and these fit guards decide acceptance.
    """
    from xtx_io import core_mask
    start=time.perf_counter();p=problem
    qcols=np.concatenate([np.arange(a,b) for a,b in p['q_offsets'].values()]);nq=len(qcols)
    n=sum(len(v['market_k']) for v in p['items'].values())
    matrices=[];bids=[];asks=[];spreads=[];hard=[];reference_rows=[];targets=[]
    lpoutside=0.;offset=0
    for e,v in p['items'].items():
        a,b=p['q_offsets'][e];nobs=len(v['market_k']);k=v['market_k'][:,None];anchors=v['anchor_k'][None,:]
        variance=v['smooth_variance']
        if variance>0:
            s=np.sqrt(variance);d1=(np.log(anchors/k)+.5*variance)/s;d2=d1-s
            G=np.where(k<1,k*ndtr(-d2)-anchors*ndtr(-d1),anchors*ndtr(d1)-k*ndtr(d2))
        else:G=np.where(k<1,np.maximum(k-anchors,0),np.maximum(anchors-k,0))
        bid,ask=quotes[e];spread=ask-bid;intrinsic=np.maximum(1-v['market_k'],0)
        fit=v['G']@lp_x[a:b];outside=np.maximum(bid-fit,0)+np.maximum(fit-ask,0)
        lpoutside+=float(np.sum(outside/spread));hard.extend(core_mask(v['quotes'])&(outside<=1e-9))
        block=sparse.csr_matrix(G/spread[:,None]);matrices.append(block)
        bids.extend((bid-intrinsic)/spread);asks.extend((ask-intrinsic)/spread);spreads.extend(spread)
        if e in references:
            target,mask=references[e]
            reference_rows.extend((offset+np.flatnonzero(mask)).tolist());targets.extend(((target[mask]-intrinsic[mask])/spread[mask]).tolist())
        offset+=nobs
    if not reference_rows:return dict(success=False,reason='no_previous_day_overlap',attempts=[])
    G=sparse.block_diag(matrices,format='csc');nr=len(reference_rows);nv=nq+n+nr
    E=p['E'][:,qcols];ne=E.shape[0]
    ca,cb=p['calendar_rows'];ca-=ne;cb-=ne
    C=p['A'][ca:cb,:][:,qcols].tocsc()
    # Scale calendar rows without changing their zero RHS or feasible set.
    scale=np.asarray(abs(C).max(axis=1).toarray()).ravel();scale=np.where(scale>0,1/scale,1)
    C=sparse.diags(scale)@C
    eq=sparse.vstack([sparse.hstack([E,sparse.csr_matrix((ne,n+nr))]),
        sparse.hstack([G[reference_rows],sparse.csr_matrix((nr,n)),-sparse.eye(nr)])],format='csc')
    beq=np.r_[np.ones(ne),targets]
    upperloss=np.where(np.asarray(hard),0.,np.inf)
    inequalities=sparse.vstack([
        sparse.hstack([G,-sparse.eye(n),sparse.csr_matrix((n,nr))]),
        sparse.hstack([-G,-sparse.eye(n),sparse.csr_matrix((n,nr))]),
        sparse.hstack([C,sparse.csr_matrix((C.shape[0],n+nr))]),
        sparse.csr_matrix(np.r_[np.zeros(nq),np.ones(n),np.zeros(nr)][None,:])],format='csc')
    rhs=np.r_[asks,-np.asarray(bids),np.zeros(C.shape[0]),lpoutside+loss_allowance]
    lower=np.r_[np.zeros(nq+n),np.full(nr,-np.inf)];upper=np.r_[np.full(nq,np.inf),upperloss,np.full(nr,np.inf)]
    H=sparse.diags(np.r_[np.zeros(nq+n),np.full(nr,2/nr)],format='csc');linear=np.zeros(nv)
    initial=np.r_[lp_x[qcols],np.zeros(n),np.zeros(nr)]
    initial[nq:nq+n]=np.maximum.reduce([np.asarray(bids)-G@initial[:nq],G@initial[:nq]-np.asarray(asks),np.zeros(n)])
    initial[nq+n:]=G[reference_rows]@initial[:nq]-np.asarray(targets)
    attempts=[]
    for backend in backend_order:
        remaining=time_limit-(time.perf_counter()-start)
        if remaining<=0:break
        t=time.perf_counter();status='';candidate=None
        try:
            if backend=='osqp':
                import osqp
                A=sparse.vstack([eq,inequalities,sparse.eye(nv)],format='csc')
                solver=osqp.OSQP();solver.setup(P=H,q=linear,A=A,l=np.r_[beq,np.full(len(rhs),-np.inf),lower],
                    u=np.r_[beq,rhs,upper],verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=20000,
                    time_limit=min(remaining,6.),polishing=True)
                solver.warm_start(x=initial);raw=solver.solve();status=raw.info.status
                if raw.info.status_val in [1,2]:candidate=np.asarray(raw.x)
            else:
                import clarabel
                finite=np.flatnonzero(np.isfinite(upper));boxlow=-sparse.eye(nv,format='csc')[:nq+n]
                boxhigh=sparse.eye(nv,format='csc')[finite]
                A=sparse.vstack([eq,inequalities,boxlow,boxhigh],format='csc')
                b=np.r_[beq,rhs,np.zeros(nq+n),upper[finite]]
                settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
                settings.time_limit=remaining;settings.max_iter=150;settings.tol_feas=1e-10
                settings.tol_gap_abs=1e-8;settings.tol_gap_rel=1e-9
                solver=clarabel.DefaultSolver(H,linear,A,b,
                    [clarabel.ZeroConeT(ne+nr),clarabel.NonnegativeConeT(len(b)-ne-nr)],settings)
                raw=solver.solve();status=str(raw.status)
                if status in {'Solved','AlmostSolved'}:candidate=np.asarray(raw.x)
        except Exception as exc:status=f'{type(exc).__name__}: {exc}'
        attempt=dict(backend=backend,status=status,seconds=time.perf_counter()-t);attempts.append(attempt)
        if candidate is None:continue
        x=np.zeros(p['cursor']);x[qcols]=candidate[:nq];recompute_loss_variables(p,x,quotes)
        upper0,cost0=quote_vectors(p,quotes);audit=validate_frame(p,upper0,cost0,x,quotes)
        actualloss=audit['actual_outside_objective'];coreviol=0.;off=0
        for e,v in p['items'].items():
            fit=v['G']@x[slice(*p['q_offsets'][e])];bid,ask=quotes[e];count=len(fit)
            mask=np.asarray(hard[off:off+count]);off+=count
            if mask.any():coreviol=max(coreviol,float(np.max(np.maximum(bid-fit,fit-ask)[mask])))
        movement=float(np.mean((G[reference_rows]@x[qcols]-np.asarray(targets))**2))
        initial_movement=float(np.mean(initial[nq+n:]**2))
        alpha=1.
        if audit['validated'] and movement<=initial_movement+1e-8 and (actualloss>lpoutside+loss_allowance+1e-6 or coreviol>1e-9):
            # A converged QP may miss a tight fit-budget boundary numerically.
            # Take a feasible convex step from today's already validated LP;
            # never enlarge the fit allowance or alter current market quotes.
            f0=[];f1=[];bb=[];aa=[]
            for e,v in p['items'].items():
                f0.extend(v['G']@lp_x[slice(*p['q_offsets'][e])])
                f1.extend(v['G']@x[slice(*p['q_offsets'][e])]);bid,ask=quotes[e];bb.extend(bid);aa.extend(ask)
            f0,f1,bb,aa=map(np.asarray,(f0,f1,bb,aa));ss=aa-bb;hm=np.asarray(hard)
            def feasible(t):
                fit=f0+t*(f1-f0);outside=np.maximum(bb-fit,0)+np.maximum(fit-aa,0)
                return np.sum(outside/ss)<=lpoutside+loss_allowance and (not hm.any() or np.max(outside[hm])<=1e-9)
            lo,hi=0.,1.
            for _ in range(45):
                mid=(lo+hi)/2
                if feasible(mid):lo=mid
                else:hi=mid
            alpha=lo*(1-1e-5)
            x=(1-alpha)*lp_x+alpha*x;recompute_loss_variables(p,x,quotes)
            audit=validate_frame(p,upper0,cost0,x,quotes);actualloss=audit['actual_outside_objective']
            coreviol=0.;off=0
            for e,v in p['items'].items():
                fit=v['G']@x[slice(*p['q_offsets'][e])];bid,ask=quotes[e];count=len(fit)
                mask=np.asarray(hard[off:off+count]);off+=count
                if mask.any():coreviol=max(coreviol,float(np.max(np.maximum(bid-fit,fit-ask)[mask])))
            movement=float(np.mean((G[reference_rows]@x[qcols]-np.asarray(targets))**2))
        accepted=audit['validated'] and actualloss<=lpoutside+loss_allowance+1e-6 and coreviol<=1e-9 and movement<=initial_movement+1e-8
        attempt.update(validated=audit['validated'],fit_guard=coreviol,actual_outside_loss=actualloss,
            loss_limit=lpoutside+loss_allowance,initial_movement=initial_movement,movement=movement,
            feasible_step_alpha=alpha,selected_point_is_optimizer_result=alpha==1,accepted=bool(accepted))
        if accepted:return dict(success=True,x=x,audit=audit,attempts=attempts,reference_quotes=nr,
            lp_movement_rms=float(np.sqrt(initial_movement)),qp_movement_rms=float(np.sqrt(movement)),
            lp_outside_loss=lpoutside,loss_allowance=loss_allowance,feasible_step_alpha=alpha,
            selected_point_is_optimizer_result=alpha==1,seconds=time.perf_counter()-start)
    return dict(success=False,reason='no_QP_candidate_passed_original_audit_and_fit_guards',attempts=attempts,
                reference_quotes=nr,seconds=time.perf_counter()-start)


def active_support_qp(problem,lp_x,quotes,references,*,loss_allowance=1e-4,time_limit=20):
    """QP extension restricted to current LP's nonzero states and neighbours.

    No positive-mass cutoff: every nonzero LP state is retained. This is NOT
    claimed to be the unrestricted full-grid QP optimum. Final checks use the
    full original problem and recomputed original-unit market prices.
    """
    from sanos_core import build_convex_order_problem
    start=time.perf_counter();inputs={};selection={}
    for e,v in problem['items'].items():
        q=lp_x[slice(*problem['q_offsets'][e])];a=v['anchor_k']
        nonzero=np.flatnonzero(q!=0)
        keep=np.unique(np.r_[0,len(q)-1,np.argmin(abs(a-1)),nonzero,
            np.maximum(nonzero-1,0),np.minimum(nonzero+1,len(q)-1)]).astype(int)
        selection[e]=keep;inputs[e]=dict(v,anchor_k=a[keep])
    compact=build_convex_order_problem(inputs);initial=np.zeros(compact['cursor'])
    for e,keep in selection.items():
        initial[slice(*compact['q_offsets'][e])]=lp_x[slice(*problem['q_offsets'][e])][keep]
    recompute_loss_variables(compact,initial,quotes)
    result=stability_qp(compact,initial,quotes,references,loss_allowance=loss_allowance,
        time_limit=max(.001,time_limit-(time.perf_counter()-start)),backend_order=('clarabel','osqp'))
    result['support_mode']='LP_nonzero_states_plus_neighbours'
    result['full_state_count']=sum(b-a for a,b in problem['q_offsets'].values())
    result['allowed_state_count']=sum(len(x) for x in selection.values())
    if result['success']:
        full=np.zeros(problem['cursor'])
        for e,keep in selection.items():
            lo,hi=problem['q_offsets'][e]
            full[lo+keep]=result['x'][slice(*compact['q_offsets'][e])]
        recompute_loss_variables(problem,full,quotes);upper,cost=quote_vectors(problem,quotes)
        audit=validate_frame(problem,upper,cost,full,quotes)
        if not audit['validated']:
            result.update(success=False,reason='full_grid_audit_rejected_restricted_QP')
        else:result.update(x=full,audit=audit)
    result['seconds']=time.perf_counter()-start
    return result
