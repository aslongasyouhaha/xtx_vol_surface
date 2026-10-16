"""Experimental QP constrained not to worsen any current quote's outside error."""
import _bootstrap
import time
import numpy as np
from scipy import sparse
from scipy.special import ndtr
from sanos_core import build_convex_order_problem
from sanos_incremental import recompute_loss_variables,quote_vectors,validate_frame


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
