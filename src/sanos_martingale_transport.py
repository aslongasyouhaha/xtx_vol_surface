"""Constructive state-dependent martingale control, not a production model.

Source-grid rounding is an explicit martingale step, not probability pruning.
Including quote strikes preserves the original source's calls at those strikes.
The resulting two-step kernel maps the ORIGINAL distribution into the target.
"""
import numpy as np
import time
from scipy import sparse
from scipy.optimize import linprog


def calls(states,weights,strikes):
    return np.maximum(np.asarray(states)[None,:]-np.asarray(strikes)[:,None],0)@weights


def couple_marginals(source_states,source_weights,target_states,target_weights,*,time_limit=30,max_states=384,marginal_tolerance=0.):
    """Fixed-marginal martingale coupling. Remove exact zero states ONLY.

    Relative fourth-power displacement is a locality preference among feasible
    couplings; it does not change marginals and is not estimated dynamics.
    """
    x,p,y,q=map(lambda a:np.asarray(a,float),(source_states,source_weights,target_states,target_weights))
    if x.shape!=p.shape or y.shape!=q.shape or np.any(~np.isfinite(np.r_[x,p,y,q])) or np.any(p<0) or np.any(q<0) or np.any(x<=0) or np.any(y<=0):
        raise ValueError('Invalid nonnegative marginals')
    ii=np.flatnonzero(p>0);jj=np.flatnonzero(q>0)
    x,p,y,q=x[ii],p[ii],y[jj],q[jj]
    if marginal_tolerance<0 or marginal_tolerance>1e-9:raise ValueError('Marginal tolerance must be between 0 and 1e-9')
    if marginal_tolerance:
        # Explicit numerical relaxation only: allow tiny mass at source states
        # that a rounded target marginal may have lost. No mass floor is used.
        union=np.unique(np.r_[x,y]);requested=np.zeros(len(union))
        requested[np.searchsorted(union,y)]=q
        y,q=union,requested;jj=np.full(len(y),-1,dtype=int)
    ns,nt=len(x),len(y)
    if not ns or not nt or max(ns,nt)>min(max_states,384):raise ValueError('Fixed coupling exceeds resource cap')
    if max(abs(p.sum()-1),abs(q.sum()-1),abs(p@x-1),abs(q@y-1))>2e-7:
        raise ValueError('Marginals must have unit mass and mean')
    knots=np.unique(np.r_[x,y])
    if np.max(calls(x,p,knots)-calls(y,q,knots))>2e-7:raise ValueError('Marginals violate convex order')
    nv=ns*nt;col=np.arange(nv);row=np.repeat(np.arange(ns),nt)
    mass=sparse.coo_matrix((np.ones(nv),(row,col)),shape=(ns,nv))
    relative=y[None,:]/x[:,None]-1
    mean=sparse.coo_matrix((relative.ravel(),(row,col)),shape=(ns,nv))
    target=sparse.coo_matrix((np.repeat(p,nt),(np.tile(np.arange(nt),ns),col)),shape=(nt,nv))
    A=sparse.vstack([mass,mean,target],format='csr');b=np.r_[np.ones(ns),np.zeros(ns),q]
    objective=(p[:,None]*relative**4).ravel()
    Aub,bub,bounds=None,None,(0,None)
    if marginal_tolerance:
        A=sparse.hstack([sparse.vstack([mass,mean]),sparse.csr_matrix((2*ns,nt))],format='csr')
        b=np.r_[np.ones(ns),np.zeros(ns)]
        Aub=sparse.vstack([sparse.hstack([target,-sparse.eye(nt)]),
                           sparse.hstack([-target,-sparse.eye(nt)])],format='csr')
        bub=np.r_[q,-q]
        objective=np.r_[1e-8*objective,np.ones(nt)]
        bounds=[(0,None)]*nv+[(0,marginal_tolerance)]*nt
    started=time.perf_counter()
    if marginal_tolerance:
        bounds=list(bounds)
    else:bounds=[(0,None)]*nv
    for method,presolve,tol in [('highs',True,1e-9),('highs-ds',False,1e-10),('highs-ipm',False,1e-10)]:
        for refinement in range(4):
            remaining=time_limit-(time.perf_counter()-started)
            if remaining<=0:raise RuntimeError('Fixed coupling budget exhausted')
            r=linprog(objective,A_eq=A,b_eq=b,A_ub=Aub,b_ub=bub,bounds=bounds,method=method,
                options={'time_limit':remaining,'presolve':presolve,'primal_feasibility_tolerance':tol,'dual_feasibility_tolerance':tol})
            if not r.success:break
            negative=np.flatnonzero(r.x[:nv]<0)
            if not len(negative):break
            # Re-solve on a feasible face, never clip a returned probability.
            for index in negative:bounds[index]=(0,0)
        if r.success and r.x[:nv].min()>=0:break
    else:raise RuntimeError(f'No nonnegative fixed coupling: {r.message}')
    Q=r.x[:nv].reshape(ns,nt);actual=p@Q
    errors=dict(conditional_mass=float(np.max(abs(Q.sum(axis=1)-1))),
        conditional_mean=float(np.max(abs(Q@y-x))),
        target_marginal=float(np.max(abs(actual-q))),
        target_mass=float(abs(actual.sum()-1)),target_mean=float(abs(actual@y-1)),
        negative_probability=max(0.,float(-Q.min())))
    return dict(validated=max(errors.values())<=2e-7,Q=Q,source_indices=ii,target_indices=jj,
        source_states=x,target_states=y,source_weights=p,target_weights=actual,
        requested_target=q,errors=errors,marginal_tolerance=marginal_tolerance,
        marginal_l1_error=float(np.sum(abs(actual-q))))


def project_source(states,weights,grid):
    s,p,g=map(lambda x:np.asarray(x,float),(states,weights,grid))
    if s.ndim!=1 or p.shape!=s.shape or g.ndim!=1 or len(g)<2 or np.any(np.diff(g)<=0):
        raise ValueError('Invalid source or projection grid')
    if np.any(~np.isfinite(np.r_[s,p,g])) or np.any(p<0) or s.min()<g[0] or s.max()>g[-1]:
        raise ValueError('Source must be nonnegative and within the grid')
    j=np.clip(np.searchsorted(g,s,side='right')-1,0,len(g)-2)
    t=(s-g[j])/(g[j+1]-g[j])
    P=sparse.coo_matrix((np.r_[1-t,t],(np.r_[np.arange(len(s)),np.arange(len(s))],np.r_[j,j+1])),
        shape=(len(s),len(g))).tocsr()
    return np.asarray(p@P).ravel(),P


def fit_state_dependent_transition(states,weights,item,*,preserve_strikes=(),time_limit=30,max_grid_states=384):
    s,p=np.asarray(states,float),np.asarray(weights,float)
    if np.any(s<=0) or np.any(np.diff(s)<=0) or abs(p.sum()-1)>2e-7 or abs(s@p-1)>2e-7:
        raise ValueError('Source must have unit mass and mean on a positive grid')
    k=np.asarray(item['market_k'],float)
    g=np.unique(np.r_[s[0],s[-1],k,preserve_strikes])
    # Bound every quadratic allocation BEFORE building any transition arrays.
    # Original 16k source nodes are mapped by a two-entry sparse projection;
    # no original-grid squared array or LP is ever constructed.
    if len(g)>max_grid_states or max_grid_states>384:
        raise ValueError(f'Transition grid {len(g)} exceeds hard resource cap 384')
    if g[0]!=s[0] or g[-1]!=s[-1]:raise ValueError('Quote outside source grid')
    source,P=project_source(s,p,g);n=len(g);m=len(k);nc=n*n
    # Q[source,destination] is conditional probability, not joint mass.
    rows=np.repeat(np.arange(n),n);cols=np.arange(nc)
    rowmass=sparse.coo_matrix((np.ones(nc),(rows,cols)),shape=(n,nc)).tocsr()
    rowmean=sparse.coo_matrix(((g[None,:]-g[:,None]).ravel(),(rows,cols)),shape=(n,nc)).tocsr()
    marginal=sparse.coo_matrix((np.repeat(source,n),(np.tile(np.arange(n),n),cols)),shape=(n,nc)).tocsr()
    Aeq=sparse.vstack([
        sparse.hstack([rowmass,sparse.csr_matrix((n,n+2*m))]),
        sparse.hstack([rowmean,sparse.csr_matrix((n,n+2*m))]),
        sparse.hstack([marginal,-sparse.eye(n),sparse.csr_matrix((n,2*m))])],format='csr')
    G=sparse.csr_matrix(np.maximum(g[None,:]-k[:,None],0));I=sparse.eye(m);Z=sparse.csr_matrix((m,m))
    front=sparse.csr_matrix((m,nc))
    Aub=sparse.vstack([sparse.hstack([front,G,-I,Z]),sparse.hstack([front,-G,-I,Z]),
        sparse.hstack([front,G,Z,-I]),sparse.hstack([front,-G,Z,-I])],format='csr')
    bid,ask,mid=[np.asarray(item[x],float) for x in ['pure_bid','pure_ask','pure_mid']]
    if np.any(ask<=bid) or not np.all(np.isfinite(np.r_[bid,ask,mid])):raise ValueError('Invalid quotes')
    objective=np.r_[np.zeros(nc+n),1/(ask-bid),1e-8/(ask-bid)]
    started=time.perf_counter()
    profiles=[('highs',True,1e-9),('highs-ds',False,1e-10),('highs-ipm',False,1e-10)]
    for method,presolve,tol in profiles:
        remaining=time_limit-(time.perf_counter()-started)
        if remaining<=0:raise RuntimeError('Transition solver budget exhausted')
        res=linprog(objective,A_ub=Aub,b_ub=np.r_[ask,-bid,mid,-mid],
            A_eq=Aeq,b_eq=np.r_[np.ones(n),np.zeros(2*n)],bounds=(0,None),method=method,
            options={'time_limit':remaining,'presolve':presolve,
                'primal_feasibility_tolerance':tol,'dual_feasibility_tolerance':tol})
        # Never propagate negative probabilities, even within LP feasibility tol.
        if res.success and np.min(res.x[:nc])>=0:break
    else:raise RuntimeError(f'No nonnegative transition returned: {res.message}')
    Q=res.x[:nc].reshape(n,n)
    # Recompute target through the actual original-source -> P -> Q chain.
    target=source@Q;fit=calls(g,target,k)
    mean=Q@g;mass=Q.sum(axis=1)
    union=np.unique(np.r_[s,g])
    # Avoid dense all-strike payoff matrix for the 16k original grid.
    def fast_calls(a,q,kk):
        sm=np.r_[np.cumsum(q[::-1])[::-1],0.];sx=np.r_[np.cumsum((a*q)[::-1])[::-1],0.]
        idx=np.searchsorted(a,kk,side='right');return sx[idx]-kk*sm[idx]
    errors=dict(conditional_mass=float(np.max(abs(mass-1))),
        conditional_mean=float(np.max(abs(mean-g))),
        composed_conditional_mass=float(np.max(abs(P@mass-1))),
        composed_conditional_mean=float(np.max(abs(P@mean-s))),
        target_mass=float(abs(target.sum()-1)),target_mean=float(abs(target@g-1)),
        negative_probability=max(0.,float(-Q.min()),float(-target.min())),
        marginal_consistency=float(np.max(abs(target-res.x[nc:nc+n]))),
        calendar_violation=max(0.,float(np.max(fast_calls(s,p,union)-fast_calls(g,target,union)))))
    preserve=np.unique(np.r_[k,preserve_strikes])
    errors['source_quote_preservation']=float(np.max(abs(fast_calls(g,source,preserve)-fast_calls(s,p,preserve))))
    outside=np.maximum.reduce([bid-fit,fit-ask,np.zeros(m)])
    return dict(validated=max(errors.values())<=2e-7,errors=errors,grid=g,
        source=source,target=target,Q=Q,projection=P,fit=fit,
        coverage=float(np.mean(outside<=1e-9)),outside_objective=float(np.sum(outside/(ask-bid))),
        objective=float(np.sum((outside+1e-8*abs(fit-mid))/(ask-bid))),
        conditional_relative_variance=(Q@(g*g)-mean*mean)/(g*g))
