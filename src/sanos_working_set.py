"""Restricted expiry updates accepted only with a global Lagrangian bound.

No original constraints are relaxed in the accepted result. Nonworking expiry
weights are temporarily fixed; a full original-unit audit plus a global bound
decides whether that restriction lost a material objective improvement.
"""
from __future__ import annotations
import time
import numpy as np
from sanos_incremental import quote_vectors,validate_frame,expiry_columns,recompute_loss_variables


def mean_one_minimum(anchors,values):
    """Min values@q subject to q>=0, sum(q)=anchors@q=1.

    Every extreme point has at most two atoms. Enumerating pairs across one
    evaluates the lower convex envelope at one without another optimization.
    """
    a=np.asarray(anchors,float);r=np.asarray(values,float)
    left=a<1;right=a>1;at=a==1
    candidates=[]
    if np.any(at):candidates.append(float(r[at].min()))
    if np.any(left) and np.any(right):
        al=a[left,None];ar=a[None,right]
        rl=r[left,None];rr=r[None,right]
        candidates.append(float((((ar-1)*rl+(1-al)*rr)/(ar-al)).min()))
    if not candidates:raise ValueError('Anchors must support a unit-mean distribution')
    return min(candidates)


def global_lower_bound(problem,matrix,upper,costs,dual):
    """A bound on the unrestricted original LP, including numerical repair.

    Equality multipliers are unrestricted; upper-inequality multipliers <=0.
    Repair each bid/ask dual pair to ensure nonnegative reduced cost for the
    unbounded-above error variable. Residual q reduced costs are minimized
    exactly over each original unit-mass/unit-mean simplex.
    """
    y=np.asarray(dual,float).copy();ne=problem['E'].shape[0]
    y[ne:]=np.minimum(y[ne:],0.)
    cursor=ne
    for ex,v in problem['items'].items():
        n=len(v['market_k']);first=y[cursor:cursor+n];second=y[cursor+n:cursor+2*n]
        budget=costs[slice(*problem['z_offsets'][ex])]
        used=-(first+second);safe_budget=budget*(1-1e-12);over=used>safe_budget
        factor=np.ones(n)
        factor[over]=safe_budget[over]/used[over]
        first*=factor;second*=factor;cursor+=2*n
    for ex,(first_row,second_row,n) in problem.get('mid_rows',{}).items():
        first=y[first_row:first_row+n];second=y[second_row:second_row+n]
        budget=costs[slice(*problem['mid_offsets'][ex])]*(1-1e-12)
        used=-(first+second);factor=np.ones(n);over=used>budget
        factor[over]=budget[over]/used[over];first*=factor;second*=factor
    reduced=costs-matrix.T@y
    correction=0.;block_min={}
    for ex,v in problem['items'].items():
        value=mean_one_minimum(v['anchor_k'],reduced[slice(*problem['q_offsets'][ex])])
        block_min[ex]=value;correction+=value
    # z has no finite upper bound; even a tiny negative reduced cost would
    # invalidate a finite bound. The interior budget margin above avoids this.
    cursor=ne
    for ex,v in problem['items'].items():
        n=len(v['market_k']);rz=reduced[slice(*problem['z_offsets'][ex])]
        if np.any(rz < 0):raise ValueError('Invalid error-variable dual budget')
        if ex in problem.get('mid_offsets',{}):
            if np.any(reduced[slice(*problem['mid_offsets'][ex])]<0):raise ValueError('Invalid midpoint-variable dual budget')
        cursor+=2*n
    base=float(upper@y)
    roundoff=1e-12*(1+float(abs(upper)@abs(y))+abs(correction))
    return dict(lower_bound=base+correction-roundoff,dual=y,reduced_cost=reduced,
                block_minimum=block_min,roundoff_allowance=roundoff)


def polish_market_duals(session,matrix,upper,costs,dual,expiries,*,deadline):
    """Fix calendar multipliers and optimize per-expiry Lagrangian subproblems."""
    import highspy
    p=session.problem;ne=p['E'].shape[0];nmarket=sum(2*len(v['market_k']) for v in p['items'].values())
    calendar_start=ne+nmarket
    calendar_end=calendar_start+p['calendar_constraint_count']
    shadow=-(matrix[calendar_start:calendar_end,:].T@dual[calendar_start:calendar_end])
    polished=dual.copy();cursor=ne;attempts=[]
    for ex_index,(ex,v) in enumerate(p['items'].items()):
        n=len(v['market_k']);market_rows=np.arange(cursor,cursor+2*n);cursor+=2*n
        if ex not in expiries or time.perf_counter()>=deadline:continue
        columns=expiry_columns(p,ex)
        rows=np.r_[2*ex_index,2*ex_index+1,market_rows]
        if ex in p.get('mid_rows',{}):
            plus,minus,nmid=p['mid_rows'][ex];rows=np.r_[rows,np.arange(plus,plus+nmid),np.arange(minus,minus+nmid)]
        M=matrix[rows,:][:,columns].tocsc()
        h=highspy.Highs()
        for key,value in dict(output_flag=False,threads=1,solver='simplex',presolve='on',simplex_scale_strategy=0,
                              primal_feasibility_tolerance=1e-9 if p.get('mid_penalty') else 1e-7,
                              dual_feasibility_tolerance=1e-9 if p.get('mid_penalty') else 1e-7,
                              time_limit=max(1e-5,deadline-time.perf_counter())).items():session._check(h.setOptionValue(key,value))
        lp=highspy.HighsLp();lp.num_col_=len(columns);lp.num_row_=len(rows)
        lp.col_cost_=costs[columns]+shadow[columns];lp.col_lower_=np.zeros(len(columns));lp.col_upper_=np.full(len(columns),np.inf)
        lp.row_lower_=np.r_[1.,1.,np.full(len(rows)-2,-np.inf)];lp.row_upper_=upper[rows]
        lp.a_matrix_.format_=highspy.MatrixFormat.kColwise;lp.a_matrix_.start_=M.indptr.astype(np.int32)
        lp.a_matrix_.index_=M.indices.astype(np.int32);lp.a_matrix_.value_=M.data;session._check(h.passModel(lp))
        t=time.perf_counter();h.run();elapsed=time.perf_counter()-t
        status=h.getModelStatus();attempts.append(dict(expiry=str(ex.date()),solve_seconds=elapsed,message=h.modelStatusToString(status)))
        if status==highspy.HighsModelStatus.kOptimal:polished[rows]=h.getSolution().row_dual
    return polished,attempts


def restricted_update(session,previous_x,previous_dual,previous_upper,previous_costs,
                      *,deadline,max_expiries=5,objective_tolerance=1e-6):
    """Try small working sets; never publish a merely locally optimal curve."""
    if session.compiled['scaling']!='none':return dict(success=False,reason='unsupported_compilation',attempts=[])
    import highspy
    p=session.problem;matrix=session.audit_cache['matrix'];lower=session.audit_cache['lower']
    upper,costs=quote_vectors(p,session.quotes);exdates=list(p['items']);dirty=[]
    cursor=p['E'].shape[0]
    for ex,v in p['items'].items():
        n=2*len(v['market_k']);z0,z1=p['z_offsets'][ex]
        if np.any(upper[cursor:cursor+n]!=previous_upper[cursor:cursor+n]) or np.any(costs[z0:z1]!=previous_costs[z0:z1]):dirty.append(ex)
        cursor+=n
    if not dirty or len(dirty)>max_expiries:return dict(success=False,reason='not_local',attempts=[])
    working=set(dirty);attempts=[];started=time.perf_counter()
    original_activity=matrix@previous_x
    polished_dual=previous_dual.copy();polished_expiries=set()
    while len(working)<=max_expiries and time.perf_counter()<deadline:
        build=time.perf_counter()
        free=np.concatenate([expiry_columns(p,ex) for ex in exdates if ex in working])
        columns=matrix[:,free].tocsc();rows=np.unique(columns.indices)
        local_matrix=columns[rows,:].tocsc()
        fixed_activity=original_activity[rows]-local_matrix@previous_x[free]
        local_lower=lower[rows]-fixed_activity;local_upper=upper[rows]-fixed_activity
        h=highspy.Highs()
        options=dict(output_flag=False,threads=1,random_seed=0,solver='simplex',simplex_strategy=1,
                     presolve='on',simplex_scale_strategy=0,primal_feasibility_tolerance=1e-9 if p.get('mid_penalty') else 1e-7,
                     dual_feasibility_tolerance=1e-9 if p.get('mid_penalty') else 1e-7,time_limit=max(1e-5,deadline-time.perf_counter()))
        for name,value in options.items():session._check(h.setOptionValue(name,value))
        lp=highspy.HighsLp();lp.num_col_=len(free);lp.num_row_=len(rows)
        lp.col_cost_=costs[free];lp.col_lower_=np.zeros(len(free));lp.col_upper_=np.full(len(free),np.inf)
        lp.row_lower_=local_lower;lp.row_upper_=local_upper
        lp.a_matrix_.format_=highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_=local_matrix.indptr.astype(np.int32)
        lp.a_matrix_.index_=local_matrix.indices.astype(np.int32);lp.a_matrix_.value_=local_matrix.data
        session._check(h.passModel(lp));build_seconds=time.perf_counter()-build
        t=time.perf_counter();api_status=h.run();solve_seconds=time.perf_counter()-t
        presolve_retry=False
        if h.getModelStatus()!=highspy.HighsModelStatus.kOptimal and time.perf_counter()<deadline:
            # Fixed neighbors can leave tiny residuals in presolve substitutions.
            # Re-solving the identical rows without presolve changes no tolerance.
            session._check(h.clearSolver());session._check(h.setOptionValue('presolve','off'))
            session._check(h.setOptionValue('time_limit',h.getRunTime()+max(1e-5,deadline-time.perf_counter())))
            t=time.perf_counter();api_status=h.run();solve_seconds+=time.perf_counter()-t;presolve_retry=True
        status=h.getModelStatus();row=dict(n_expiries=len(working),n_variables=len(free),n_rows=len(rows),
            build_seconds=build_seconds,solve_seconds=solve_seconds,message=h.modelStatusToString(status),
            iterations=h.getInfo().simplex_iteration_count,api_status=int(api_status),presolve_retry=presolve_retry)
        if status!=highspy.HighsModelStatus.kOptimal:
            attempts.append(row)
            adjacent=[]
            for ex in sorted(working):
                j=exdates.index(ex)
                for index in [j-1,j+1]:
                    if 0<=index<len(exdates) and exdates[index] not in working and exdates[index] not in adjacent:adjacent.append(exdates[index])
            additions=adjacent[:min(2,max_expiries-len(working))]
            if not additions:break
            working.update(additions);continue
        raw=h.getSolution();candidate=previous_x.copy();candidate[free]=raw.col_value
        # Recompute loss variables exactly. This never changes q or arbitrage
        # constraints, and cannot increase the optimum's price loss.
        recompute_loss_variables(p,candidate,session.quotes)
        audit=validate_frame(p,upper,costs,candidate,session.quotes,audit_cache=session.audit_cache)
        dual=previous_dual.copy();dual[rows]=raw.row_dual
        certs=[global_lower_bound(p,matrix,upper,costs,dual),
               global_lower_bound(p,matrix,upper,costs,previous_dual)]
        polished_dual,polish_attempts=polish_market_duals(session,matrix,upper,costs,polished_dual,
            working-polished_expiries,deadline=deadline)
        polished_expiries.update(working)
        certs.append(global_lower_bound(p,matrix,upper,costs,polished_dual))
        cert=max(certs,key=lambda c:c['lower_bound'])
        gap=audit['actual_objective']-cert['lower_bound']
        row.update(validated=audit['validated'],objective_gap=gap,lower_bound=cert['lower_bound'],
                   actual_outside_objective=audit['actual_outside_objective'],dual_polishing=polish_attempts)
        attempts.append(row)
        if audit['validated'] and -objective_tolerance<=gap<=objective_tolerance:
            return dict(success=True,x=candidate,dual=cert['dual'],audit=audit,
                        lower_bound=cert['lower_bound'],objective_gap=gap,attempts=attempts,
                        request_seconds=time.perf_counter()-started)
        # Only a certified global gap can accept. Otherwise free the blocks
        # with the largest unexplained Lagrangian contribution, adding adjacent
        # maturities when numerical ties give no useful ranking.
        scores={ex:float(cert['reduced_cost'][slice(*p['q_offsets'][ex])]@candidate[slice(*p['q_offsets'][ex])]
                         -cert['block_minimum'][ex]) for ex in exdates if ex not in working}
        adjacent=set()
        for ex in working:
            j=exdates.index(ex)
            if j:adjacent.add(exdates[j-1])
            if j+1<len(exdates):adjacent.add(exdates[j+1])
        ranked=sorted(adjacent-working,key=lambda ex:scores.get(ex,0),reverse=True)
        ranked += [ex for ex in sorted(scores,key=scores.get,reverse=True) if ex not in ranked]
        additions=ranked[:min(2,max_expiries-len(working))]
        if not additions:break
        working.update(additions)
    return dict(success=False,reason='no_global_certificate',attempts=attempts,
                request_seconds=time.perf_counter()-started)
