"""Optional solver controls and benchmarks; accepted fits retain original-unit audits."""

from __future__ import annotations
import time
import numpy as np
import copy
import json
import pandas as pd
import struct
import subprocess
from sanos_incremental import quote_vectors, validate_frame, expiry_columns, recompute_loss_variables, NUMERICS_PROFILES, PersistentSanosLP, prepare_validation, compile_native_model, native_quote_vectors
from xtx_io import load_inputs, core_mask
from sanos_core import solve_convex_order_lp, build_convex_order_problem
from scipy.optimize import linprog

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


# Active-state column control, with unrestricted objective certification.
def prepare_active_model(session,previous_x):
    """Prepare factorization before updates; retain all original basic columns."""
    if session.compiled['scaling']!='none':raise ValueError('Unsupported compilation')
    p=session.problem;state=set();selected=set();basis=session.h.getBasis()
    for e,v in p['items'].items():
        lo,hi=p['q_offsets'][e];state.update(range(lo,hi))
        selected.update(i for i in range(lo,hi) if previous_x[i]>0 or basis.col_status[i]==session.highspy.HighsBasisStatus.kBasic)
        selected.update([lo,hi-1,lo+int(np.argmin(abs(v['anchor_k']-1)))])
    columns=np.array(sorted(selected|(set(range(p['cursor']))-state)),dtype=np.int32)
    M=session.audit_cache['matrix'][:,columns].tocsc();h=session.highspy.Highs()
    for name,value in dict(output_flag=False,threads=1,random_seed=0,solver='simplex',presolve='off',simplex_strategy=1,
        primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9).items():session._check(h.setOptionValue(name,value))
    for name,value in NUMERICS_PROFILES[session._profile_name].items():
        if name not in {'solver','presolve'}:session._check(h.setOptionValue(name,value))
    upper,cost=quote_vectors(p,session.quotes);lp=session.highspy.HighsLp()
    lp.num_col_=len(columns);lp.num_row_=M.shape[0]
    lp.col_cost_=cost[columns];lp.col_lower_=np.zeros(len(columns));lp.col_upper_=np.full(len(columns),np.inf)
    lp.row_lower_=session.audit_cache['lower'];lp.row_upper_=upper
    lp.a_matrix_.format_=session.highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_=M.indptr.astype(np.int32);lp.a_matrix_.index_=M.indices.astype(np.int32);lp.a_matrix_.value_=M.data
    session._check(h.passModel(lp))
    reduced_basis=session.highspy.HighsBasis()
    reduced_basis.col_status=[basis.col_status[i] for i in columns];reduced_basis.row_status=basis.row_status
    session._check(h.setBasis(reduced_basis));session._check(h.setOptionValue('time_limit',10.))
    h.run()
    if h.getModelStatus()!=session.highspy.HighsModelStatus.kOptimal:raise ValueError('Active-model preparation failed')
    return dict(h=h,columns=columns,selected=selected,prewarm_iterations=h.getInfo().simplex_iteration_count)


def active_state_update(session,previous_x,*,budget=1.0,max_rounds=8,objective_tolerance=1e-6,prepared=None):
    start=time.perf_counter();deadline=start+budget
    if session.compiled['scaling']!='none':return dict(success=False,reason='unsupported_compilation',attempts=[])
    if budget<=0 or max_rounds<1:raise ValueError('Positive budget and round count required')
    p=session.problem;matrix=session.audit_cache['matrix'];lower=session.audit_cache['lower']
    upper,cost=quote_vectors(p,session.quotes)
    state_cols=set();selected=set()
    for e,v in p['items'].items():
        lo,hi=p['q_offsets'][e];allcols=np.arange(lo,hi);state_cols.update(allcols)
        # Retain every strictly positive previous weight, without a mass cutoff.
        selected.update(allcols[np.asarray(previous_x)[lo:hi]>0])
        a=v['anchor_k'];selected.update([lo,hi-1,lo+int(np.argmin(abs(a-1)))])
    nonstate=set(range(p['cursor']))-state_cols
    columns=np.array(sorted(selected|nonstate),dtype=np.int32)
    h=session.highspy.Highs()
    for name,value in dict(output_flag=False,threads=1,random_seed=0,solver='simplex',
        simplex_strategy=1,presolve='off',simplex_scale_strategy=0,
        primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9).items():
        session._check(h.setOptionValue(name,value))
    M=matrix[:,columns].tocsc();lp=session.highspy.HighsLp()
    lp.num_col_=len(columns);lp.num_row_=matrix.shape[0]
    lp.col_cost_=cost[columns];lp.col_lower_=np.zeros(len(columns));lp.col_upper_=np.full(len(columns),np.inf)
    lp.row_lower_=lower;lp.row_upper_=upper
    lp.a_matrix_.format_=session.highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_=M.indptr.astype(np.int32);lp.a_matrix_.index_=M.indices.astype(np.int32);lp.a_matrix_.value_=M.data
    if prepared is None:session._check(h.passModel(lp))
    else:
        h=prepared['h'];columns=prepared['columns'].copy();selected=set(prepared['selected'])
        indices=np.arange(len(upper),dtype=np.int32)
        session._check(h.changeRowsBounds(len(indices),indices,lower,upper))
        indices=np.arange(len(columns),dtype=np.int32)
        session._check(h.changeColsCost(len(indices),indices,cost[columns]))
    attempts=[]
    for iteration in range(max_rounds):
        if time.perf_counter()>=deadline:break
        session._check(h.setOptionValue('time_limit',h.getRunTime()+max(1e-5,deadline-time.perf_counter())))
        t=time.perf_counter();h.run();solve_seconds=time.perf_counter()-t
        status=h.getModelStatus()
        row=dict(round=iteration,n_columns=len(columns),state_columns=len(selected),
            solve_seconds=solve_seconds,iterations=h.getInfo().simplex_iteration_count,message=h.modelStatusToString(status))
        attempts.append(row)
        if status!=session.highspy.HighsModelStatus.kOptimal:break
        solution=h.getSolution();x=np.zeros(p['cursor']);x[columns]=solution.col_value
        recompute_loss_variables(p,x,session.quotes)
        audit=validate_frame(p,upper,cost,x,session.quotes,audit_cache=session.audit_cache)
        cert=global_lower_bound(p,matrix,upper,cost,np.asarray(solution.row_dual))
        gap=audit['actual_objective']-cert['lower_bound']
        row.update(validated=audit['validated'],global_gap=gap,actual_objective=audit['actual_objective'])
        if audit['validated'] and -objective_tolerance<=gap<=objective_tolerance:
            if prepared is not None:prepared.update(columns=columns.copy(),selected=set(selected))
            return dict(success=True,x=x,dual=cert['dual'],audit=audit,global_gap=gap,
                global_lower_bound=cert['lower_bound'],attempts=attempts,request_seconds=time.perf_counter()-start)
        added=set()
        for e,v in p['items'].items():
            lo,hi=p['q_offsets'][e];reduced=cert['reduced_cost'][lo:hi]
            omitted=np.array([i for i in range(hi-lo) if lo+i not in selected],dtype=int)
            if len(omitted):
                # Negative reduced costs identify improving missing columns.
                best=omitted[np.argsort(reduced[omitted])[:8]]
                added.update(lo+best)
        if not added:break
        new=np.array(sorted(added),dtype=np.int32);N=matrix[:,new].tocsc()
        session._check(h.addCols(len(new),cost[new],np.zeros(len(new)),np.full(len(new),np.inf),
            N.nnz,N.indptr.astype(np.int32),N.indices.astype(np.int32),N.data))
        selected.update(added);columns=np.r_[columns,new].astype(np.int32)
    if prepared is not None:prepared.update(columns=columns.copy(),selected=set(selected))
    return dict(success=False,reason='no_global_certificate',attempts=attempts,request_seconds=time.perf_counter()-start)


# Calendar-row generation retains the canonical objective and variable set.
def solve_generated(session, *, previous_dual=None, time_limit=30., max_rounds=30):
    """Keep all variables/market rows; generate only calendar rows.

    All original rows and a global lower bound are checked before acceptance.
    Failure returns diagnostics, never an uncertified candidate as successful.
    This owns a separate backend and does not modify the persistent full basis.
    """
    import highspy
    started=time.perf_counter();deadline=started+time_limit
    p=session.problem;M=session.audit_cache['matrix'];lower=session.audit_cache['lower']
    upper,costs=quote_vectors(p,session.quotes)
    start=p['E'].shape[0]+sum(2*len(v['market_k']) for v in p['items'].values())
    end=start+p['calendar_constraint_count']
    selected=np.r_[np.arange(start),np.arange(end,M.shape[0])].astype(np.int32)
    if previous_dual is not None:
        selected=np.r_[selected,start+np.flatnonzero(np.abs(previous_dual[start:end])>1e-9)].astype(np.int32)
    present=np.zeros(M.shape[0],bool);present[selected]=True
    local=M[selected].tocsc();h=highspy.Highs()
    options=dict(output_flag=False,threads=1,random_seed=0,solver='simplex',simplex_strategy=1,
                 presolve='off',primal_feasibility_tolerance=1e-9 if p.get('mid_penalty') else 1e-7,
                 dual_feasibility_tolerance=1e-9 if p.get('mid_penalty') else 1e-7)
    options.update(NUMERICS_PROFILES[session._profile_name])
    for name,value in options.items():session._check(h.setOptionValue(name,value))
    lp=highspy.HighsLp();lp.num_col_=M.shape[1];lp.num_row_=len(selected)
    lp.col_cost_=costs;lp.col_lower_=np.zeros(M.shape[1]);lp.col_upper_=np.full(M.shape[1],np.inf)
    lp.row_lower_=lower[selected];lp.row_upper_=upper[selected]
    lp.a_matrix_.format_=highspy.MatrixFormat.kColwise;lp.a_matrix_.start_=local.indptr.astype(np.int32)
    lp.a_matrix_.index_=local.indices.astype(np.int32);lp.a_matrix_.value_=local.data
    session._check(h.passModel(lp));attempts=[]
    for iteration in range(max_rounds):
        if time.perf_counter()>=deadline:break
        session._check(h.setOptionValue('time_limit',h.getRunTime()+max(1e-6,deadline-time.perf_counter())))
        t=time.perf_counter();h.run();elapsed=time.perf_counter()-t
        status=h.getModelStatus();row=dict(round=iteration,rows=len(selected),calendar_rows=int(present[start:end].sum()),
            solve_seconds=elapsed,iterations=h.getInfo().simplex_iteration_count,message=h.modelStatusToString(status))
        attempts.append(row)
        if status!=highspy.HighsModelStatus.kOptimal:break
        raw=h.getSolution();x=np.asarray(raw.col_value).copy()
        recompute_loss_variables(p,x,session.quotes)
        violations=M@x-upper
        omitted=np.flatnonzero(~present & (violations>1e-8))
        row.update(violated_omitted_rows=len(omitted),max_omitted_violation=float(np.max(violations[~present],initial=0.)))
        if not len(omitted):
            audit=validate_frame(p,upper,costs,x,session.quotes,audit_cache=session.audit_cache)
            dual=np.zeros(M.shape[0]);dual[selected]=raw.row_dual
            cert=global_lower_bound(p,M,upper,costs,dual)
            gap=audit['actual_objective']-cert['lower_bound']
            row.update(global_objective_gap=gap,validated=audit['validated'])
            if audit['validated'] and abs(gap)<=1e-6:
                return dict(success=True,x=x,dual=cert['dual'],audit=audit,objective_gap=gap,
                    attempts=attempts,request_seconds=time.perf_counter()-started)
            break
        # Add every currently violated omitted row, retaining the native basis.
        new=M[omitted].tocsr()
        session._check(h.addRows(len(omitted),lower[omitted],upper[omitted],new.nnz,
            new.indptr.astype(np.int32),new.indices.astype(np.int32),new.data))
        selected=np.r_[selected,omitted];present[omitted]=True
    return dict(success=False,reason='no_global_certificate',attempts=attempts,
                request_seconds=time.perf_counter()-started)


# Fixed-geometry benchmarks share the maintained solver and validation routines.
def run_paper_benchmark(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for date in args.dates:
        inputs=load_inputs(date,args.eta)
        s=PersistentSanosLP(inputs,time_limit=args.time_limit)
        initial=s.solve()
        row=dict(date=date,eta=args.eta,objective_mode=s.objective_mode,variables=s.problem['cursor'],
                 initial_success=initial['success'],initial_seconds=initial['request_seconds'],
                 initial_message=initial['message'],initial_numerics=initial.get('numerics_attempts'))
        if initial['success']:
            frames=[]
            for key,item in s.problem['items'].items():
                g=item['quotes'].copy()
                g['fit']=item['G']@initial['x'][slice(*s.problem['q_offsets'][key])]
                g['outside']=np.maximum(g.pure_bid-g['fit'],0)+np.maximum(g['fit']-g.pure_ask,0)
                g['outside_spreads']=g.outside/(g.pure_ask-g.pure_bid)
                g['inside']=g.outside<=1e-9;g['core']=core_mask(g)
                frames.append(g)
            fitted=pd.concat(frames,ignore_index=True)
            fitted.to_csv(out/f'{date}_initial_quotes.csv',index=False)
            row.update(initial_coverage=float(fitted.inside.mean()),
                initial_core_coverage=float(fitted.loc[fitted.core,'inside'].mean()),
                initial_max_constraint_violation=initial['max_constraint_violation'])
            repeated=s.solve();row.update(noop_success=repeated['success'],noop_seconds=repeated['request_seconds'],noop_iterations=repeated['iterations'])
            ex=min(s.problem['items'],key=lambda key:abs(s.problem['items'][key]['dte']-30))
            v=s.problem['items'][ex];indices=np.flatnonzero(core_mask(v['quotes']));index=int(indices[len(indices)//2])
            quotes={key:(b.copy(),a.copy()) for key,(b,a) in s.quotes.items()}
            bid,ask=quotes[ex];width=ask[index]-bid[index]
            oldfit=float(v['G'][index]@initial['x'][slice(*s.problem['q_offsets'][ex])])
            bid[index]=max(ask[index],oldfit)+.05*width;ask[index]=bid[index]+width
            row['old_fit_violation']=float(bid[index]-oldfit)
            s.update_quotes(quotes);updated=s.solve()
            row.update(update_success=updated['success'],update_seconds=updated['request_seconds'],
                update_message=updated['message'],update_path=updated['optimization_path'],
                fallback_reason=updated.get('working_set_fallback_reason'),local_attempts=updated.get('working_set_attempts'))
            control=copy.deepcopy(inputs)
            for key,item in control.items():
                item['pure_bid'],item['pure_ask']=quotes[key]
                item['pure_mid']=.5*(quotes[key][0]+quotes[key][1]) if key==ex else item['pure_mid']
            cold=solve_convex_order_lp(control,time_limit=args.time_limit)
            row.update(cold_success=cold['success'],cold_seconds=cold['solve_seconds'])
            if cold['success'] and updated['success']:
                delta=abs(updated['actual_objective']-cold['actual_objective'])
                row.update(actual_objective=updated['actual_objective'],cold_actual_objective=cold['actual_objective'],
                    objective_difference=delta,objective_agrees_1e6=delta<=1e-6,
                    max_constraint_violation=updated['max_constraint_violation'],
                    global_gap=updated.get('global_objective_gap'))
        rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
        print(json.dumps(row),flush=True)


def frames(problem,scenario='broad'):
    """Base, local/broad small moves, then widened spreads; fixed model geometry."""
    updates=[];rng=np.random.default_rng(20261015)
    base={ex:(np.asarray(v['pure_bid']).copy(),np.asarray(v['pure_ask']).copy()) for ex,v in problem['items'].items()}
    candidates=[(ex,i) for ex,v in problem['items'].items() for i in np.flatnonzero(core_mask(v['quotes']))]
    order=rng.permutation(len(candidates)) if scenario=='sparse' else None;counts=[0,1,5,10,25]
    for frame in range(5):
        selected=set(candidates[i] for i in order[:counts[frame]]) if scenario=='sparse' else None
        quotes={}
        for ex,v in problem['items'].items():
            bid,ask=base[ex];spread=ask-bid;core=core_mask(v['quotes'])
            selection=core.copy()
            if scenario=='sparse':selection=np.array([(ex,i) in selected for i in range(len(bid))])
            elif frame==1: selection &= rng.random(len(bid))<.1
            movement=np.zeros(len(bid))
            if frame: movement[selection]=(.05 if scenario=='sparse' or frame%2 else -.05)*spread[selection]
            widen=np.where(core,.025*spread,0) if frame==4 and scenario=='broad' else np.zeros(len(bid))
            # Selected core OTM bids are positive; nevertheless reject invalid
            # synthetic prices instead of clipping and changing spread weights.
            new_bid=bid+movement-widen;new_ask=ask+movement+widen
            if np.any(new_bid<0): raise ValueError('Synthetic bid below zero')
            quotes[ex]=(new_bid,new_ask)
        updates.append((quotes,*quote_vectors(problem,quotes)))
    return updates


def write_binary(path,problem,compiled,updates):
    matrix,lower=compiled['matrix'],compiled['lower']
    with path.open('wb') as f:
        f.write(b'XTXLP001');f.write(struct.pack('<4Q',matrix.shape[1],matrix.shape[0],matrix.nnz,len(updates)))
        for array,dtype in [(matrix.indptr,'<i4'),(matrix.indices,'<i4'),(matrix.data,'<f8'),(lower,'<f8')]:
            f.write(np.asarray(array,dtype=dtype).tobytes())
        for quotes,_,_ in updates:
            upper,costs=native_quote_vectors(problem,compiled,quotes)
            f.write(np.asarray(costs,dtype='<f8').tobytes());f.write(np.asarray(upper,dtype='<f8').tobytes())


def benchmark_summary(records):
    import pandas as pd
    df=pd.DataFrame(records).query('frame > 0')
    counts=df.groupby(['date','method']).agg(all_passed=('success','all'),total_updates=('success','size'))
    timings=df[df.success].groupby(['date','method']).agg(
        median_seconds=('total_seconds','median'),min_seconds=('total_seconds','min'),
        max_seconds=('total_seconds','max'),median_iterations=('iterations','median'),
        successful_updates=('success','size'))
    return counts.join(timings,how='left')


def paired_language_comparison(records):
    import pandas as pd
    df=pd.DataFrame(records)
    cpp=df[(df.method=='cpp_warm') & (df.frame>0) & df.success]
    py=df[(df.method=='python_warm') & (df.frame>0) & df.success]
    paired=cpp.merge(py,on=['date','frame'],suffixes=('_cpp','_py'))
    return paired.groupby('date').agg(paired_frames=('frame','size'),
        cpp_median=('total_seconds_cpp','median'),python_median=('total_seconds_py','median'))


def run_language_benchmark(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    records=[];profiles=[]

    def save(row):
        records.append(row)
        (out/'results.json').write_text(json.dumps(records,indent=2),encoding='utf8')
        import pandas as pd
        pd.DataFrame(records).to_csv(out/'results.csv',index=False)
        print(json.dumps(row),flush=True)

    for date in args.dates:
        eta=.1 if date=='2025-04-01' else .25
        started=time.perf_counter();inputs=load_inputs(date,eta,baseline_variant='baseline');data_seconds=time.perf_counter()-started
        started=time.perf_counter();problem=build_convex_order_problem(inputs,weight_mode='legacy',mid_penalty=0.);build_seconds=time.perf_counter()-started
        updates=frames(problem,args.scenario)
        audit_cache=prepare_validation(problem)
        profile=dict(date=date,eta=eta,data_seconds=data_seconds,build_seconds=build_seconds,
                     kernel_seconds=problem['kernel_seconds'],assembly_seconds=problem['assembly_seconds'],
                     n_vars=problem['cursor'],n_quotes=sum(len(v['market_k']) for v in problem['items'].values()))
        profiles.append(profile);print('PROFILE '+json.dumps(profile),flush=True)
        if not args.skip_scipy:
            t=time.perf_counter()
            res=linprog(problem['c'],A_ub=problem['A'],b_ub=problem['b'],A_eq=problem['E'],
                b_eq=np.ones(problem['E'].shape[0]),bounds=(0,None),method='highs-ds',
                options=dict(presolve=False,time_limit=args.time_limit,
                             primal_feasibility_tolerance=1e-7,dual_feasibility_tolerance=1e-7))
            row=dict(date=date,eta=eta,method='scipy_cold',frame=0,solve_seconds=time.perf_counter()-t,
                     success=bool(res.success),message=res.message,iterations=int(res.nit),
                     total_seconds=time.perf_counter()-t)
            if res.success:
                row.update(validate_frame(problem,updates[0][1],updates[0][2],res.x,updates[0][0],audit_cache=audit_cache));row['objective']=float(res.fun)
                row['success']=row['success'] and row['validated']
            save(row)
        compiled=compile_native_model(problem)
        path=out/f'{date}.bin';write_binary(path,problem,compiled,updates)
        for mode in ['cold','warm']:
            solpath=out/f'{date}_cpp_{mode}.solutions'
            completed=subprocess.run([str(args.cpp_exe),str(args.dll),str(path),mode,str(solpath),str(args.time_limit)],
                                     capture_output=True,text=True)
            if completed.returncode:
                raise RuntimeError(f'C++ benchmark failed: {completed.stderr}')
            (out/f'{date}_cpp_{mode}.jsonl').write_text(completed.stdout,encoding='utf8')
            cpp_rows=[json.loads(line) for line in completed.stdout.splitlines() if line.startswith('{')]
            if len(cpp_rows)!=len(updates):raise RuntimeError('Incomplete C++ result sequence')
            with solpath.open('rb') as f:
                for frame,row in enumerate(cpp_rows):
                    status,objective=struct.unpack('<id',f.read(12))
                    x=np.frombuffer(f.read(problem['cursor']*8),dtype='<f8').copy()*compiled['column_scale']
                    row.update(method=row.pop('mode'),date=date,eta=eta,
                               objective=objective*compiled['objective_scale'],success=status==7)
                    if row['success']:
                        row.update(validate_frame(problem,updates[frame][1],updates[frame][2],x,updates[frame][0],audit_cache=audit_cache))
                        row['success']=row['success'] and row['validated']
                    save(row)
        for method in ['python_cold','python_reuse_no_basis','python_warm']:
            session=None
            for frame,(quotes,upper,costs) in enumerate(updates):
                started=time.perf_counter()
                if session is None or method=='python_cold':
                    session=PersistentSanosLP(inputs,problem=problem,compiled=compiled,time_limit=args.time_limit,numerics_profile='legacy')
                session.update_quotes(quotes)
                setup_seconds=time.perf_counter()-started
                row=session.solve(discard_basis=method=='python_reuse_no_basis',validate=False)
                row.update(total_seconds=time.perf_counter()-started,setup_seconds=setup_seconds,
                           method=method,date=date,eta=eta,frame=frame,version=session.h.version())
                x=row.pop('x')
                if row['success']:
                    row.update(validate_frame(problem,upper,costs,x,quotes,audit_cache=audit_cache));row['success']=row['success'] and row['validated']
                save(row)
        (out/'profiles.json').write_text(json.dumps(profiles,indent=2),encoding='utf8')
    # Same-version cold solutions anchor every objective comparison. Differences
    # beyond explicit tolerance are failures, not speed improvements.
    for row in records:
        controls=[r for r in records if r['date']==row['date'] and r['frame']==row['frame'] and r['method']=='cpp_cold' and r['success']]
        if row.get('success') and controls:
            delta=abs(row['actual_outside_objective']-controls[0]['actual_outside_objective'])
            row['outside_objective_difference_to_cpp_cold']=delta
            row['objective_agrees']=bool(delta<=1e-6)
            row['success']=row['success'] and row['objective_agrees']
    (out/'results.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    import pandas as pd
    pd.DataFrame(records).to_csv(out/'results.csv',index=False)
    summary=benchmark_summary(records)
    summary.to_csv(out/'summary.csv');print(summary.to_string(),flush=True)
    paired_language_comparison(records).to_csv(out/'paired_language_comparison.csv')
    metadata=dict(synthetic_updates=True,frames_per_date=5,seed=20261015,
                  solver='dual simplex',presolve='off',threads=1,primal_dual_tolerance=1e-7,
                  independent_feasibility_tolerance=2e-7,objective_comparison_tolerance=1e-6,
                  fixed_geometry=True,exclude_data_io_and_validation_from_total=True)
    metadata.update(native_otm_and_spread_scaling=False,scipy_uses_original_unscaled_form=True)
    metadata.update(scenario=args.scenario,sparse_quote_counts=[0,1,5,10,25] if args.scenario=='sparse' else None)
    metadata.update(sparse_incremental_dirty_quotes=[0,1,4,5,15] if args.scenario=='sparse' else None)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    if not all(row['success'] for row in records):raise SystemExit('One or more benchmarks failed validation or optimality.')
