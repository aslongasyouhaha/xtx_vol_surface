"""All-expiry active-state LP with unrestricted original-objective certification.

No expiry is frozen. All quote-loss variables and all original constraints stay
present. Omitted state columns are priced against the complete original grid.
Only a feasible candidate with global absolute gap <=1e-6 can be accepted.
"""
import time
import numpy as np
from sanos_incremental import quote_vectors,validate_frame,recompute_loss_variables,NUMERICS_PROFILES
from sanos_working_set import global_lower_bound


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
