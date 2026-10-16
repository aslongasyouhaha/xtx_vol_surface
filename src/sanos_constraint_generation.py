"""Experimental row generation for the unchanged canonical SANOS LP."""
import time
import numpy as np
from sanos_incremental import quote_vectors, validate_frame, NUMERICS_PROFILES,recompute_loss_variables
from sanos_working_set import global_lower_bound


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
