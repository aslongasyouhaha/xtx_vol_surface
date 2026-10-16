"""Same paper LP and quote update: native simplex, native IPM, Clarabel LP.

No quadratic term is added. Every returned candidate is audited against the
original matrices and compared on reconstructed original-paper objective.
"""
import _bootstrap  # project paths for direct script execution
import json
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from diagnose_xtx_paper_latency import clone_session
from xtx_inputs import load_inputs,core_mask
from sanos_incremental import PersistentSanosLP,quote_vectors,validate_frame


def main():
    import clarabel
    out=Path('data/outputs/lp_backends');out.mkdir(parents=True,exist_ok=True);rows=[]
    for date in ['2025-01-02','2025-04-01','2025-06-02']:
        inputs=load_inputs(date,.1 if date=='2025-04-01' else .25)
        parent=PersistentSanosLP(inputs,time_limit=30);initial=parent.solve()
        if not initial['success']:raise RuntimeError('Initial simplex solve failed')
        p=parent.problem;ex=min(p['items'],key=lambda e:abs(p['items'][e]['dte']-30));v=p['items'][ex]
        indices=np.flatnonzero(core_mask(v['quotes']));idx=int(indices[len(indices)//2])
        quotes={e:(b.copy(),a.copy()) for e,(b,a) in parent.quotes.items()}
        b,a=quotes[ex];width=a[idx]-b[idx]
        b[idx]=max(a[idx],float(v['G'][idx]@initial['x'][slice(*p['q_offsets'][ex])]))+.05*width;a[idx]=b[idx]+width
        upper,cost=quote_vectors(p,quotes);cases=[]
        for method in ['simplex','ipm']:
            s=clone_session(inputs,parent,parent.h.getBasis(),working_set=False)
            if not s.solve()['success']:raise RuntimeError('Prewarm failed')
            s.update_quotes(quotes)
            s._check(s.h.setOptionValue('solver',method));s._check(s.h.setOptionValue('presolve','on' if method=='ipm' else 'off'))
            if method=='ipm':s._check(s.h.setOptionValue('ipm_optimality_tolerance',1e-10))
            s._check(s.h.setOptionValue('time_limit',s.h.getRunTime()+10))
            t=time.perf_counter();s.h.run();elapsed=time.perf_counter()-t
            status=s.h.getModelStatus();info=s.h.getInfo()
            row=dict(backend='highs_'+method,status=s.h.modelStatusToString(status),solve_seconds=elapsed,
                simplex_iterations=info.simplex_iteration_count,ipm_iterations=info.ipm_iteration_count)
            if status==s.highspy.HighsModelStatus.kOptimal:
                x=np.asarray(s.h.getSolution().col_value)
                row.update(validate_frame(p,upper,cost,x,quotes,audit_cache=parent.audit_cache))
            cases.append(row)
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
        settings.time_limit=10.;settings.max_iter=100
        settings.tol_feas=1e-10;settings.tol_gap_abs=1e-8;settings.tol_gap_rel=1e-10
        A=sparse.vstack([p['E'],p['A'],-sparse.eye(p['cursor'])],format='csc')
        b0=np.r_[np.ones(p['E'].shape[0]),p['b'],np.zeros(p['cursor'])]
        t=time.perf_counter()
        solver=clarabel.DefaultSolver(sparse.csc_matrix((p['cursor'],p['cursor'])),p['c'],A,b0,
            [clarabel.ZeroConeT(p['E'].shape[0]),clarabel.NonnegativeConeT(p['A'].shape[0]+p['cursor'])],settings)
        build=time.perf_counter()-t;t=time.perf_counter();cold=solver.solve();cold_seconds=time.perf_counter()-t
        solver.update(q=cost,b=np.r_[upper,np.zeros(p['cursor'])])
        t=time.perf_counter();raw=solver.solve();elapsed=time.perf_counter()-t
        row=dict(backend='clarabel_lp',status=str(raw.status),solve_seconds=elapsed,iterations=raw.iterations,
            construction_seconds=build,cold_seconds=cold_seconds,cold_status=str(cold.status))
        if str(raw.status) in {'Solved','AlmostSolved'}:
            row.update(validate_frame(p,upper,cost,np.asarray(raw.x),quotes,audit_cache=parent.audit_cache))
        cases.append(row)
        reference=cases[0].get('actual_objective')
        for row in cases:
            row['original_objective_difference']=abs(row['actual_objective']-reference) if row.get('validated') and reference is not None else None
            row['accepted']=bool(row.get('validated') and row['original_objective_difference']<=1e-6)
        result=dict(date=date,cases=cases);rows.append(result)
        (out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
