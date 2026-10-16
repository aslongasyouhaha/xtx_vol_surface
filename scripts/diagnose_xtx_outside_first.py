"""Outside-loss first, accepted ONLY with an original-paper global gap bound."""
import _bootstrap  # project paths for direct script execution
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_xtx_paper_latency import clone_session,summarize
from xtx_inputs import load_inputs,core_mask
from sanos_incremental import PersistentSanosLP,quote_vectors,recompute_loss_variables,validate_frame
from sanos_working_set import global_lower_bound


def main():
    out=Path('data/outputs/outside_first');out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for date in ['2025-01-02','2025-04-01','2025-06-02']:
        inputs=load_inputs(date,.1 if date=='2025-04-01' else .25)
        parent=PersistentSanosLP(inputs,time_limit=30,working_set=False);initial=parent.solve()
        if not initial['success']:raise RuntimeError('Initial solve failed')
        p=parent.problem;ex=min(p['items'],key=lambda e:abs(p['items'][e]['dte']-30));v=p['items'][ex]
        indices=np.flatnonzero(core_mask(v['quotes']));idx=int(indices[len(indices)//2])
        quotes={e:(b.copy(),a.copy()) for e,(b,a) in parent.quotes.items()}
        bid,ask=quotes[ex];width=ask[idx]-bid[idx]
        oldfit=float(v['G'][idx]@initial['x'][slice(*p['q_offsets'][ex])])
        bid[idx]=max(ask[idx],oldfit)+.05*width;ask[idx]=bid[idx]+width
        upper,cost=quote_vectors(p,quotes)
        mid=np.concatenate([np.arange(a,b,dtype=np.int32) for a,b in p['mid_offsets'].values()])
        records=[]
        for repeat in range(3):
            for mode in (['full','outside_first'] if repeat%2==0 else ['outside_first','full']):
                s=clone_session(inputs,parent,parent.h.getBasis(),working_set=False)
                if not s.solve()['success']:raise RuntimeError('Prewarm failed')
                s.update_quotes(quotes)
                if mode=='full':
                    r=s.solve();record=dict(mode=mode,repeat=repeat,**summarize(r))
                else:
                    start=time.perf_counter()
                    s._check(s.h.changeColsCost(len(mid),mid,np.zeros(len(mid))))
                    s._check(s.h.setOptionValue('time_limit',s.h.getRunTime()+10))
                    t=time.perf_counter();s.h.run();solve=time.perf_counter()-t
                    status=s.h.getModelStatus();cert=None;audit=None
                    if status==s.highspy.HighsModelStatus.kOptimal:
                        x=np.asarray(s.h.getSolution().col_value).copy()
                        recompute_loss_variables(p,x,quotes)
                        audit=validate_frame(p,upper,cost,x,quotes,audit_cache=parent.audit_cache)
                        t=time.perf_counter()
                        cert=global_lower_bound(p,parent.audit_cache['matrix'],upper,cost,np.asarray(s.h.getSolution().row_dual))
                        certificate_seconds=time.perf_counter()-t
                        gap=audit['actual_objective']-cert['lower_bound']
                        accepted=audit['validated'] and -1e-6<=gap<=1e-6
                    else:accepted=False;gap=None;certificate_seconds=0.
                    record=dict(mode=mode,repeat=repeat,certified=bool(accepted),
                        candidate_seconds=time.perf_counter()-start,first_solve_seconds=solve,
                        certificate_seconds=certificate_seconds,gap=gap,
                        first_iterations=s.h.getInfo().simplex_iteration_count)
                    if accepted:
                        record.update(success=True,request_seconds=time.perf_counter()-start,
                            actual_objective=audit['actual_objective'],max_constraint_violation=audit['max_constraint_violation'])
                    else:
                        # Restore exact original cost vector and retain its basis.
                        s._check(s.h.changeColsCost(len(mid),mid,cost[mid]))
                        s.time_limit=max(.001,30-(time.perf_counter()-start))
                        r=s.solve()
                        record.update(fallback=summarize(r),success=r['success'],
                            request_seconds=time.perf_counter()-start,actual_objective=r.get('actual_objective'))
                records.append(record)
        full=[r['actual_objective'] for r in records if r['mode']=='full']
        row=dict(date=date,records=records,medians={mode:float(np.median([r['request_seconds'] for r in records if r['mode']==mode])) for mode in ['full','outside_first']},
            certified_candidates=sum(r.get('certified',False) for r in records),
            full_objective=full[0])
        rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
        print(json.dumps({k:row[k] for k in ['date','medians','certified_candidates']}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
