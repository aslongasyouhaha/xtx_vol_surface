"""Equivalent next-knot versus union-knot LP, audited with full original rows."""
import _bootstrap  # project paths for direct script execution
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_xtx_paper_latency import clone_session,summarize
from xtx_inputs import load_inputs,core_mask
from sanos_core import build_convex_order_problem
from sanos_incremental import PersistentSanosLP,quote_vectors,validate_frame


def main():
    out=Path('data/outputs/calendar_reduction');out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for date in ['2025-01-02','2025-04-01','2025-06-02']:
        inputs=load_inputs(date,.1 if date=='2025-04-01' else .25)
        sessions={};initials={};result=dict(date=date,cases={})
        for mode in ['union','next']:
            started=time.perf_counter()
            problem=build_convex_order_problem(inputs,calendar_grid=mode)
            session=PersistentSanosLP(inputs,problem=problem,working_set=False,time_limit=30)
            build=time.perf_counter()-started;initial=session.solve()
            if not initial['success']:raise RuntimeError(initial['message'])
            sessions[mode]=session;initials[mode]=initial
            result['cases'][mode]=dict(construction_seconds=build,rows=session.matrix.shape[0],
                nnz=session.matrix.nnz,calendar_rows=problem['calendar_constraint_count'],
                initial=summarize(initial),updates=[])
        original=sessions['union'];p=original.problem
        ex=min(p['items'],key=lambda e:abs(p['items'][e]['dte']-30));v=p['items'][ex]
        idxs=np.flatnonzero(core_mask(v['quotes']));idx=int(idxs[len(idxs)//2])
        fits=[float(s.problem['items'][ex]['G'][idx]@initials[mode]['x'][slice(*s.problem['q_offsets'][ex])])
              for mode,s in sessions.items()]
        quotes={e:(b.copy(),a.copy()) for e,(b,a) in original.quotes.items()}
        bid,ask=quotes[ex];width=ask[idx]-bid[idx]
        bid[idx]=max(ask[idx],*fits)+.05*width;ask[idx]=bid[idx]+width
        upper,cost=quote_vectors(p,quotes)
        for repeat in range(3):
            for mode in (['union','next'] if repeat%2==0 else ['next','union']):
                parent=sessions[mode]
                session=clone_session(inputs,parent,parent.h.getBasis(),working_set=False)
                prewarm=session.solve()
                if not prewarm['success']:raise RuntimeError('Prewarm failed')
                session.update_quotes(quotes);update=session.solve()
                if not update['success']:raise RuntimeError('Update failed')
                audit=validate_frame(p,upper,cost,update['x'],quotes,audit_cache=original.audit_cache)
                if not audit['validated']:raise RuntimeError('Reduced LP failed ORIGINAL constraints')
                result['cases'][mode]['updates'].append(dict(repeat=repeat,**summarize(update),
                    original_audit_max_violation=audit['max_constraint_violation'],
                    original_audit_convex_min=audit['convex_order_min']))
        objectives=[]
        for case in result['cases'].values():
            case['median_update_seconds']=float(np.median([r['request_seconds'] for r in case['updates']]))
            objectives.extend(r['actual_objective'] for r in case['updates'])
        result['objective_range']=max(objectives)-min(objectives)
        result['all_objectives_agree_1e6']=result['objective_range']<=1e-6
        rows.append(result);(out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
        print(json.dumps(dict(date=date,objective_range=result['objective_range'],
            cases={name:{k:r[k] for k in ['rows','nnz','calendar_rows','median_update_seconds']} for name,r in result['cases'].items()})),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
