"""Equivalent OTM price rows versus call rows, with original-unit audits."""
import _bootstrap  # project paths for direct script execution
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_xtx_paper_latency import clone_session,summarize
from xtx_inputs import load_inputs,core_mask
from sanos_core import build_convex_order_problem
from sanos_incremental import PersistentSanosLP,compile_native_model,quote_vectors,validate_frame


def main():
    out=Path('data/outputs/otm_numerics');out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for date in ['2025-01-02','2025-04-01','2025-06-02']:
        inputs=load_inputs(date,.1 if date=='2025-04-01' else .25)
        p=build_convex_order_problem(inputs);sessions={};initials={};row=dict(date=date,cases={})
        for mode in ['none','otm']:
            start=time.perf_counter()
            s=PersistentSanosLP(inputs,problem=p,compiled=compile_native_model(p,scaling=mode),working_set=False,time_limit=30)
            construction=time.perf_counter()-start;initial=s.solve()
            if not initial['success']:raise RuntimeError(f'{date} {mode} {initial["message"]}')
            sessions[mode]=s;initials[mode]=initial
            row['cases'][mode]=dict(construction_seconds=construction,initial=summarize(initial),nnz=s.matrix.nnz,updates=[])
        ex=min(p['items'],key=lambda e:abs(p['items'][e]['dte']-30));v=p['items'][ex]
        indices=np.flatnonzero(core_mask(v['quotes']));idx=int(indices[len(indices)//2])
        fits=[float(v['G'][idx]@r['x'][slice(*p['q_offsets'][ex])]) for r in initials.values()]
        quotes={e:(b.copy(),a.copy()) for e,(b,a) in sessions['none'].quotes.items()}
        bid,ask=quotes[ex];width=ask[idx]-bid[idx]
        bid[idx]=max(ask[idx],*fits)+.05*width;ask[idx]=bid[idx]+width
        upper,costs=quote_vectors(p,quotes)
        for repeat in range(3):
            for mode in (['none','otm'] if repeat%2==0 else ['otm','none']):
                parent=sessions[mode]
                s=clone_session(inputs,parent,parent.h.getBasis(),working_set=False)
                prepared=s.solve()
                if not prepared['success']:raise RuntimeError('Prewarm failed')
                s.update_quotes(quotes);r=s.solve()
                if not r['success']:raise RuntimeError('Update failed')
                audit=validate_frame(p,upper,costs,r['x'],quotes,audit_cache=sessions['none'].audit_cache)
                if not audit['validated']:raise RuntimeError('Original constraint audit failed')
                row['cases'][mode]['updates'].append(dict(repeat=repeat,**summarize(r),
                    original_audit_max_violation=audit['max_constraint_violation'],
                    original_audit_convex_min=audit['convex_order_min']))
        objectives=[]
        for case in row['cases'].values():
            case['median_update_seconds']=float(np.median([r['request_seconds'] for r in case['updates']]))
            objectives.extend(r['actual_objective'] for r in case['updates'])
        row['objective_range']=max(objectives)-min(objectives)
        row['all_objectives_agree_1e6']=row['objective_range']<=1e-6
        rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
        print(json.dumps(dict(date=date,objective_range=row['objective_range'],
            cases={m:dict(initial_seconds=c['initial']['request_seconds'],initial_attempts=len(c['initial']['numerics_attempts']),
                median_update_seconds=c['median_update_seconds'],update_iterations=c['updates'][0]['iterations'])
                for m,c in row['cases'].items()})),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
