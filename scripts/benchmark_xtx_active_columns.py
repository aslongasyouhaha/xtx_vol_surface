"""Same update, full warm solve versus globally certified active-state columns."""
import _bootstrap  # project paths for direct script execution
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_xtx_paper_latency import clone_session,summarize
from xtx_inputs import load_inputs,core_mask
from sanos_incremental import PersistentSanosLP
from sanos_active_columns import active_state_update,prepare_active_model


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepared',action='store_true');args=parser.parse_args()
    out=Path('data/outputs/active_columns_prepared' if args.prepared else 'data/outputs/active_columns');out.mkdir(parents=True,exist_ok=True);rows=[]
    for date in ['2025-01-02','2025-04-01','2025-06-02']:
        inputs=load_inputs(date,.1 if date=='2025-04-01' else .25)
        parent=PersistentSanosLP(inputs,time_limit=30);initial=parent.solve()
        if not initial['success']:raise RuntimeError('Initial failed')
        p=parent.problem;ex=min(p['items'],key=lambda e:abs(p['items'][e]['dte']-30));v=p['items'][ex]
        ii=np.flatnonzero(core_mask(v['quotes']));idx=int(ii[len(ii)//2])
        quotes={e:(b.copy(),a.copy()) for e,(b,a) in parent.quotes.items()}
        b,a=quotes[ex];width=a[idx]-b[idx]
        oldfit=float(v['G'][idx]@initial['x'][slice(*p['q_offsets'][ex])])
        b[idx]=max(a[idx],oldfit)+.05*width;a[idx]=b[idx]+width
        records=[]
        for repeat in range(3):
            for mode in (['full','active_columns'] if repeat%2==0 else ['active_columns','full']):
                s=clone_session(inputs,parent,parent.h.getBasis(),working_set=False)
                if not s.solve()['success']:raise RuntimeError('Prewarm failed')
                prepare_start=time.perf_counter()
                prepared=prepare_active_model(s,s._accepted_x) if mode=='active_columns' and args.prepared else None
                prepare_seconds=time.perf_counter()-prepare_start
                s.update_quotes(quotes)
                if mode=='full':r=s.solve();record=dict(mode=mode,repeat=repeat,**summarize(r))
                else:
                    start=time.perf_counter();candidate=active_state_update(s,s._accepted_x,budget=1.,prepared=prepared)
                    record=dict(mode=mode,repeat=repeat,certified=candidate['success'],
                        candidate_seconds=candidate['request_seconds'],attempts=candidate['attempts'],prepare_seconds=prepare_seconds,
                        prewarm_iterations=prepared['prewarm_iterations'] if prepared else None)
                    if candidate['success']:
                        record.update(success=True,request_seconds=time.perf_counter()-start,
                            global_gap=candidate['global_gap'],actual_objective=candidate['audit']['actual_objective'],
                            max_constraint_violation=candidate['audit']['max_constraint_violation'])
                    else:
                        s.time_limit=max(.001,30-(time.perf_counter()-start));r=s.solve()
                        record.update(success=r['success'],request_seconds=time.perf_counter()-start,
                            actual_objective=r.get('actual_objective'),fallback=summarize(r))
                records.append(record)
        row=dict(date=date,records=records,certified_candidates=sum(r.get('certified',False) for r in records),
            medians={m:float(np.median([r['request_seconds'] for r in records if r['mode']==m])) for m in ['full','active_columns']})
        rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
        print(json.dumps({k:row[k] for k in ['date','certified_candidates','medians']}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
