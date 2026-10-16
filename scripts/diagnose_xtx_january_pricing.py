"""January identical-basis simplex pricing controls and developer trace."""
import _bootstrap  # project paths for direct script execution
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_xtx_paper_latency import clone_session,summarize
from xtx_inputs import load_inputs,core_mask
from sanos_incremental import PersistentSanosLP


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--update-limits',action='store_true')
    args=parser.parse_args()
    option='simplex_update_limit' if args.update_limits else 'simplex_dual_edge_weight_strategy'
    out=Path('data/outputs/january_update_limit' if args.update_limits else 'data/outputs/january_pricing');out.mkdir(parents=True,exist_ok=True)
    inputs=load_inputs('2025-01-02',.25)
    parent=PersistentSanosLP(inputs,time_limit=30);initial=parent.solve()
    if not initial['success']:raise RuntimeError('Initial solve failed')
    p=parent.problem;ex=min(p['items'],key=lambda e:abs(p['items'][e]['dte']-30));v=p['items'][ex]
    idxs=np.flatnonzero(core_mask(v['quotes']));idx=int(idxs[len(idxs)//2])
    quotes={e:(b.copy(),a.copy()) for e,(b,a) in parent.quotes.items()}
    bid,ask=quotes[ex];width=ask[idx]-bid[idx]
    oldfit=float(v['G'][idx]@initial['x'][slice(*p['q_offsets'][ex])])
    bid[idx]=max(ask[idx],oldfit)+.05*width;ask[idx]=bid[idx]+width
    rows=[]
    for repeat in range(3):
        variants=([('default_5000',5000),('limit_50',50),('limit_100',100),('limit_250',250)]
                  if args.update_limits else [('auto',-1),('devex',1),('steepest',2)])
        if repeat%2:variants=variants[::-1]
        for name,value in variants:
            s=clone_session(inputs,parent,parent.h.getBasis(),working_set=False)
            if not s.solve()['success']:raise RuntimeError('Prewarm failed')
            s._check(s.h.setOptionValue(option,value));s.time_limit=5
            s.update_quotes(quotes);r=s.solve()
            effective=s.h.getOptionValue(option)[1]
            rows.append(dict(case=name,repeat=repeat,requested_option=value,effective_option=effective,**summarize(r)))
            print(json.dumps({k:rows[-1].get(k) for k in ['case','repeat','success','request_seconds','iterations','actual_objective','effective_option']}),flush=True)
            (out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    # Separate untimed developer trace: do not mix logging IO into benchmarks.
    s=clone_session(inputs,parent,parent.h.getBasis(),working_set=False)
    if not s.solve()['success']:raise RuntimeError('Trace prewarm failed')
    log=out/'warm_update.log'
    for name,value in [('output_flag',True),('log_to_console',False),('log_file',str(log.resolve())),('log_dev_level',1)]:
        s._check(s.h.setOptionValue(name,value))
    s.update_quotes(quotes);r=s.solve()
    (out/'trace_summary.json').write_text(json.dumps(summarize(r),indent=2),encoding='utf8')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
