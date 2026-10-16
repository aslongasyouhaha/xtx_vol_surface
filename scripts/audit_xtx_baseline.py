"""Evidence inventory for the basic paper LP; generalized experiments excluded."""
import _bootstrap  # project paths for direct script execution
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
DEPS=ROOT/'data/outputs/native_deps'
if (DEPS/'highspy').exists():sys.path.insert(0,str(DEPS))
from xtx_inputs import load_inputs,core_mask
from sanos_incremental import PersistentSanosLP
from sanos_core import call_kernel
from sanos_surface import audit_surface


def january_smoothing(out):
    rows=[]
    for eta,left in [(0.,None),(.1,None),(.25,.2),(.25,.01)]:
        inputs=load_inputs('2025-01-02',eta);start=time.perf_counter()
        if left is not None:
            for item in inputs.values():
                item['anchor_k']=np.unique(np.r_[left,item['anchor_k']])
                item['G']=call_kernel(item['anchor_k'],item['market_k'],item['smooth_variance'])
        session=PersistentSanosLP(inputs,time_limit=60)
        result=session.solve()
        row=dict(date='2025-01-02',eta=eta,added_left_state=left,success=result['success'],
            setup_plus_request_seconds=time.perf_counter()-start)
        if result['success']:
            surface_audit=audit_surface(session.surface())
            row['surface_audit']=surface_audit
            row['success']=surface_audit['passed']
            arrays={}
            frames=[]
            for ex,item in session.problem['items'].items():
                prefix=str(ex.date())
                arrays[prefix+'__anchors']=item['anchor_k']
                arrays[prefix+'__q']=result['x'][slice(*session.problem['q_offsets'][ex])]
                arrays[prefix+'__smooth_variance']=np.asarray(item['smooth_variance'])
                arrays[prefix+'__T']=np.asarray(item.get('T',item['dte']/365))
                g=item['quotes'].copy();g['fit']=item['G']@result['x'][slice(*session.problem['q_offsets'][ex])]
                g['outside']=np.maximum(g.pure_bid-g['fit'],0)+np.maximum(g['fit']-g.pure_ask,0)
                g['outside_spreads']=g.outside/(g.pure_ask-g.pure_bid)
                g['inside']=g.outside<=1e-9;g['core']=core_mask(g);frames.append(g)
            q=pd.concat(frames,ignore_index=True);q.to_csv(out/f'january_eta_{eta}_left_{left}_quotes.csv',index=False)
            np.savez_compressed(out/f'january_eta_{eta}_left_{left}_model.npz',**arrays)
            row.update(coverage=float(q.inside.mean()),core_coverage=float(q.loc[q.core,'inside'].mean()),
                misses=int((~q.inside).sum()),max_outside_spreads=float(q.outside_spreads.max()),
                objective=result['actual_objective'],max_constraint_violation=result['max_constraint_violation'])
        rows.append(row);print(json.dumps(row),flush=True)
    (out/'january_smoothing.json').write_text(json.dumps(rows,indent=2),encoding='utf8')


def main():
    out=ROOT/'data/outputs/baseline_audit';out.mkdir(parents=True,exist_ok=True)
    january_smoothing(out)
    old=json.loads((ROOT/'data/outputs/paper_incremental/results.json').read_text(encoding='utf8'))
    april=json.loads((out/'april_eta01/results.json').read_text(encoding='utf8'))[0]
    rows=[]
    for date,eta in [('2025-01-02',.25),('2025-04-01',.1),('2025-06-02',.25)]:
        if date=='2025-04-01':
            q=pd.read_csv(out/f'april_eta01/{date}_initial_quotes.csv');benchmark=april
            provenance=str(out/f'april_eta01/{date}_initial_quotes.csv')
        else:
            path=ROOT/f'data/outputs/baseline_paper_gap001/{date}/quotes.csv'
            q=pd.read_csv(path);benchmark=next(x for x in old if x['date']==date);provenance=str(path)
        # Cached quote frames may carry diagnostics from a different eta.
        q['outside_spreads']=q.outside/(q.pure_ask-q.pure_bid)
        misses=q.loc[~q.inside].copy();misses.to_csv(out/f'{date}_misses.csv',index=False)
        core=q.loc[q.core]
        row=dict(date=date,eta=eta,quotes=len(q),expiries=int(q.exdate.nunique()),
            coverage=float(q.inside.mean()),core_quotes=len(core),core_coverage=float(core.inside.mean()),
            misses=len(misses),max_outside_spreads=float(q.outside_spreads.max()),
            native_first_request_seconds=benchmark['initial_seconds'],native_noop_seconds=benchmark['noop_seconds'],
            native_binding_update_seconds=benchmark['update_seconds'],update_path=benchmark['update_path'],
            update_cold_objective_difference=benchmark['objective_difference'],quotes_source=provenance)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
    settings=dict(paper_aligned=['OTM puts converted by put-call parity; OTM calls on right',
        'EOD volume >= 100; normalized vega/sqrt(T) >= 0.001; >=20 quotes per expiry',
        'positive lognormal call mixtures with latent discrete convex order',
        'inverse bid-ask spread weights; midpoint penalty 1e-8',
        'ATM bid-IV variance backbone; default variance factor eta=0.25'],
        engineering_choices=['1--120 DTE; normalized strike k>=0.45',
        'gap-fill max spacing 0.01; model support determined before volume selection',
        'shared exterior anchor states; parity-based forward and yield curve handling',
        'isotonic variance correction where needed',
        'April eta=0.1 is a retrospective diagnostic choice, not an automated policy',
        'core abs OTM forward delta >=0.1 is a diagnostic region, not agreed quoting policy'],
        tolerances=dict(quote_coverage_pure_price=1e-9,original_constraint_audit=2e-7,
                        local_update_absolute_objective_gap=1e-6),
        performance_scope='Frozen geometry; EOD snapshots; one synthetic binding quote update per date. '
            'Native request timings exclude initial model construction and data preparation. '
            'Three dates do not establish intraday latency or robustness.',
        outstanding=['Explain/decide wing handling using measured price errors, not volume alone',
            'Broader date validation and actual quote-stream replay',
            'Rule for eta selection without looking at final fit',
            'Explicit rebuild policy when expiries, anchors or variance geometry change'])
    (out/'settings_and_limits.json').write_text(json.dumps(settings,indent=2),encoding='utf8')
    print(pd.DataFrame(rows).drop(columns='quotes_source').to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
