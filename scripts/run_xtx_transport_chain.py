"""Sequential state-dependent martingale fits, not independent fixed-prefix tests."""
import _bootstrap  # project paths for direct script execution
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from xtx_inputs import load_inputs,core_mask
from sanos_martingale_transport import fit_state_dependent_transition,calls
from sanos_generalized_grid import GridProduct


def fit_chain(items,output=None):
    ordered=sorted(items.items(),key=lambda x:x[1]['dte'])
    # Same broad boundaries used by the previously audited generalized grid.
    reference=GridProduct(16385).states
    states=np.array([reference[0],1.,reference[-1]]);p=np.array([0.,1.,0.])
    previous_strikes=[];rows=[]
    for step,(e,it) in enumerate(ordered):
        start=time.perf_counter()
        prior=calls(states,p,it['market_k'])
        jensen=np.maximum(prior-it['pure_ask'],0)
        r=fit_state_dependent_transition(states,p,it,preserve_strikes=previous_strikes,time_limit=30)
        if not r['validated']:raise ValueError(f'Transition audit failed at {e}: {r["errors"]}')
        core=core_mask(it['quotes']) if 'quotes' in it else np.ones(len(prior),bool)
        outside=np.maximum.reduce([it['pure_bid']-r['fit'],r['fit']-it['pure_ask'],np.zeros(len(prior))])
        inside=outside<=1e-9
        row=dict(exdate=str(e),dte=int(it['dte']),n_quotes=len(prior),n_inside=int(inside.sum()),
            n_core=int(core.sum()),n_core_inside=int(inside[core].sum()),
            coverage=float(inside.mean()),core_coverage=float(inside[core].mean()) if core.any() else None,
            outside_objective=r['outside_objective'],objective=r['objective'],
            jensen_blocked=int((jensen>1e-9).sum()),core_jensen_blocked=int(((jensen>1e-9)&core).sum()),
            jensen_loss_lower_bound=float(np.sum(jensen/(it['pure_ask']-it['pure_bid']))),
            max_audit_error=max(r['errors'].values()),seconds=time.perf_counter()-start)
        rows.append(row)
        if output:
            output.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(output/f'transition_{step}.npz',original_states=states,original_weights=p,
                grid=r['grid'],source=r['source'],target=r['target'],Q=r['Q'],
                market_k=it['market_k'],bid=it['pure_bid'],ask=it['pure_ask'],fit=r['fit'])
            sparse.save_npz(output/f'projection_{step}.npz',r['projection'])
            pd.DataFrame(rows).to_csv(output/'per_expiry.csv',index=False)
            (output/f'audit_{step}.json').write_text(json.dumps(r['errors'],indent=2),encoding='utf8')
        states,p=r['grid'],r['target'];previous_strikes=it['market_k']
    f=pd.DataFrame(rows)
    return dict(validated=True,fit_target_met=bool(f.n_inside.sum()==f.n_quotes.sum()),
        expiries=len(f),coverage=float(f.n_inside.sum()/f.n_quotes.sum()),
        core_coverage=float(f.n_core_inside.sum()/f.n_core.sum()),
        objective=float(f.objective.sum()),max_audit_error=float(f.max_audit_error.max()),
        jensen_blocked=int(f.jensen_blocked.sum()),core_jensen_blocked=int(f.core_jensen_blocked.sum()))


def main():
    out=Path('data/outputs/generalized_grid/transport_chain');out.mkdir(parents=True,exist_ok=True)
    results=[]
    for date in ['2025-01-02','2025-04-01','2025-06-02']:
        start=time.perf_counter()
        try:r=dict(date=date,**fit_chain(load_inputs(date,.1),out/date))
        except Exception as exc:r=dict(date=date,validated=False,error=str(exc))
        r['seconds']=time.perf_counter()-start;results.append(r);print(json.dumps(r),flush=True)
        (out/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf8')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
