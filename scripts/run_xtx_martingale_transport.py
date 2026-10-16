"""April fixed-prefix state-dependent transition feasibility experiment."""
import _bootstrap  # project paths for direct script execution
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from xtx_inputs import load_inputs,core_mask
from sanos_generalized_grid import GridProduct
from sanos_martingale_transport import fit_state_dependent_transition


def main():
    root=Path('data/outputs/generalized_grid')
    out=root/'state_dependent_transport';out.mkdir(parents=True,exist_ok=True)
    items=list(load_inputs('2025-04-01',.1).values())
    saved=np.load(root/'2025-04-01_16385.npz');rows=[]
    for j,it in enumerate(items):
        if it['dte'] not in [3,6,7,10]:continue
        start=time.perf_counter()
        try:
            result=fit_state_dependent_transition(saved['states'],saved[f'density_{j-1}'],it,
                preserve_strikes=items[j-1]['market_k'],time_limit=40)
            core=core_mask(it['quotes']);fit=result['fit'];inside=(fit>=it['pure_bid']-1e-9)&(fit<=it['pure_ask']+1e-9)
            row=dict(dte=int(it['dte']),validated=result['validated'],
                coverage=result['coverage'],core_coverage=float(inside[core].mean()),
                objective=result['objective'],outside_objective=result['outside_objective'],
                errors=result['errors'],grid_size=len(result['grid']))
            idx=np.searchsorted(np.cumsum(result['source']),[.1,.5,.9])
            row['conditional_samples']=[dict(state=float(result['grid'][i]),
                relative_variance=float(result['conditional_relative_variance'][i])) for i in idx]
            prefix=out/f'dte_{it["dte"]}'
            np.savez_compressed(str(prefix)+'.npz',grid=result['grid'],source=result['source'],
                target=result['target'],Q=result['Q'],original_states=saved['states'],
                original_weights=saved[f'density_{j-1}'],market_k=it['market_k'],
                bid=it['pure_bid'],ask=it['pure_ask'],mid=it['pure_mid'],fit=fit)
            from scipy import sparse
            sparse.save_npz(str(prefix)+'_projection.npz',result['projection'])
        except Exception as exc:row=dict(dte=int(it['dte']),validated=False,error=str(exc))
        row['seconds']=time.perf_counter()-start;rows.append(row);print(json.dumps(row),flush=True)
        (out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf8')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
