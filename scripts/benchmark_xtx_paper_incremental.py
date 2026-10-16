"""Same paper LP, frozen geometry: full, no-op, binding update and cold control."""
import _bootstrap  # project paths for direct script execution
import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEPS=ROOT/'data/outputs'/'native_deps'
if (DEPS/'highspy').exists():sys.path.insert(0,str(DEPS))
from xtx_inputs import load_inputs,core_mask
from sanos_core import solve_convex_order_lp
from sanos_incremental import PersistentSanosLP,quote_vectors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dates',nargs='+',default=['2025-01-02','2025-04-01','2025-06-02'])
    parser.add_argument('--time-limit',type=float,default=60.)
    parser.add_argument('--eta',type=float,default=.25)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();out=args.output_dir or ROOT/'data/outputs'/'paper_incremental';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for date in args.dates:
        inputs=load_inputs(date,args.eta)
        s=PersistentSanosLP(inputs,time_limit=args.time_limit)
        initial=s.solve()
        row=dict(date=date,eta=args.eta,objective_mode=s.objective_mode,variables=s.problem['cursor'],
                 initial_success=initial['success'],initial_seconds=initial['request_seconds'],
                 initial_message=initial['message'],initial_numerics=initial.get('numerics_attempts'))
        if initial['success']:
            frames=[]
            for key,item in s.problem['items'].items():
                g=item['quotes'].copy()
                g['fit']=item['G']@initial['x'][slice(*s.problem['q_offsets'][key])]
                g['outside']=np.maximum(g.pure_bid-g['fit'],0)+np.maximum(g['fit']-g.pure_ask,0)
                g['outside_spreads']=g.outside/(g.pure_ask-g.pure_bid)
                g['inside']=g.outside<=1e-9;g['core']=core_mask(g)
                frames.append(g)
            fitted=pd.concat(frames,ignore_index=True)
            fitted.to_csv(out/f'{date}_initial_quotes.csv',index=False)
            row.update(initial_coverage=float(fitted.inside.mean()),
                initial_core_coverage=float(fitted.loc[fitted.core,'inside'].mean()),
                initial_max_constraint_violation=initial['max_constraint_violation'])
            repeated=s.solve();row.update(noop_success=repeated['success'],noop_seconds=repeated['request_seconds'],noop_iterations=repeated['iterations'])
            ex=min(s.problem['items'],key=lambda key:abs(s.problem['items'][key]['dte']-30))
            v=s.problem['items'][ex];indices=np.flatnonzero(core_mask(v['quotes']));index=int(indices[len(indices)//2])
            quotes={key:(b.copy(),a.copy()) for key,(b,a) in s.quotes.items()}
            bid,ask=quotes[ex];width=ask[index]-bid[index]
            oldfit=float(v['G'][index]@initial['x'][slice(*s.problem['q_offsets'][ex])])
            bid[index]=max(ask[index],oldfit)+.05*width;ask[index]=bid[index]+width
            row['old_fit_violation']=float(bid[index]-oldfit)
            s.update_quotes(quotes);updated=s.solve()
            row.update(update_success=updated['success'],update_seconds=updated['request_seconds'],
                update_message=updated['message'],update_path=updated['optimization_path'],
                fallback_reason=updated.get('working_set_fallback_reason'),local_attempts=updated.get('working_set_attempts'))
            control=copy.deepcopy(inputs)
            for key,item in control.items():
                item['pure_bid'],item['pure_ask']=quotes[key]
                item['pure_mid']=.5*(quotes[key][0]+quotes[key][1]) if key==ex else item['pure_mid']
            cold=solve_convex_order_lp(control,time_limit=args.time_limit)
            row.update(cold_success=cold['success'],cold_seconds=cold['solve_seconds'])
            if cold['success'] and updated['success']:
                delta=abs(updated['actual_objective']-cold['actual_objective'])
                row.update(actual_objective=updated['actual_objective'],cold_actual_objective=cold['actual_objective'],
                    objective_difference=delta,objective_agrees_1e6=delta<=1e-6,
                    max_constraint_violation=updated['max_constraint_violation'],
                    global_gap=updated.get('global_objective_gap'))
        rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
