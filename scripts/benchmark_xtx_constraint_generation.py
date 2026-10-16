"""Full LP versus calendar-row generation on identical synthetic updates."""
import _bootstrap  # project paths for direct script execution
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEPS=ROOT/'data/outputs'/'native_deps'
if (DEPS/'highspy').exists():sys.path.insert(0,str(DEPS))
from xtx_inputs import load_inputs
from diagnose_xtx_active_update import active_quotes
from sanos_incremental import PersistentSanosLP
from sanos_constraint_generation import solve_generated


def main():
    output=ROOT/'data/outputs'/'constraint_generation';output.mkdir(parents=True,exist_ok=True)
    records=[]
    for date in ['2025-01-02','2025-04-01','2025-06-02']:
        s=PersistentSanosLP(load_inputs(date,.1 if date=='2025-04-01' else .25,baseline_variant='baseline'),working_set=False,time_limit=30.,objective_mode='legacy')
        initial=s.solve();assert initial['success']
        dual=np.asarray(s.h.getSolution().row_dual).copy()
        quotes,_,_=active_quotes(s.problem);s.update_quotes(quotes)
        generated=solve_generated(s,previous_dual=dual,time_limit=12.)
        full=s.solve();assert full['success']
        row=dict(date=date,generated_success=generated['success'],generated_seconds=generated['request_seconds'],
                 attempts=generated['attempts'],full_seconds=full['request_seconds'],full_objective=full['actual_outside_objective'],
                 original_calendar_rows=s.problem['calendar_constraint_count'])
        if generated['success']:
            delta=abs(generated['audit']['actual_outside_objective']-full['actual_outside_objective'])
            row.update(audit=generated['audit'],gap=generated['objective_gap'],objective_difference_to_full=delta)
            assert delta<=1e-6
        records.append(row);(output/'results.json').write_text(json.dumps(records,indent=2),encoding='utf8')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
