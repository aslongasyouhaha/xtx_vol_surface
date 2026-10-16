"""Replay saved daily LPs with a sequential, fit-protected sparse-support QP."""
import _bootstrap  # project paths for direct script execution
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_xtx_daily_validation import BLOCKS,ROOT,save_fit
from xtx_inputs import load_inputs
from sanos_core import build_convex_order_problem
from sanos_daily import snapshot,daily_references,active_support_qp
from sanos_incremental import recompute_loss_variables,quote_vectors,validate_frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks',nargs='+',choices=list(BLOCKS),default=list(BLOCKS))
    parser.add_argument('--loss-allowance',type=float,default=.01,
        help='Total additional spread-normalized outside loss; core LP fits stay hard constraints.')
    args=parser.parse_args();root=ROOT/'data/outputs/daily_validation';out=root/'active_support_qp';out.mkdir(parents=True,exist_ok=True);rows=[]
    for block in args.blocks:
        eta,dates=BLOCKS[block];previous=None
        for date in dates:
            path=root/block/date/'lp/model.npz'
            if not path.exists():raise ValueError(f'Run daily LP validation first: {date}')
            p=build_convex_order_problem(load_inputs(date,eta));x=np.zeros(p['cursor'])
            with np.load(path) as model:
                for e,v in p['items'].items():
                    prefix=str(e.date());np.testing.assert_array_equal(v['anchor_k'],model[prefix+'__anchors'])
                    x[slice(*p['q_offsets'][e])]=model[prefix+'__q']
            quotes={e:(v['pure_bid'],v['pure_ask']) for e,v in p['items'].items()}
            recompute_loss_variables(p,x,quotes);upper,cost=quote_vectors(p,quotes)
            if not validate_frame(p,upper,cost,x,quotes)['validated']:raise ValueError('Stored LP audit failed')
            if previous is None:
                previous=snapshot(date,p,x);continue
            refs=daily_references(previous,p,date);start=time.perf_counter()
            result=active_support_qp(p,x,quotes,refs,time_limit=20.,loss_allowance=args.loss_allowance)
            metadata=dict(block=block,eta=eta,method='cross_day_active_support_QP',source_date=previous['date'],
                support_mode=result['support_mode'],full_state_count=result['full_state_count'],
                allowed_state_count=result['allowed_state_count'],attempts=result['attempts'],seconds=result['seconds'])
            if result['success']:
                metadata.update(reference_quotes=result['reference_quotes'],lp_reference_rms_spreads=result['lp_movement_rms'],
                    qp_reference_rms_spreads=result['qp_movement_rms'],loss_allowance=result['loss_allowance'],
                    lp_outside_loss=result['lp_outside_loss'],feasible_step_alpha=result['feasible_step_alpha'],
                    selected_point_is_optimizer_result=result['selected_point_is_optimizer_result'])
                row=save_fit(out/block/date,date,p,result['x'],metadata)
                previous=snapshot(date,p,result['x'])
            else:
                row=dict(date=date,success=False,reason=result['reason'],deployed_fallback='current_validated_LP',**metadata)
                previous=snapshot(date,p,x)
            rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
            pd.DataFrame([{k:v for k,v in r.items() if k not in {'surface_audit','attempts'}} for r in rows]).to_csv(out/'summary.csv',index=False)
            print(json.dumps({k:row.get(k) for k in ['date','success','core_coverage','lp_reference_rms_spreads','qp_reference_rms_spreads','reason','allowed_state_count']}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
