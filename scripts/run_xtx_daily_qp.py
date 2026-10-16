"""Replay saved daily LPs with a sequential, fit-protected sparse-support QP."""
import _bootstrap  # project paths for direct script execution
import argparse
import json
import pandas as pd
from threadpoolctl import threadpool_limits
from xtx_results import BLOCKS,ROOT,save_fit,load_saved_lp
from sanos_daily import snapshot,daily_references,active_support_qp


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
            p,x,quotes=load_saved_lp(date,block,eta)
            if previous is None:
                previous=snapshot(date,p,x);continue
            refs=daily_references(previous,p,date)
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
