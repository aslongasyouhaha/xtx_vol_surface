"""Fixed-quote/eta joint LP controls for left-wing boundary reversals."""
import _bootstrap  # project paths for direct script execution
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_xtx_daily_validation import save_fit
from xtx_inputs import load_inputs,core_mask
from sanos_core import build_convex_order_problem
from sanos_incremental import PersistentSanosLP
from diagnose_xtx_wing_turns import otm,iv
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/outputs/wing_boundary'

def shape(p,x,e):
    v=p['items'][e];k=np.linspace(v['market_k'].min(),v['market_k'].max(),1501)
    q=x[slice(*p['q_offsets'][e])];T=v.get('T',v['dte']/365)
    sigma=iv(k,otm(k,v['anchor_k'],v['smooth_variance'],q),T)
    idx=np.flatnonzero(k<min(1,k[300]));left=sigma[idx]
    return k,sigma,dict(left_edge_iv=float(left[0]),left_peak_iv=float(np.nanmax(left)),
        left_turn_amplitude=float(np.nanmax(left)-left[0]),
        left_peak_k=float(k[idx[np.nanargmax(left)]]),anchor_min=float(v['anchor_k'].min()))

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for date,block,eta,dtes in [('2025-01-02','january',.25,[88]),('2025-04-03','april',.1,[43,88])]:
        inputs=load_inputs(date,eta);p=build_convex_order_problem(inputs);x=np.zeros(p['cursor'])
        with np.load(ROOT/f'data/outputs/daily_validation/{block}/{date}/lp/model.npz') as m:
            for e,v in p['items'].items():
                prefix=str(e.date());np.testing.assert_array_equal(v['anchor_k'],m[prefix+'__anchors'])
                x[slice(*p['q_offsets'][e])]=m[prefix+'__q']
        variants=[('saved_baseline',p,x)]
        for name,left in [('add_state_020',.2)]:
            changed={e:dict(v,anchor_k=np.unique(np.r_[left,v['anchor_k']])) for e,v in p['items'].items()}
            session=PersistentSanosLP(changed,time_limit=45);r=session.solve()
            if not r['success']:
                rows.append(dict(date=date,variant=name,success=False,message=r['message']))
                print(json.dumps(rows[-1]),flush=True);continue
            report=save_fit(OUT/date/name,date,session.problem,r['x'],dict(eta=eta,variant=name))
            variants.append((name,session.problem,r['x']))
            print(json.dumps(dict(date=date,variant=name,coverage=report['coverage'],core_coverage=report['core_coverage'])),flush=True)
        for dte in dtes:
            e=next(e for e,v in p['items'].items() if v['dte']==dte);g=p['items'][e]['quotes']
            fig,axs=plt.subplots(1,2,figsize=(12,4))
            axs[0].scatter(g.k,g.source_otm_iv,s=8,c='black',label='eligible market mid')
            T=p['items'][e].get('T',dte/365);kq=g.k.to_numpy();intrinsic=np.maximum(1-kq,0)
            bidiv=iv(kq,g.pure_bid.to_numpy()-intrinsic,T);askiv=iv(kq,g.pure_ask.to_numpy()-intrinsic,T)
            for ax in axs:ax.fill_between(kq,bidiv,askiv,color='gray',alpha=.25,label='eligible bid/ask')
            curves={}
            for name,pp,xx in variants:
                k,s,stats=shape(pp,xx,e);curves[name]=(k,s)
                fit=pp['items'][e]['G']@xx[slice(*pp['q_offsets'][e])]
                outside=np.maximum(g.pure_bid.to_numpy()-fit,0)+np.maximum(fit-g.pure_ask.to_numpy(),0)
                row=dict(date=date,dte=dte,variant=name,success=True,coverage=float(np.mean(outside<=1e-9)),
                    core_coverage=float(np.mean(outside[core_mask(g)]<=1e-9)),**stats)
                rows.append(row)
                for ax in axs:ax.plot(k,s,label=name)
            lo=g.k.min();hi=min(1,lo+.2*(g.k.max()-lo));axs[1].set_xlim(lo,hi)
            for ax in axs:ax.set_xlabel('k');ax.set_ylabel('OTM IV');ax.grid(alpha=.2);ax.legend(fontsize=7)
            fig.suptitle(f'{date} {dte}D: identical quotes, eta={eta}; only add common left state 0.2')
            fig.tight_layout();fig.savefig(OUT/f'{date}_{dte}D.png',dpi=150);plt.close(fig)
        (OUT/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
        pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
