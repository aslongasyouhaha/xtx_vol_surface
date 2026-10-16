"""Saved calibration results and shared observation-date validation cases."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sanos_core import build_convex_order_problem
from sanos_incremental import recompute_loss_variables,quote_vectors,validate_frame
from sanos_surface import SanosSurface,audit_surface
from xtx_inputs import load_inputs,core_mask

ROOT=Path(__file__).resolve().parents[1]
BLOCKS={'january':(.25,['2025-01-02','2025-01-03','2025-01-06']),
        'april':(.1,['2025-04-01','2025-04-02','2025-04-03']),
        'june':(.25,['2025-06-02','2025-06-03','2025-06-04'])}


def save_fit(directory,date,p,x,metadata):
    directory.mkdir(parents=True,exist_ok=True);arrays={};frames=[]
    for e,v in p['items'].items():
        prefix=str(e.date());q=x[slice(*p['q_offsets'][e])]
        g=v['quotes'].copy();fit=v['G']@q
        g['fit']=fit;g['outside']=np.maximum(g.pure_bid-fit,0)+np.maximum(fit-g.pure_ask,0)
        g['inside']=g.outside<=1e-9;g['core']=core_mask(g)
        g['outside_spreads']=g.outside/(g.pure_ask-g.pure_bid);frames.append(g)
        arrays[prefix+'__q']=q;arrays[prefix+'__anchors']=v['anchor_k']
        arrays[prefix+'__smooth_variance']=np.asarray(v['smooth_variance'])
        arrays[prefix+'__dte']=np.asarray(v['dte']);arrays[prefix+'__T']=np.asarray(v.get('T',v['dte']/365))
    f=pd.concat(frames,ignore_index=True);core=f[f.core]
    surface=SanosSurface([float(v.get('T',v['dte']/365)) for v in p['items'].values()],
        [v['anchor_k'] for v in p['items'].values()],[x[slice(*p['q_offsets'][e])] for e in p['items']],
        [v['smooth_variance'] for v in p['items'].values()])
    audit=audit_surface(surface)
    row=dict(date=date,success=audit['passed'],quotes=len(f),expiries=len(p['items']),
        coverage=float(f.inside.mean()),core_coverage=float(core.inside.mean()),
        core_misses=int((~core.inside).sum()),outside_loss=float(f.outside_spreads.sum()),surface_audit=audit,**metadata)
    f.to_csv(directory/'quotes.csv',index=False);np.savez_compressed(directory/'model.npz',**arrays)
    (directory/'summary.json').write_text(json.dumps(row,indent=2),encoding='utf8')
    if not audit['passed']:raise ValueError('Saved fit failed independent surface audit')
    return row


def fit_stats(p,x):
    total=inside=nc=ic=0;loss=0.
    for e,v in p['items'].items():
        fit=v['G']@x[slice(*p['q_offsets'][e])]
        outside=np.maximum(v['pure_bid']-fit,0)+np.maximum(fit-v['pure_ask'],0)
        c=core_mask(v['quotes']);total+=len(fit);inside+=int(np.sum(outside<=1e-9))
        nc+=int(c.sum());ic+=int(np.sum(outside[c]<=1e-9));loss+=float(np.sum(outside/(v['pure_ask']-v['pure_bid'])))
    surface=SanosSurface([v.get('T',v['dte']/365) for v in p['items'].values()],
        [v['anchor_k'] for v in p['items'].values()],[x[slice(*p['q_offsets'][e])] for e in p['items']],
        [v['smooth_variance'] for v in p['items'].values()])
    return dict(coverage=inside/total,core_coverage=ic/nc,outside_loss=loss,surface_audit=audit_surface(surface))


def load_saved_lp(date,block,eta):
    p=build_convex_order_problem(load_inputs(date,eta));x=np.zeros(p['cursor'])
    with np.load(ROOT/f'data/outputs/daily_validation/{block}/{date}/lp/model.npz') as model:
        for e,v in p['items'].items():
            key=str(e.date());np.testing.assert_array_equal(v['anchor_k'],model[key+'__anchors'])
            x[slice(*p['q_offsets'][e])]=model[key+'__q']
    quotes={e:(v['pure_bid'],v['pure_ask']) for e,v in p['items'].items()}
    recompute_loss_variables(p,x,quotes)
    upper,cost=quote_vectors(p,quotes)
    if not validate_frame(p,upper,cost,x,quotes)['validated']:
        raise ValueError('Stored LP audit failed')
    return p,x,quotes
