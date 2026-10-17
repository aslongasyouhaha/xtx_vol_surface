"""Shared calibration input/output, public fixtures and diagnostic quote regions."""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import ndtr
from xtx_data import build_baseline_inputs
from sanos_core import build_convex_order_problem
from sanos_incremental import recompute_loss_variables, quote_vectors, validate_frame
from sanos_surface import SanosSurface, audit_surface

ROOT=Path(__file__).resolve().parents[1]


OUT=ROOT/'data/outputs/input_cache'


TESTCASES=('clean','wide_wings','conflicting_quotes')


def load_testcase(case,data_dir=None):
    """Load public synthetic quotes without opening any licensed market files."""
    directory=Path(data_dir) if data_dir is not None else ROOT/'data/testcases'
    manifest=json.loads((directory/'cases.json').read_text(encoding='utf8'))
    if manifest.get('data_kind')!='synthetic':raise ValueError('Expected synthetic testcase metadata')
    config=next((c for c in manifest['cases'] if c['id']==case),None)
    if config is None:raise ValueError(f'Unknown testcase: {case}')
    quotes=pd.read_csv(directory/config['file'])
    if not quotes.data_kind.eq('synthetic').all():raise ValueError('Testcase rows must identify synthetic data')
    from xtx_data import implied_vol_from_pure_option
    asof=pd.Timestamp(manifest['observation_date']);items={}
    for specification in manifest['slices']:
        dte=specification['dte'];g=quotes.loc[quotes.dte.eq(dte)].copy();T=dte/365.
        if g.empty:raise ValueError(f'Testcase has no {dte}D quotes')
        expiry=asof+pd.Timedelta(days=dte)
        g['date']=str(asof.date());g['exdate']=str(expiry.date());g['T']=T
        g['source_cp_flag']=np.where(g.k<1,'P','C')
        g['source_otm_iv']=[implied_vol_from_pure_option(k,c-max(1-k,0),T,cp)
            for k,c,cp in zip(g.k,g.pure_mid,g.source_cp_flag)]
        items[expiry]=dict(dte=dte,T=T,anchor_k=np.asarray(manifest['anchors'],float),
            market_k=g.k.to_numpy(),smooth_variance=specification['variance'],
            pure_bid=g.pure_bid.to_numpy(),pure_mid=g.pure_mid.to_numpy(),pure_ask=g.pure_ask.to_numpy(),quotes=g)
    return items,config


def load_inputs(date, eta, *, baseline_variant='baseline_paper_gap001'):
    base = ROOT / 'data/outputs' / baseline_variant / date
    cache = OUT / f'{date}_{baseline_variant}_bid_backbone_gap001_inputs.pkl'
    cache.parent.mkdir(parents=True, exist_ok=True)
    summary_path = base / 'summary.json'
    metadata = json.loads(summary_path.read_text(encoding='utf8')) if summary_path.exists() else {}
    if (base / 'model.npz').exists() and metadata.get('success', False):
        model = np.load(base / 'model.npz')
        quotes = pd.read_csv(base / 'quotes.csv')
        items = {}
        for (ex, g) in quotes.groupby('exdate', sort=True):
            items[pd.Timestamp(ex)] = {'dte': int(g.dte.iloc[0]), 'anchor_k': model[ex + '__anchors'], 'market_k': g.k.to_numpy(), 'smooth_variance': eta * float(g.atm_total_variance.iloc[0]), 'pure_bid': g.pure_bid.to_numpy(), 'pure_ask': g.pure_ask.to_numpy(), 'pure_mid': g.pure_mid.to_numpy(), 'quotes': g}
    else:
        if cache.exists():
            with cache.open('rb') as f:
                items = pickle.load(f)
        else:
            items = build_baseline_inputs(date)['joint_inputs']
            with cache.open('wb') as f:
                pickle.dump(items, f)
        for v in items.values():
            v['smooth_variance'] = eta * float(v['quotes'].atm_total_variance.iloc[0])
    return items


def core_mask(g):
    sigma = g.source_otm_iv.to_numpy() * np.sqrt(g['T'].to_numpy())
    d1 = -np.log(g.k.to_numpy()) / sigma + 0.5 * sigma
    return np.where(g.source_cp_flag.eq('P'), ndtr(-d1), ndtr(d1)) >= 0.1


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


def load_saved_lp(date,block,eta,*,baseline_dir=None):
    p=build_convex_order_problem(load_inputs(date,eta));x=np.zeros(p['cursor'])
    root=Path(baseline_dir) if baseline_dir is not None else ROOT/'data/outputs/daily_validation'
    with np.load(root/block/date/'lp/model.npz') as model:
        for e,v in p['items'].items():
            key=str(e.date());np.testing.assert_array_equal(v['anchor_k'],model[key+'__anchors'])
            x[slice(*p['q_offsets'][e])]=model[key+'__q']
    quotes={e:(v['pure_bid'],v['pure_ask']) for e,v in p['items'].items()}
    recompute_loss_variables(p,x,quotes)
    upper,cost=quote_vectors(p,quotes)
    if not validate_frame(p,upper,cost,x,quotes)['validated']:
        raise ValueError('Stored LP audit failed')
    return p,x,quotes
