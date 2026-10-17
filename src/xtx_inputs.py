"""Shared saved-input loading and diagnostic quote-region definitions."""
from pathlib import Path
import json, pickle
import numpy as np
import pandas as pd
from scipy.special import ndtr
from xtx_data import build_baseline_inputs
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
