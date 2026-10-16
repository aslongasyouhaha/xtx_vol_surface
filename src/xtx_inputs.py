"""Shared saved-input loading and diagnostic quote-region definitions."""
from pathlib import Path
import json, pickle
import numpy as np
import pandas as pd
from scipy.special import ndtr
from xtx_data import build_baseline_inputs
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/outputs/input_cache'

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
