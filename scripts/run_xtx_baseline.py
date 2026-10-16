"""Reproducible 1-120D SANOS convex-order baseline and date validation."""
from __future__ import annotations
import _bootstrap  # project paths for direct script execution

import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from sanos_core import solve_convex_order_lp,call_kernel
from sanos_surface import SanosSurface,audit_surface
from vol_surface_utils import DEFAULT_MAX_DTE, DEFAULT_MIN_DAILY_VOLUME,DEFAULT_MODEL_MAX_GAP

ROOT = Path(__file__).resolve().parents[1]


from xtx_data import build_baseline_inputs


def run_date(trade_date, output_root, *, data_path=None, yield_dir=None,
             min_daily_volume=DEFAULT_MIN_DAILY_VOLUME, eta=None):
    if eta is not None and (not np.isfinite(eta) or not 0<=eta<1):
        raise ValueError('eta must be finite and in [0, 1).')
    started = time.perf_counter()
    out = Path(output_root) / str(pd.Timestamp(trade_date).date())
    out.mkdir(parents=True, exist_ok=True)
    env = build_baseline_inputs(trade_date, data_path=data_path, yield_dir=yield_dir,
                                min_daily_volume=min_daily_volume)
    if eta is not None:
        env['ETA']=float(eta)
        for item in env['joint_inputs'].values():
            item['smooth_variance']=float(eta)*float(item['quotes'].atm_total_variance.iloc[0])
            item['G']=call_kernel(item['anchor_k'],item['market_k'],item['smooth_variance'])
    env['atm_rows'].to_csv(out/'variance_backbone.csv',index=False)
    print(f"{trade_date}: {len(env['joint_inputs'])} expiries; solving convex-order LP", flush=True)
    sol = solve_convex_order_lp(env['joint_inputs'])
    summary = {'trade_date': str(pd.Timestamp(trade_date).date()), 'max_dte': DEFAULT_MAX_DTE,
               'volume_filter_applied': min_daily_volume is not None,
               'min_daily_volume': min_daily_volume, 'data_timing': 'end_of_day', 'eta': env['ETA'],
               'anchor_policy': 'eligible_pre_volume_gap_fill','model_max_gap':DEFAULT_MODEL_MAX_GAP,
               'formulation': sol['formulation'], 'success': sol['success'],
               'message': sol['message'], 'n_expiries': len(sol['exdates']),
               'solve_seconds': sol['solve_seconds'], 'mid_tiebreak': sol['mid_tiebreak'],
               'mid_penalty':sol['mid_penalty'],'weight_mode':sol['weight_mode'],
               'variance_source':'atm_call_bid_iv',
               'variance_monotone_adjustment':'isotonic_if_needed',
               'variance_adjusted_expiries':int(env['atm_rows']['atm_total_variance_adjustment'].abs().gt(1e-14).sum()),
               'n_vars': sol['n_vars'], 'n_eq': sol['n_eq'], 'n_ub': sol['n_ub'],
               'matrix_nnz': sol['A_eq_nnz']+sol['A_ub_nnz'],
               'solver_attempts': sol['solver_attempts'],
               'validation_tolerance': sol['validation_tolerance'],
               'quote_coverage_tolerance': 1e-9,
               'data_path': str(env['DATA_PATH'])}
    if not sol['success']:
        summary['residuals'] = sol.get('residuals')
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
        return summary
    try:
        surface=SanosSurface.from_solution(sol)
        summary['surface_audit']=audit_surface(surface)
        if not summary['surface_audit']['passed']:raise ValueError('Intermediate-time surface audit failed')
    except ValueError as exc:
        summary.update(success=False,message=str(exc))
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
        return summary
    frames = []
    for ex, item in sol['inputs'].items():
        q = item['quotes'].copy()
        fit = sol['fit_by_expiry'][ex]
        q['fit'] = fit
        q['outside'] = np.maximum(q.pure_bid - fit, 0) + np.maximum(fit - q.pure_ask, 0)
        q['inside'] = q.outside <= 1e-9
        q['outside_spreads'] = q.outside / (q.source_pure_ask - q.source_pure_bid)
        iv = q.source_otm_iv.to_numpy()
        sqrtv = iv * np.sqrt(q['T'].to_numpy())
        d1 = (-np.log(q.k.to_numpy()) + 0.5 * sqrtv**2) / sqrtv
        q['abs_forward_delta'] = np.where(q.source_cp_flag.eq('P'), norm.cdf(-d1), norm.cdf(d1))
        q['core'] = q.abs_forward_delta >= 0.10
        frames.append(q)
    quotes = pd.concat(frames, ignore_index=True)
    core = quotes[quotes.core]
    per = sol['per_expiry'].copy()
    core_stats = core.groupby('exdate').agg(core_quotes=('inside', 'size'), core_coverage=('inside', 'mean')).reset_index()
    per = per.merge(core_stats, on='exdate', how='left')
    per.to_csv(out / 'per_expiry.csv', index=False)
    quotes.to_csv(out / 'quotes.csv', index=False)
    sol['calendar_checks'].to_csv(out / 'calendar_checks.csv', index=False)
    sol['convex_order_checks'].to_csv(out / 'convex_order_checks.csv', index=False)
    np.savez(out / 'weights.npz', **{str(ex.date()): sol['q_by_expiry'][ex] for ex in sol['exdates']})
    model_arrays = {}
    for ex, item in sol['inputs'].items():
        prefix = str(ex.date())
        model_arrays[prefix + '__q'] = sol['q_by_expiry'][ex]
        model_arrays[prefix + '__anchors'] = item['anchor_k']
        for key in ['smooth_variance', 'dte']:
            model_arrays[prefix + '__' + key] = np.asarray(item[key])
        model_arrays[prefix+'__T']=np.asarray(item.get('T',item['dte']/365.))
        for key in ['forward', 'df']:
            model_arrays[prefix + '__' + key] = np.asarray(item['quotes'][key].iloc[0])
    np.savez(out / 'model.npz', **model_arrays)
    summary.update(n_quotes=len(quotes), coverage=float(quotes.inside.mean()),
                   core_quotes=len(core), core_coverage=float(core.inside.mean()) if len(core) else None,
                   core_misses=int((~core.inside).sum()),
                   core_outside_gt_01_spread=int((~core.inside & core.outside_spreads.gt(.1)).sum()),
                   calendar_min=float(sol['calendar_checks'].min_calendar_diff.min()) if len(sol['calendar_checks']) else 0.0,
                   convex_order_min=float(sol['convex_order_checks'].min_convex_order_diff.min()) if len(sol['convex_order_checks']) else 0.0,
                   residuals=sol['residuals'], objective=sol['objective'],
                   actual_objective=sol['actual_objective'],
                   objective_reconstruction_error=sol['objective_reconstruction_error'],
                   weighted_outside_error=float(sol['per_expiry'].weighted_outside_error.sum()),
                   weighted_mid_penalty=float(sol['per_expiry'].weighted_mid_tiebreak_error.sum()),
                   elapsed_seconds=time.perf_counter()-started)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    ax.plot(per.dte, per.bid_ask_coverage, 'o-', label='All quotes')
    ax.plot(per.dte, per.core_coverage, 'o-', label='OTM absolute forward delta >= 10%')
    ax.set(xlabel='Days to expiry', ylabel='Bid/ask coverage', ylim=(0, 1.02), title=f'SANOS convex-order baseline: {trade_date}')
    ax.grid(alpha=.2); ax.legend()
    fig.savefig(out / 'fit_diagnostics.png', dpi=160); plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trade-date', nargs='+', default=['2025-06-02'])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-volume-filter', action='store_true', help='Explicit historical no-volume control; use a separate output directory.')
    parser.add_argument('--data-path', type=Path)
    parser.add_argument('--yield-dir', type=Path)
    parser.add_argument('--eta',type=float,help='Explicit smoothing scale in [0,1); otherwise use the data recipe default. This is not automatic eta selection.')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir=ROOT/'data/outputs'/('baseline_paper_gap001_no_volume_control' if args.no_volume_filter else 'baseline_paper_gap001')
    rows = []
    for day in args.trade_date:
        print(f'Baseline {day}: build data and solve', flush=True)
        try:
            row = run_date(day, args.output_dir, data_path=args.data_path, yield_dir=args.yield_dir,
                           min_daily_volume=None if args.no_volume_filter else DEFAULT_MIN_DAILY_VOLUME,eta=args.eta)
        except Exception as exc:
            row = {'trade_date': day, 'success': False, 'message': f'{type(exc).__name__}: {exc}'}
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        saved = {}
        for path in args.output_dir.glob('*/summary.json'):
            previous = json.loads(path.read_text(encoding='utf8'))
            saved[previous['trade_date']] = previous
        saved.update({r['trade_date']: r for r in rows})
        pd.DataFrame([saved[k] for k in sorted(saved)]).drop(columns=['residuals', 'solver_attempts'], errors='ignore').to_csv(args.output_dir / 'multi_day_validation.csv', index=False)
    if not all(r['success'] for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
