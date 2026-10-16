"""SPXW EOD input preparation, independent of notebook execution.

Normalization, bid-IV backbone, volume-selected observations and pre-volume states.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.isotonic import IsotonicRegression
from vol_surface_utils import DEFAULT_MAX_DTE, DEFAULT_MIN_DAILY_VOLUME, DEFAULT_MODEL_MAX_GAP
from sanos_core import build_model_anchor_grid, add_common_boundary_anchors

def bs_call_forward_unit(k, total_variance):
    total_variance = max(float(total_variance), 0.0)
    if total_variance < 1e-14:
        return max(1.0 - float(k), 0.0)
    vol_sqrt_t = np.sqrt(total_variance)
    d1 = (-np.log(float(k)) + 0.5 * total_variance) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return norm.cdf(d1) - float(k) * norm.cdf(d2)

def bs_put_forward_unit(k, total_variance):
    total_variance = max(float(total_variance), 0.0)
    if total_variance < 1e-14:
        return max(float(k) - 1.0, 0.0)
    vol_sqrt_t = np.sqrt(total_variance)
    d1 = (-np.log(float(k)) + 0.5 * total_variance) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(k) * norm.cdf(-d2) - norm.cdf(-d1)

def implied_vol_from_pure_option(k: float, pure_price: float, T: float, cp_flag: str) -> float:
    """Invert normalized forward-unit option price into Black-Scholes IV.

    k is K/F, and the price is divided by DF*F. For IV diagnostics we use
    OTM options: puts when k < 1 and calls when k >= 1.
    """
    k = float(k)
    pure_price = float(pure_price)
    T = float(T)
    cp_flag = str(cp_flag).upper()[0]
    if not np.isfinite(k) or not np.isfinite(pure_price) or (not np.isfinite(T)) or (k <= 0) or (T <= 0):
        return np.nan
    if cp_flag == 'P':
        lower = max(k - 1.0, 0.0)
        upper = k
        pricing_fn = bs_put_forward_unit
    else:
        lower = max(1.0 - k, 0.0)
        upper = 1.0
        pricing_fn = bs_call_forward_unit
    tol = 1e-10
    pure_price = min(max(pure_price, lower + tol), upper - tol)

    def objective(sigma):
        return pricing_fn(k, sigma * sigma * T) - pure_price
    try:
        return brentq(objective, 1e-06, 6.0, maxiter=100)
    except ValueError:
        return np.nan

def implied_vol_from_pure_call(k: float, pure_price: float, T: float) -> float:
    """Backward-compatible helper for ATM-call total-variance construction."""
    return implied_vol_from_pure_option(k, pure_price, T, 'C')

def otm_cp_flag_from_k(k: float) -> str:
    return 'C' if float(k) >= 1.0 else 'P'

def otm_price_from_pure_call(k: float, pure_call_price: float) -> float:
    """Convert fitted pure call into the corresponding OTM option price.

    In normalized forward units, put-call parity is C - P = 1 - k,
    so the put price is P = C + k - 1.
    """
    k = float(k)
    pure_call_price = float(pure_call_price)
    if k >= 1.0:
        return pure_call_price
    return pure_call_price + k - 1.0

def bs_vega_forward_unit(k: float, sigma: float, T: float) -> float:
    k = float(k)
    sigma = float(sigma)
    T = float(T)
    if not np.isfinite(k) or not np.isfinite(sigma) or (not np.isfinite(T)) or (k <= 0) or (sigma <= 0) or (T <= 0):
        return np.nan
    total_variance = sigma * sigma * T
    if total_variance < 1e-14:
        return np.nan
    vol_sqrt_t = np.sqrt(total_variance)
    d1 = (-np.log(k) + 0.5 * total_variance) / vol_sqrt_t
    return norm.pdf(d1) * np.sqrt(T)

def bs_call_kernel(spot: np.ndarray, strike: np.ndarray, total_variance: float) -> np.ndarray:
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    if total_variance <= 0:
        return np.maximum(spot - strike, 0.0)
    sqrt_v = np.sqrt(total_variance)
    d1 = (np.log(spot / strike) + 0.5 * total_variance) / sqrt_v
    d2 = d1 - sqrt_v
    return spot * norm.cdf(d1) - strike * norm.cdf(d2)

def build_kernel_matrix(anchor_k: np.ndarray, market_k: np.ndarray, total_variance: float) -> np.ndarray:
    return bs_call_kernel(spot=anchor_k[None, :], strike=market_k[:, None], total_variance=total_variance)

def build_variance_backbone(call_surface):
    call_surface['bs_iv'] = [implied_vol_from_pure_call(k, price, T) for (k, price, T) in zip(call_surface['k'], call_surface['pure_mid'], call_surface['T'])]
    call_surface['bs_total_variance'] = call_surface['bs_iv'] ** 2 * call_surface['T']
    call_surface['bs_bid_iv'] = [implied_vol_from_pure_call(k, price, T) if price > max(1.0 - k, 0.0) else np.nan for (k, price, T) in zip(call_surface['k'], call_surface['pure_bid'], call_surface['T'])]
    call_surface['bs_bid_total_variance'] = call_surface['bs_bid_iv'] ** 2 * call_surface['T']
    atm_candidates = call_surface[np.isfinite(call_surface['bs_bid_iv']) & call_surface['bs_bid_iv'].gt(0) & np.isfinite(call_surface['bs_bid_total_variance']) & call_surface['bs_bid_total_variance'].gt(0)].copy()
    atm_candidates['abs_moneyness'] = (atm_candidates['k'] - 1.0).abs()
    atm_idx = atm_candidates.groupby('exdate')['abs_moneyness'].idxmin()
    atm_rows = atm_candidates.loc[atm_idx, ['exdate', 'dte', 'T', 'k', 'bs_bid_iv', 'bs_bid_total_variance']].rename(columns={'k': 'atm_k', 'bs_bid_iv': 'atm_iv', 'bs_bid_total_variance': 'atm_total_variance_raw'}).sort_values('T', kind='mergesort').reset_index(drop=True)
    atm_rows['variance_diff_raw'] = atm_rows['atm_total_variance_raw'].diff()
    atm_rows['variance_monotone_violation'] = atm_rows['variance_diff_raw'].lt(-1e-12).fillna(False)
    has_variance_violation = bool(atm_rows['variance_monotone_violation'].any())
    if has_variance_violation:
        iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
        atm_rows['atm_total_variance'] = iso.fit_transform(atm_rows['T'].to_numpy(), atm_rows['atm_total_variance_raw'].to_numpy())
    else:
        atm_rows['atm_total_variance'] = atm_rows['atm_total_variance_raw']
    atm_rows['atm_total_variance_adjustment'] = atm_rows['atm_total_variance'] - atm_rows['atm_total_variance_raw']
    return call_surface, atm_rows, has_variance_violation

def select_calibration_quotes(option_chain, atm_rows, min_daily_volume=DEFAULT_MIN_DAILY_VOLUME):
    FIT_MAX_DTE = DEFAULT_MAX_DTE
    FIT_MIN_DAILY_VOLUME = min_daily_volume
    FIT_MIN_VEGA_OVER_SQRT_T = 0.001
    FIT_MIN_QUOTES_PER_EXPIRY = 20
    FIT_MIN_NORMALIZED_K = 0.45
    sanos_market_quotes = option_chain.loc[option_chain['T'].gt(0) & option_chain['dte'].le(FIT_MAX_DTE) & option_chain['k'].ge(FIT_MIN_NORMALIZED_K) & (option_chain['cp_flag'].eq('C') & option_chain['k'].ge(1.0) | option_chain['cp_flag'].eq('P') & option_chain['k'].lt(1.0)) & option_chain['best_offer'].gt(option_chain['best_bid']) & option_chain['best_bid'].ge(0) & option_chain['pure_bid'].ge(0) & option_chain['pure_ask'].gt(option_chain['pure_bid']) & np.isfinite(option_chain['k']) & np.isfinite(option_chain['pure_mid'])].copy()
    sanos_market_quotes['source_otm_iv'] = [implied_vol_from_pure_option(k, price, T, cp) for (k, price, T, cp) in zip(sanos_market_quotes['k'], sanos_market_quotes['pure_mid'], sanos_market_quotes['T'], sanos_market_quotes['cp_flag'])]
    sanos_market_quotes['source_vega'] = [bs_vega_forward_unit(k, iv, T) for (k, iv, T) in zip(sanos_market_quotes['k'], sanos_market_quotes['source_otm_iv'], sanos_market_quotes['T'])]
    sanos_market_quotes['source_vega_over_sqrt_t'] = sanos_market_quotes['source_vega'] / np.sqrt(sanos_market_quotes['T'])
    sanos_market_quotes = sanos_market_quotes.loc[np.isfinite(sanos_market_quotes['source_otm_iv']) & np.isfinite(sanos_market_quotes['source_vega_over_sqrt_t']) & sanos_market_quotes['source_vega_over_sqrt_t'].ge(FIT_MIN_VEGA_OVER_SQRT_T)].copy()
    model_anchor_quotes = sanos_market_quotes.copy()
    if FIT_MIN_DAILY_VOLUME is not None:
        sanos_market_quotes = sanos_market_quotes.loc[sanos_market_quotes['volume'].ge(FIT_MIN_DAILY_VOLUME)].copy()
    active_expiries = sanos_market_quotes.groupby('exdate')['k'].size().loc[lambda s: s.ge(FIT_MIN_QUOTES_PER_EXPIRY)].index
    sanos_market_quotes = sanos_market_quotes.loc[sanos_market_quotes['exdate'].isin(active_expiries)].copy()
    put_to_call_shift = np.where(sanos_market_quotes['cp_flag'].eq('P'), 1.0 - sanos_market_quotes['k'], 0.0)
    sanos_market_quotes['source_cp_flag'] = sanos_market_quotes['cp_flag']
    sanos_market_quotes['source_pure_bid'] = sanos_market_quotes['pure_bid']
    sanos_market_quotes['source_pure_mid'] = sanos_market_quotes['pure_mid']
    sanos_market_quotes['source_pure_ask'] = sanos_market_quotes['pure_ask']
    sanos_market_quotes['pure_bid'] = sanos_market_quotes['pure_bid'] + put_to_call_shift
    sanos_market_quotes['pure_mid'] = sanos_market_quotes['pure_mid'] + put_to_call_shift
    sanos_market_quotes['pure_ask'] = sanos_market_quotes['pure_ask'] + put_to_call_shift
    sanos_market_quotes['pure_bid'] = sanos_market_quotes['pure_bid'].clip(lower=0.0)
    sanos_market_quotes['pure_mid'] = sanos_market_quotes['pure_mid'].clip(lower=0.0)
    sanos_market_quotes['pure_ask'] = np.maximum(sanos_market_quotes['pure_ask'], sanos_market_quotes['pure_bid'] + 1e-12)
    sanos_quotes = sanos_market_quotes.merge(atm_rows[['exdate', 'atm_k', 'atm_iv', 'atm_total_variance_raw', 'atm_total_variance', 'atm_total_variance_adjustment', 'variance_monotone_violation']], on='exdate', how='left')
    sanos_quotes = sanos_quotes[['date', 'exdate', 'dte', 'T', 'symbol', 'source_cp_flag', 'strike', 'k', 'best_bid', 'best_offer', 'mid', 'source_pure_bid', 'source_pure_mid', 'source_pure_ask', 'source_otm_iv', 'source_vega_over_sqrt_t', 'pure_bid', 'pure_mid', 'pure_ask', 'impl_volatility', 'forward', 'df', 'atm_k', 'atm_iv', 'atm_total_variance_raw', 'atm_total_variance', 'atm_total_variance_adjustment', 'variance_monotone_violation', 'volume', 'open_interest', 'bid_ask_spread']].sort_values(['exdate', 'k'], kind='mergesort').reset_index(drop=True)
    return sanos_quotes, model_anchor_quotes

def build_joint_inputs(sanos_quotes, model_anchor_quotes, eta=.25):
    ETA = eta
    MIN_QUOTES_PER_EXPIRY = 8
    sanos_expiry_inputs = {}
    for (exdate, one) in sanos_quotes.groupby('exdate', sort=True):
        one = one.sort_values('k', kind='mergesort').reset_index(drop=True)
        one = one.dropna(subset=['k', 'pure_bid', 'pure_mid', 'pure_ask', 'atm_total_variance'])
        if len(one) < MIN_QUOTES_PER_EXPIRY:
            continue
        market_k = one['k'].to_numpy(dtype=float)
        model_market_k = np.unique(model_anchor_quotes.loc[model_anchor_quotes['exdate'].eq(exdate), 'k'].to_numpy(dtype=float))
        anchor_k = model_market_k.copy()
        V_j = float(one['atm_total_variance'].iloc[0])
        smooth_variance = ETA * V_j
        G_j = build_kernel_matrix(anchor_k=anchor_k, market_k=market_k, total_variance=smooth_variance)
        sanos_expiry_inputs[pd.Timestamp(exdate)] = {'dte': int(one['dte'].iloc[0]), 'T': float(one['T'].iloc[0]), 'V_j': V_j, 'smooth_variance': smooth_variance, 'market_k': market_k, 'anchor_k': anchor_k, 'model_market_k': model_market_k, 'pure_bid': one['pure_bid'].to_numpy(dtype=float), 'pure_mid': one['pure_mid'].to_numpy(dtype=float), 'pure_ask': one['pure_ask'].to_numpy(dtype=float), 'G': G_j, 'quotes': one}
    kernel_summary = pd.DataFrame([{'exdate': exdate, 'dte': item['dte'], 'T': item['T'], 'n_market': len(item['market_k']), 'n_anchor': len(item['anchor_k']), 'min_k': item['market_k'].min(), 'max_k': item['market_k'].max(), 'V_j': item['V_j'], 'smooth_variance': item['smooth_variance'], 'kernel_shape': item['G'].shape} for (exdate, item) in sanos_expiry_inputs.items()])
    from vol_surface_utils import DEFAULT_MODEL_MAX_GAP
    from sanos_core import build_model_anchor_grid, add_common_boundary_anchors
    JOINT_MAX_EXPIRIES = None
    MODEL_MAX_GAP = DEFAULT_MODEL_MAX_GAP
    CALENDAR_DENSE_GRID_POINTS = 801
    CALENDAR_TOL = 1e-07
    ordered_exdates = kernel_summary.sort_values('dte')['exdate']
    joint_exdates = ordered_exdates.tolist() if JOINT_MAX_EXPIRIES is None else ordered_exdates.head(JOINT_MAX_EXPIRIES).tolist()
    joint_inputs = {}
    for exdate in joint_exdates:
        item = sanos_expiry_inputs[pd.Timestamp(exdate)].copy()
        item['anchor_k'] = build_model_anchor_grid(item['model_market_k'], max_gap=MODEL_MAX_GAP)
        item['G'] = build_kernel_matrix(item['anchor_k'], item['market_k'], item['smooth_variance'])
        joint_inputs[pd.Timestamp(exdate)] = item
    joint_inputs = add_common_boundary_anchors(joint_inputs)
    GLOBAL_CALENDAR_K_MIN = min((float(np.min(item['market_k'])) for item in joint_inputs.values()))
    GLOBAL_CALENDAR_K_MAX = max((float(np.max(item['market_k'])) for item in joint_inputs.values()))
    return joint_inputs

def build_baseline_inputs(trade_date, *, data_path=None, yield_dir=None, min_daily_volume=DEFAULT_MIN_DAILY_VOLUME):
    TRADE_DATE = pd.Timestamp(trade_date)
    DATA_PATH = Path(data_path) if data_path is not None else Path('D:\\Projects\\data\\options_data\\spx_index_2025.parquet')
    YIELD_DIR = Path(yield_dir) if yield_dir is not None else Path('D:\\Projects\\data\\yield_data')
    PRODUCT_PREFIX = 'SPXW'
    FIT_MAX_DTE = DEFAULT_MAX_DTE
    ETA = 0.25
    columns = ['date', 'ticker', 'symbol', 'exdate', 'cp_flag', 'strike', 'strike_price', 'best_bid', 'best_offer', 'mid', 'bid_ask_spread', 'volume', 'open_interest', 'impl_volatility', 'delta', 'gamma', 'vega', 'theta', 'dte', 'exercise_style']
    raw = pd.read_parquet(DATA_PATH, columns=columns, filters=[('date', '==', TRADE_DATE)])
    option_chain = raw.loc[raw['date'].eq(TRADE_DATE) & raw['dte'].between(1, FIT_MAX_DTE) & raw['symbol'].astype('string').str.startswith(PRODUCT_PREFIX, na=False)].sort_values(['strike', 'exdate', 'cp_flag', 'symbol'], kind='mergesort').reset_index(drop=True)
    yield_files = {'DGS1MO': 1 / 12, 'DGS3MO': 3 / 12, 'DGS6MO': 6 / 12, 'DGS1': 1.0, 'DGS2': 2.0}
    curve_points = []
    for (series, tenor_years) in yield_files.items():
        rates = pd.read_csv(YIELD_DIR / f'{series}.csv', parse_dates=['observation_date'])
        row = rates.loc[rates['observation_date'].eq(TRADE_DATE), series]
        if not row.empty and pd.notna(row.iloc[0]):
            curve_points.append((tenor_years, row.iloc[0] / 100.0))
    yield_curve = pd.DataFrame(curve_points, columns=['tenor_years', 'rate'])

    def interpolated_rate(years: pd.Series) -> np.ndarray:
        xp = yield_curve['tenor_years'].to_numpy()
        fp = yield_curve['rate'].to_numpy()
        return np.interp(years.to_numpy(dtype=float), xp, fp)
    expiry_terms = option_chain[['exdate', 'dte']].drop_duplicates().sort_values('exdate').reset_index(drop=True)
    expiry_terms['T'] = expiry_terms['dte'] / 365.0
    expiry_terms['rate'] = interpolated_rate(expiry_terms['T'])
    expiry_terms['df'] = np.exp(-expiry_terms['rate'] * expiry_terms['T'])
    paired_mid = option_chain.pivot_table(index=['exdate', 'dte', 'strike'], columns='cp_flag', values=['mid', 'best_bid', 'best_offer', 'bid_ask_spread'], aggfunc='mean')
    paired_mid.columns = [f'{flag}_{field}' for (field, flag) in paired_mid.columns]
    paired_mid = paired_mid.reset_index()
    paired_mid = paired_mid.merge(expiry_terms[['exdate', 'T', 'rate', 'df']], on='exdate', how='left')
    paired_mid = paired_mid.dropna(subset=['C_mid', 'P_mid', 'C_best_bid', 'P_best_bid', 'C_best_offer', 'P_best_offer', 'C_bid_ask_spread', 'P_bid_ask_spread', 'df'])
    paired_mid = paired_mid.loc[paired_mid['C_best_offer'].gt(paired_mid['C_best_bid']) & paired_mid['P_best_offer'].gt(paired_mid['P_best_bid']) & paired_mid['C_bid_ask_spread'].gt(0) & paired_mid['P_bid_ask_spread'].gt(0)].copy()
    paired_mid['pair_spread'] = paired_mid['C_bid_ask_spread'] + paired_mid['P_bid_ask_spread']
    paired_mid['implied_forward_at_strike'] = paired_mid['strike'] + (paired_mid['C_mid'] - paired_mid['P_mid']) / paired_mid['df']

    def weighted_median(values: pd.Series, weights: pd.Series) -> float:
        order = np.argsort(values.to_numpy(dtype=float))
        sorted_values = values.to_numpy(dtype=float)[order]
        sorted_weights = weights.to_numpy(dtype=float)[order]
        cutoff = 0.5 * sorted_weights.sum()
        return sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff)]

    def select_forward_quotes(expiry_pairs: pd.DataFrame, atm_width: float=0.05, spread_quantile: float=0.6, min_points: int=8) -> pd.DataFrame:
        clean = expiry_pairs.replace([np.inf, -np.inf], np.nan).dropna(subset=['strike', 'pair_spread', 'implied_forward_at_strike'])
        clean = clean.loc[clean['pair_spread'].gt(0)].copy()
        if len(clean) < min_points:
            return clean
        f0 = clean['implied_forward_at_strike'].median()
        atm = clean.loc[(clean['strike'] / f0 - 1.0).abs().le(atm_width)].copy()
        if len(atm) < min_points:
            atm = clean.assign(distance=(clean['strike'] - f0).abs()).nsmallest(min_points, 'distance').copy()
        spread_cutoff = atm['pair_spread'].quantile(spread_quantile)
        filtered = atm.loc[atm['pair_spread'].le(spread_cutoff)].copy()
        if len(filtered) < min_points:
            filtered = atm.copy()
        return filtered

    def atm_filtered_parity_forward(expiry_pairs: pd.DataFrame) -> pd.Series:
        filtered = select_forward_quotes(expiry_pairs)
        if filtered.empty:
            return pd.Series({'forward': np.nan, 'forward_points': 0})
        weights = 1.0 / filtered['pair_spread']
        forward = weighted_median(filtered['implied_forward_at_strike'], weights)
        return pd.Series({'forward': forward, 'forward_points': len(filtered)})
    forwards = paired_mid.groupby('exdate')[['strike', 'pair_spread', 'implied_forward_at_strike']].apply(atm_filtered_parity_forward).reset_index()
    expiry_terms = expiry_terms.merge(forwards, on='exdate', how='left')
    paired_quotes = paired_mid.copy()
    paired_quotes['parity_y'] = paired_quotes['C_mid'] - paired_quotes['P_mid']

    def fit_weighted_line(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
        X = np.column_stack([np.ones_like(x), x])
        Xw = X * np.sqrt(w)[:, None]
        yw = y * np.sqrt(w)
        return tuple(np.linalg.lstsq(Xw, yw, rcond=None)[0])

    def fit_huber_line(x: np.ndarray, y: np.ndarray, base_w: np.ndarray, max_iter: int=8, c: float=1.345) -> tuple[float, float]:
        robust_w = base_w.copy()
        (intercept, slope) = fit_weighted_line(x, y, robust_w)
        for _ in range(max_iter):
            residual = y - (intercept + slope * x)
            scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
            if not np.isfinite(scale) or scale <= 1e-12:
                break
            z = np.abs(residual) / scale
            huber_w = np.where(z <= c, 1.0, c / z)
            (new_intercept, new_slope) = fit_weighted_line(x, y, base_w * huber_w)
            if np.hypot(new_intercept - intercept, new_slope - slope) < 1e-10:
                (intercept, slope) = (new_intercept, new_slope)
                break
            (intercept, slope) = (new_intercept, new_slope)
        return (intercept, slope)

    def weighted_parity_regression(expiry_pairs: pd.DataFrame) -> pd.Series:
        clean = select_forward_quotes(expiry_pairs)
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna(subset=['strike', 'parity_y', 'pair_spread'])
        if len(clean) < 5:
            return pd.Series({'df_reg': np.nan, 'forward_reg': np.nan, 'df_robust': np.nan, 'forward_robust': np.nan, 'parity_points': len(clean), 'parity_rmse': np.nan, 'parity_rmse_robust': np.nan})
        x = clean['strike'].to_numpy(dtype=float)
        y = clean['parity_y'].to_numpy(dtype=float)
        w = 1.0 / clean['pair_spread'].to_numpy(dtype=float)
        (intercept, slope) = fit_weighted_line(x, y, w)
        (intercept_robust, slope_robust) = fit_huber_line(x, y, w)
        df_reg = -slope
        forward_reg = intercept / df_reg if df_reg > 0 else np.nan
        df_robust = -slope_robust
        forward_robust = intercept_robust / df_robust if df_robust > 0 else np.nan
        fitted = intercept + slope * x
        fitted_robust = intercept_robust + slope_robust * x
        rmse = np.sqrt(np.average((y - fitted) ** 2, weights=w))
        rmse_robust = np.sqrt(np.average((y - fitted_robust) ** 2, weights=w))
        return pd.Series({'df_reg': df_reg, 'forward_reg': forward_reg, 'df_robust': df_robust, 'forward_robust': forward_robust, 'parity_points': len(clean), 'parity_rmse': rmse, 'parity_rmse_robust': rmse_robust})
    parity_regression = paired_quotes.groupby('exdate')[['strike', 'parity_y', 'pair_spread', 'implied_forward_at_strike']].apply(weighted_parity_regression).reset_index()
    method_compare = expiry_terms.merge(parity_regression, on='exdate', how='left')
    method_compare['forward_diff'] = method_compare['forward_reg'] - method_compare['forward']
    method_compare['df_diff'] = method_compare['df_reg'] - method_compare['df']
    method_compare['forward_robust_diff'] = method_compare['forward_robust'] - method_compare['forward']
    method_compare['df_robust_diff'] = method_compare['df_robust'] - method_compare['df']
    option_chain = option_chain.merge(expiry_terms[['exdate', 'T', 'rate', 'df', 'forward']], on='exdate', how='left')
    option_chain['k'] = option_chain['strike'] / option_chain['forward']
    option_chain['pure_bid'] = option_chain['best_bid'] / (option_chain['df'] * option_chain['forward'])
    option_chain['pure_mid'] = option_chain['mid'] / (option_chain['df'] * option_chain['forward'])
    option_chain['pure_ask'] = option_chain['best_offer'] / (option_chain['df'] * option_chain['forward'])
    option_chain = option_chain.sort_values(['k', 'exdate', 'cp_flag', 'symbol'], kind='mergesort').reset_index(drop=True)
    call_surface = option_chain.loc[option_chain['cp_flag'].eq('C') & option_chain['best_offer'].gt(option_chain['best_bid']) & option_chain['best_bid'].ge(0) & option_chain['pure_bid'].ge(0) & option_chain['pure_ask'].gt(option_chain['pure_bid']) & np.isfinite(option_chain['k']) & np.isfinite(option_chain['pure_mid'])].copy()
    call_surface = call_surface.sort_values(['exdate', 'k'], kind='mergesort').reset_index(drop=True)
    (call_surface, atm_rows, has_variance_violation) = build_variance_backbone(call_surface)
    (sanos_quotes, model_anchor_quotes) = select_calibration_quotes(option_chain, atm_rows, min_daily_volume)
    if sanos_quotes.empty:
        raise ValueError(f'No eligible quotes for {trade_date}.')
    joint_inputs = build_joint_inputs(sanos_quotes, model_anchor_quotes, ETA)
    return locals()
