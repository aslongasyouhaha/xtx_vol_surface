from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_MAX_DTE = 120
DEFAULT_MIN_DAILY_VOLUME = 100  # Paper's end-of-day sample selection.
DEFAULT_MODEL_MAX_GAP = 0.01  # Normalized strike spacing; our explicit engineering choice.
DEFAULT_TENOR_MONTHS = (1, 3, 6, 12, 24)
DEFAULT_YIELD_FILES = {
    1: "DGS1MO.csv",
    3: "DGS3MO.csv",
    6: "DGS6MO.csv",
    12: "DGS1.csv",
    24: "DGS2.csv",
}
DEFAULT_YIELD_COLUMNS = {
    1: "DGS1MO",
    3: "DGS3MO",
    6: "DGS6MO",
    12: "DGS1",
    24: "DGS2",
}


@dataclass(frozen=True)
class DiscountCurvePoint:
    """Interpolated rate and discount factor for one expiry."""

    ttm: float
    rate: float
    discount_factor: float


@dataclass(frozen=True)
class ForwardFilterConfig:
    """Filters used when estimating forwards from call-put parity."""

    min_bid: float = 0.0
    min_mid: float = 0.01
    max_pair_spread: float | None = None
    max_relative_pair_spread: float | None = 0.50
    moneyness_window: float | None = 0.25
    outlier_mad_threshold: float | None = 5.0
    min_pairs: int = 5


def interpolate_discount_factor(
    ttm: float | Sequence[float] | pd.Series | np.ndarray,
    rates: Mapping[int, float] | pd.Series | pd.DataFrame,
    *,
    tenor_months: Sequence[int] = DEFAULT_TENOR_MONTHS,
    rate_columns: Mapping[int, str] | None = None,
    interpolation: str = "log_df",
    compounding: str = "continuous",
    extrapolate: str = "flat_forward",
) -> DiscountCurvePoint | pd.DataFrame:
    """Interpolate rates and discount factors for arbitrary option maturities.

    Parameters
    ----------
    ttm:
        Time to maturity in years, e.g. ``dte / 365``. Can be scalar or array-like.
    rates:
        Either a mapping/Series from tenor month to annualized decimal rate, or a
        one-row DataFrame. Rates should be decimals, e.g. ``0.052`` for 5.2%.
    tenor_months:
        Available rate tenors in months. Defaults to 1, 3, 6, 12, 24.
    rate_columns:
        Optional mapping from tenor month to DataFrame column name. Use this if
        your rate table has columns such as ``rate_1m`` or ``DGS1MO``.
    interpolation:
        ``log_df`` linearly interpolates log discount factors. This is the
        default and includes a ``T=0, DF=1`` anchor, so sub-1M maturities are
        handled naturally. ``rate`` linearly interpolates annualized rates.
    compounding:
        Compounding convention used to convert input rates to tenor discount
        factors. ``continuous`` uses ``exp(-rT)``. ``simple`` uses
        ``1 / (1 + rT)``.
    extrapolate:
        For ``log_df`` interpolation, ``flat_forward`` extends the final
        log-DF slope beyond the last tenor, while ``flat_zero`` uses the last
        zero rate. For ``rate`` interpolation, ``flat`` clamps outside the tenor
        range and ``linear`` extends the endpoint slope.

    Returns
    -------
    DiscountCurvePoint for scalar input, otherwise a DataFrame with columns
    ``ttm``, ``rate``, and ``discount_factor``.
    """

    tenor_years = np.asarray(tenor_months, dtype=float) / 12.0
    curve_rates = _coerce_rates(rates, tenor_months, rate_columns)

    x = np.asarray(ttm, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)

    if np.any(x < 0):
        raise ValueError("ttm must be non-negative")

    if interpolation == "log_df":
        discount_factors = _discount_factors_from_rates(curve_rates, tenor_years, compounding)
        log_df_nodes = np.concatenate([[0.0], np.log(discount_factors)])
        time_nodes = np.concatenate([[0.0], tenor_years])
        log_df = np.interp(x, time_nodes, log_df_nodes)

        right = x > time_nodes[-1]
        if np.any(right):
            if extrapolate == "flat_forward":
                slope = (log_df_nodes[-1] - log_df_nodes[-2]) / (time_nodes[-1] - time_nodes[-2])
                log_df[right] = log_df_nodes[-1] + slope * (x[right] - time_nodes[-1])
            elif extrapolate == "flat_zero":
                log_df[right] = -curve_rates[-1] * x[right]
            else:
                raise ValueError("log_df extrapolate must be 'flat_forward' or 'flat_zero'")
        elif extrapolate not in {"flat_forward", "flat_zero"}:
            raise ValueError("log_df extrapolate must be 'flat_forward' or 'flat_zero'")

        discount_factors = np.exp(log_df)
        interp_rates = np.divide(
            -log_df,
            x,
            out=np.full_like(x, curve_rates[0], dtype=float),
            where=x > 0,
        )
    elif interpolation == "rate":
        interp_rates = np.interp(x, tenor_years, curve_rates)
        if extrapolate == "linear":
            interp_rates = _linear_extrapolate(x, tenor_years, curve_rates, interp_rates)
        elif extrapolate != "flat":
            raise ValueError("rate extrapolate must be 'flat' or 'linear'")
        discount_factors = _discount_factors_from_rates(interp_rates, x, compounding)
    else:
        raise ValueError("interpolation must be 'log_df' or 'rate'")

    if scalar:
        return DiscountCurvePoint(
            ttm=float(x[0]),
            rate=float(interp_rates[0]),
            discount_factor=float(discount_factors[0]),
        )

    return pd.DataFrame(
        {
            "ttm": x,
            "rate": interp_rates,
            "discount_factor": discount_factors,
        }
    )


def add_discount_factors(
    expiries: pd.DataFrame,
    rates: Mapping[int, float] | pd.Series | pd.DataFrame,
    *,
    date_col: str = "date",
    exdate_col: str = "exdate",
    year_basis: float = 365.0,
    tenor_months: Sequence[int] = DEFAULT_TENOR_MONTHS,
    rate_columns: Mapping[int, str] | None = None,
    compounding: str = "continuous",
    interpolation: str = "log_df",
    extrapolate: str = "flat_forward",
) -> pd.DataFrame:
    """Add ``dte``, ``ttm``, ``rate``, and ``discount_factor`` to an expiry table."""

    out = expiries.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    out[exdate_col] = pd.to_datetime(out[exdate_col]).dt.normalize()
    out["dte"] = (out[exdate_col] - out[date_col]).dt.days
    out["ttm"] = out["dte"] / year_basis

    curve = interpolate_discount_factor(
        out["ttm"].to_numpy(),
        rates,
        tenor_months=tenor_months,
        rate_columns=rate_columns,
        interpolation=interpolation,
        compounding=compounding,
        extrapolate=extrapolate,
    )
    out["rate"] = curve["rate"].to_numpy()
    out["discount_factor"] = curve["discount_factor"].to_numpy()
    return out


def estimate_forward_by_expiry(
    chain: pd.DataFrame,
    discount_factors: pd.DataFrame,
    *,
    date_col: str = "date",
    exdate_col: str = "exdate",
    cp_col: str = "cp_flag",
    strike_col: str = "strike",
    strike_price_col: str = "strike_price",
    bid_col: str = "best_bid",
    ask_col: str = "best_offer",
    discount_col: str = "discount_factor",
    filter_config: ForwardFilterConfig | None = None,
    weighted: bool = True,
    return_pairs: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate one forward ``F(T)`` per expiry from call-put parity.

    For each same-expiry same-strike call/put pair, the implied forward is

    ``F_i(T) = K_i + (C_mid_i - P_mid_i) / DF(T)``.

    The expiry-level forward is the median or weighted median of the filtered
    ``F_i`` values. Weights default to inverse total bid-ask spread.
    """

    cfg = filter_config or ForwardFilterConfig()
    prepared = _prepare_chain_for_forward(
        chain,
        date_col=date_col,
        exdate_col=exdate_col,
        cp_col=cp_col,
        strike_col=strike_col,
        strike_price_col=strike_price_col,
        bid_col=bid_col,
        ask_col=ask_col,
    )
    dfs = discount_factors[[date_col, exdate_col, discount_col]].copy()
    dfs[date_col] = pd.to_datetime(dfs[date_col]).dt.normalize()
    dfs[exdate_col] = pd.to_datetime(dfs[exdate_col]).dt.normalize()

    pairs = _call_put_pairs(
        prepared,
        date_col=date_col,
        exdate_col=exdate_col,
        cp_col=cp_col,
        strike_col=strike_col,
        bid_col=bid_col,
        ask_col=ask_col,
    )
    pairs = pairs.merge(dfs, on=[date_col, exdate_col], how="inner")
    if pairs.empty:
        empty_forwards = pd.DataFrame(
            columns=[date_col, exdate_col, "forward", "num_pairs", "num_pairs_raw", "median_abs_error"]
        )
        return (empty_forwards, pairs) if return_pairs else empty_forwards

    pairs["pair_spread"] = pairs["call_spread"] + pairs["put_spread"]
    pairs["pair_mid"] = pairs["call_mid"] + pairs["put_mid"]
    pairs["relative_pair_spread"] = np.divide(
        pairs["pair_spread"],
        pairs["pair_mid"],
        out=np.full(len(pairs), np.nan, dtype=float),
        where=pairs["pair_mid"].to_numpy() > 0,
    )
    pairs["forward_i"] = pairs[strike_col] + (pairs["call_mid"] - pairs["put_mid"]) / pairs[discount_col]
    pairs["weight"] = 1.0 / np.maximum(pairs["pair_spread"], 1e-12)

    rows = []
    kept_pairs = []
    for keys, group in pairs.groupby([date_col, exdate_col], sort=True):
        raw_count = len(group)
        filtered = _filter_forward_pairs(group, cfg, strike_col=strike_col)
        if len(filtered) < cfg.min_pairs:
            continue

        if weighted:
            forward = _weighted_median(filtered["forward_i"].to_numpy(), filtered["weight"].to_numpy())
        else:
            forward = float(filtered["forward_i"].median())

        filtered = filtered.copy()
        filtered["forward"] = forward
        filtered["forward_error"] = filtered["forward_i"] - forward
        kept_pairs.append(filtered)

        date_value, exdate_value = keys
        rows.append(
            {
                date_col: date_value,
                exdate_col: exdate_value,
                "forward": forward,
                "num_pairs": len(filtered),
                "num_pairs_raw": raw_count,
                "median_abs_error": float(np.median(np.abs(filtered["forward_error"]))),
                "max_abs_error": float(np.max(np.abs(filtered["forward_error"]))),
            }
        )

    forwards = pd.DataFrame(rows)
    if not forwards.empty:
        forwards = forwards.sort_values([date_col, exdate_col]).reset_index(drop=True)
    pair_details = pd.concat(kept_pairs, ignore_index=True) if kept_pairs else pairs.iloc[0:0].copy()
    return (forwards, pair_details) if return_pairs else forwards


def estimate_robust_forward_by_expiry(
    chain: pd.DataFrame,
    discount_factors: pd.DataFrame,
    *,
    filter_config: ForwardFilterConfig | None = None,
    **kwargs,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate robust expiry forwards with practical quote-quality filters.

    This is a convenience wrapper around ``estimate_forward_by_expiry`` with
    weighted medians enabled. Use ``ForwardFilterConfig`` to tune liquidity,
    spread, moneyness-window, and outlier filters.
    """

    cfg = filter_config or ForwardFilterConfig(
        min_bid=0.05,
        min_mid=0.05,
        max_relative_pair_spread=0.35,
        moneyness_window=0.20,
        outlier_mad_threshold=4.0,
        min_pairs=5,
    )
    return estimate_forward_by_expiry(
        chain,
        discount_factors,
        filter_config=cfg,
        weighted=True,
        **kwargs,
    )


def build_normalized_call_quotes(
    chain: pd.DataFrame,
    forwards: pd.DataFrame,
    discount_factors: pd.DataFrame,
    *,
    date_col: str = "date",
    exdate_col: str = "exdate",
    cp_col: str = "cp_flag",
    strike_col: str = "strike",
    strike_price_col: str = "strike_price",
    bid_col: str = "best_bid",
    ask_col: str = "best_offer",
    forward_col: str = "forward",
    discount_col: str = "discount_factor",
    source_otm_only: bool = False,
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """Convert call/put quotes to forward-normalized call quotes.

    Calls are kept as calls. Puts are converted with put-call parity:

    ``C_from_put = P + DF(T) * (F(T) - K)``.

    The output uses normalized strike and call prices:

    ``k_norm = K / F(T)`` and ``c_norm = C / (DF(T) * F(T))``.
    """

    prepared = _prepare_chain_for_forward(
        chain,
        date_col=date_col,
        exdate_col=exdate_col,
        cp_col=cp_col,
        strike_col=strike_col,
        strike_price_col=strike_price_col,
        bid_col=bid_col,
        ask_col=ask_col,
    )

    fwd = forwards[[date_col, exdate_col, forward_col]].copy()
    fwd[date_col] = pd.to_datetime(fwd[date_col]).dt.normalize()
    fwd[exdate_col] = pd.to_datetime(fwd[exdate_col]).dt.normalize()
    dfs = discount_factors[[date_col, exdate_col, discount_col]].copy()
    dfs[date_col] = pd.to_datetime(dfs[date_col]).dt.normalize()
    dfs[exdate_col] = pd.to_datetime(dfs[exdate_col]).dt.normalize()

    out = prepared.merge(fwd, on=[date_col, exdate_col], how="inner")
    out = out.merge(dfs, on=[date_col, exdate_col], how="inner")
    out = out[out[forward_col] > 0].copy()
    out["ttm"] = (out[exdate_col] - out[date_col]).dt.days / 365.0
    out["k_norm"] = out[strike_col] / out[forward_col]

    parity_shift = out[discount_col] * (out[forward_col] - out[strike_col])
    is_call = out[cp_col] == "C"
    is_put = out[cp_col] == "P"

    out["call_bid"] = np.where(is_call, out[bid_col], out[bid_col] + parity_shift)
    out["call_ask"] = np.where(is_call, out[ask_col], out[ask_col] + parity_shift)
    out["call_mid"] = (out["call_bid"] + out["call_ask"]) / 2.0
    out["call_source"] = np.where(is_call, "call", "put_synthetic_call")

    denominator = out[discount_col] * out[forward_col]
    out["call_bid_norm"] = out["call_bid"] / denominator
    out["call_mid_norm"] = out["call_mid"] / denominator
    out["call_ask_norm"] = out["call_ask"] / denominator

    if source_otm_only:
        out = out[((is_put & (out["k_norm"] <= 1.0)) | (is_call & (out["k_norm"] > 1.0)))].copy()

    if drop_invalid:
        out = out[
            (out["ttm"] > 0)
            & (out["call_ask"] > out["call_bid"])
            & (out["call_ask_norm"] > out["call_bid_norm"])
            & (out["call_ask_norm"] > 0)
            & np.isfinite(out["k_norm"])
            & np.isfinite(out["call_mid_norm"])
        ].copy()

    keep_cols = [
        date_col,
        exdate_col,
        "ttm",
        "k_norm",
        strike_col,
        forward_col,
        discount_col,
        cp_col,
        "call_source",
        "call_bid",
        "call_mid",
        "call_ask",
        "call_bid_norm",
        "call_mid_norm",
        "call_ask_norm",
    ]
    optional_cols = [col for col in ["impl_volatility", "delta", "vega", "volume", "open_interest"] if col in out.columns]
    return out[keep_cols + optional_cols].sort_values([date_col, exdate_col, "k_norm", "call_source"]).reset_index(drop=True)


def build_normalized_surface_input(
    chain: pd.DataFrame,
    rates: Mapping[int, float] | pd.Series | pd.DataFrame,
    *,
    date_col: str = "date",
    exdate_col: str = "exdate",
    cp_col: str = "cp_flag",
    strike_col: str = "strike",
    strike_price_col: str = "strike_price",
    bid_col: str = "best_bid",
    ask_col: str = "best_offer",
    tenor_months: Sequence[int] = DEFAULT_TENOR_MONTHS,
    rate_columns: Mapping[int, str] | None = None,
    year_basis: float = 365.0,
    source_otm_only: bool = True,
    forward_filter_config: ForwardFilterConfig | None = None,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build SANOS-ready normalized call quotes from one option chain.

    This composes the first surface-preparation steps:

    1. interpolate ``DF(T)`` for each expiry from the input rate curve;
    2. estimate robust expiry forwards using call-put parity;
    3. convert puts to synthetic calls and normalize strikes/prices.

    Returns normalized quotes by default. With ``return_diagnostics=True`` it
    returns ``(quotes, forwards, discount_factors, forward_pairs)``.
    Otherwise it returns ``(quotes, forwards, discount_factors)``.
    """

    expiries = chain[[date_col, exdate_col]].drop_duplicates().copy()
    discount_factors = add_discount_factors(
        expiries,
        rates,
        date_col=date_col,
        exdate_col=exdate_col,
        year_basis=year_basis,
        tenor_months=tenor_months,
        rate_columns=rate_columns,
        interpolation="log_df",
        extrapolate="flat_forward",
    )

    if return_diagnostics:
        forwards, forward_pairs = estimate_robust_forward_by_expiry(
            chain,
            discount_factors,
            date_col=date_col,
            exdate_col=exdate_col,
            cp_col=cp_col,
            strike_col=strike_col,
            strike_price_col=strike_price_col,
            bid_col=bid_col,
            ask_col=ask_col,
            filter_config=forward_filter_config,
            return_pairs=True,
        )
    else:
        forwards = estimate_robust_forward_by_expiry(
            chain,
            discount_factors,
            date_col=date_col,
            exdate_col=exdate_col,
            cp_col=cp_col,
            strike_col=strike_col,
            strike_price_col=strike_price_col,
            bid_col=bid_col,
            ask_col=ask_col,
            filter_config=forward_filter_config,
        )
        forward_pairs = None

    quotes = build_normalized_call_quotes(
        chain,
        forwards,
        discount_factors,
        date_col=date_col,
        exdate_col=exdate_col,
        cp_col=cp_col,
        strike_col=strike_col,
        strike_price_col=strike_price_col,
        bid_col=bid_col,
        ask_col=ask_col,
        source_otm_only=source_otm_only,
    )

    if return_diagnostics:
        return quotes, forwards, discount_factors, forward_pairs
    return quotes, forwards, discount_factors


def load_treasury_rates_for_date(
    date: str | pd.Timestamp,
    yield_dir: str | Path,
    *,
    tenor_months: Sequence[int] = DEFAULT_TENOR_MONTHS,
    file_map: Mapping[int, str] = DEFAULT_YIELD_FILES,
    column_map: Mapping[int, str] = DEFAULT_YIELD_COLUMNS,
    allow_previous: bool = True,
) -> pd.Series:
    """Load 1M/3M/6M/1Y/2Y Treasury rates for one date as decimals.

    The CSV files are expected to contain ``observation_date`` plus one yield
    column, with FRED-style percentage values. For example, ``5.25`` becomes
    ``0.0525``.
    """

    target = pd.Timestamp(date).normalize()
    yield_path = Path(yield_dir)
    values = {}
    used_dates = {}

    for tenor in tenor_months:
        csv_path = yield_path / file_map[tenor]
        col = column_map[tenor]
        df = pd.read_csv(csv_path, parse_dates=["observation_date"])
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=[col]).sort_values("observation_date")
        if allow_previous:
            eligible = df[df["observation_date"].dt.normalize() <= target]
            if eligible.empty:
                raise ValueError(f"No {col} yield available on or before {target.date()}")
            row = eligible.iloc[-1]
        else:
            eligible = df[df["observation_date"].dt.normalize() == target]
            if eligible.empty:
                raise ValueError(f"No {col} yield available on {target.date()}")
            row = eligible.iloc[0]

        values[tenor] = float(row[col]) / 100.0
        used_dates[f"{tenor}m_date"] = pd.Timestamp(row["observation_date"]).normalize()

    out = pd.Series(values, name=target)
    out.attrs["used_dates"] = used_dates
    return out


def _coerce_rates(
    rates: Mapping[int, float] | pd.Series | pd.DataFrame,
    tenor_months: Sequence[int],
    rate_columns: Mapping[int, str] | None,
) -> np.ndarray:
    if isinstance(rates, pd.DataFrame):
        if len(rates) != 1:
            raise ValueError("rates DataFrame must contain exactly one row")
        row = rates.iloc[0]
        if rate_columns is None:
            values = [row[m] for m in tenor_months]
        else:
            values = [row[rate_columns[m]] for m in tenor_months]
    elif isinstance(rates, pd.Series):
        values = [rates[m] for m in tenor_months]
    else:
        values = [rates[m] for m in tenor_months]

    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("rates contain missing or non-finite values")
    return values


def _prepare_chain_for_forward(
    chain: pd.DataFrame,
    *,
    date_col: str,
    exdate_col: str,
    cp_col: str,
    strike_col: str,
    strike_price_col: str,
    bid_col: str,
    ask_col: str,
) -> pd.DataFrame:
    out = chain.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    out[exdate_col] = pd.to_datetime(out[exdate_col]).dt.normalize()
    if strike_col not in out.columns:
        if strike_price_col not in out.columns:
            raise ValueError(f"chain must contain either {strike_col!r} or {strike_price_col!r}")
        out[strike_col] = out[strike_price_col] / 1000.0
    out["mid"] = (out[bid_col] + out[ask_col]) / 2.0
    out["spread"] = out[ask_col] - out[bid_col]
    out = out[out[cp_col].isin(["C", "P"])].copy()
    return out


def _call_put_pairs(
    chain: pd.DataFrame,
    *,
    date_col: str,
    exdate_col: str,
    cp_col: str,
    strike_col: str,
    bid_col: str,
    ask_col: str,
) -> pd.DataFrame:
    index_cols = [date_col, exdate_col, strike_col]
    values = [bid_col, ask_col, "mid", "spread"]
    piv = chain.pivot_table(index=index_cols, columns=cp_col, values=values, aggfunc="median")
    if ("mid", "C") not in piv.columns or ("mid", "P") not in piv.columns:
        return pd.DataFrame(columns=index_cols)

    pairs = pd.DataFrame(index=piv.index)
    pairs["call_bid"] = piv[(bid_col, "C")]
    pairs["call_ask"] = piv[(ask_col, "C")]
    pairs["call_mid"] = piv[("mid", "C")]
    pairs["call_spread"] = piv[("spread", "C")]
    pairs["put_bid"] = piv[(bid_col, "P")]
    pairs["put_ask"] = piv[(ask_col, "P")]
    pairs["put_mid"] = piv[("mid", "P")]
    pairs["put_spread"] = piv[("spread", "P")]
    return pairs.reset_index().dropna()


def _filter_forward_pairs(group: pd.DataFrame, cfg: ForwardFilterConfig, *, strike_col: str) -> pd.DataFrame:
    filtered = group[
        (group["call_bid"] >= cfg.min_bid)
        & (group["put_bid"] >= cfg.min_bid)
        & (group["call_ask"] > group["call_bid"])
        & (group["put_ask"] > group["put_bid"])
        & (group["call_mid"] >= cfg.min_mid)
        & (group["put_mid"] >= cfg.min_mid)
        & np.isfinite(group["forward_i"])
        & (group["forward_i"] > 0)
    ].copy()

    if cfg.max_pair_spread is not None:
        filtered = filtered[filtered["pair_spread"] <= cfg.max_pair_spread]
    if cfg.max_relative_pair_spread is not None:
        filtered = filtered[filtered["relative_pair_spread"] <= cfg.max_relative_pair_spread]
    if filtered.empty:
        return filtered

    rough_forward = float(filtered["forward_i"].median())
    if cfg.moneyness_window is not None and rough_forward > 0:
        mny = filtered[strike_col] / rough_forward
        filtered = filtered[(mny >= 1.0 - cfg.moneyness_window) & (mny <= 1.0 + cfg.moneyness_window)]
    if filtered.empty:
        return filtered

    if cfg.outlier_mad_threshold is not None:
        center = float(filtered["forward_i"].median())
        abs_dev = np.abs(filtered["forward_i"] - center)
        mad = float(np.median(abs_dev))
        if mad > 0:
            filtered = filtered[abs_dev <= cfg.outlier_mad_threshold * 1.4826 * mad]

    return filtered


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return float("nan")

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * weights.sum()
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def _discount_factors_from_rates(rates: np.ndarray, times: np.ndarray, compounding: str) -> np.ndarray:
    if compounding == "continuous":
        return np.exp(-rates * times)
    if compounding == "simple":
        return 1.0 / (1.0 + rates * times)
    raise ValueError("compounding must be 'continuous' or 'simple'")


def _linear_extrapolate(
    x: np.ndarray,
    tenor_years: np.ndarray,
    rates: np.ndarray,
    interp_rates: np.ndarray,
) -> np.ndarray:
    out = interp_rates.copy()

    left = x < tenor_years[0]
    if np.any(left):
        slope = (rates[1] - rates[0]) / (tenor_years[1] - tenor_years[0])
        out[left] = rates[0] + slope * (x[left] - tenor_years[0])

    right = x > tenor_years[-1]
    if np.any(right):
        slope = (rates[-1] - rates[-2]) / (tenor_years[-1] - tenor_years[-2])
        out[right] = rates[-1] + slope * (x[right] - tenor_years[-1])

    return out
