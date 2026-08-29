# XTX / SANOS Vol Surface Project

This project studies and engineers a reproducible implementation path for the
SANOS/XTX option surface construction method on SPXW option data.

The local paper copy is:

- `xtx vol surface.pdf`

The main research notebook is:

- `option_surface_step1.ipynb`

The final engineering entry point is:

- `reproduce_xtx_sanos_project.py`

## What This Project Builds

The goal is to transform filtered option quotes into a smooth, usable,
arbitrage-aware call price surface.

The implementation works in normalized forward coordinates:

```text
k = K / F(T)
pure_call = C / (DF(T) * F(T))
```

The fitting input is an OTM call-equivalent quote set:

```text
k < 1   : use OTM put, converted by parity into call-equivalent price
k >= 1  : use OTM call
```

The model fits pure call prices, not implied volatilities. IV plots are
diagnostics only.

## Final Pipeline

The final recommended pipeline is:

```text
1. Build filtered OTM call-equivalent quotes.
2. Run single-expiry LP diagnostics.
3. Run joint LP with calendar constraints.
4. Diagnose bid/ask coverage, outside error, calendar checks, and q stability.
5. Apply automatic regime-based remediation only to weak expiries.
6. Optionally run the wing-weighted QP stabilization for smoother q densities.
```

The baseline notebook originally used:

```text
ETA = 0.25
market-only anchors
```

The multi-day study suggests a more careful production rule:

```text
Use ETA=0.25 as a benchmark, not as a universal production default.
Use dense anchors only when diagnostics trigger remediation.
When anchors are dense, use smaller ETA values, typically 0.01 to 0.10.
Avoid very large ETA values such as 0.50 in production.
```

## Reproduction Commands

Run the main case study:

```bash
python reproduce_xtx_sanos_project.py --trade-date 2025-06-02
```

Run only automatic remediation:

```bash
python generate_xtx_auto_regime_remediation.py --trade-date 2025-06-02
```

Summarize the existing multi-day ETA study:

```bash
python summarize_xtx_multi_day_eta_partial.py
```

Run expiry-case forensics, defaulting to the 74D project example:

```bash
python generate_xtx_expiry_case_deep_diagnostics.py --target-dte 74
python generate_xtx_expiry_case_shape_diagnostics.py --target-dte 74
python generate_xtx_expiry_case_eta_sweep.py --target-dte 74
python generate_xtx_expiry_case_anchor_eta_sweep.py --target-dte 74
python generate_xtx_expiry_case_joint_candidate_sweep.py --target-dte 74
```

Run the full multi-day ETA sweep:

```bash
python reproduce_xtx_sanos_project.py --multi-day-regime
```

The full multi-day sweep is intentionally not part of the default quick path,
because dense-anchor LPs can be slow for some date/expiry combinations.

## Key Outputs

Core diagnostics:

- `xtx_diagnostics/diagnostic_summary.md`
- `xtx_diagnostics/01_joint_lp_fit_diagnostics.png`
- `xtx_diagnostics/05_pure_call_surface.png`
- `xtx_diagnostics/06_otm_iv_surface.png`

Expiry-case diagnostics, default example 74D:

- `xtx_diagnostics/expiry_case_deep_summary.md`
- `xtx_diagnostics/expiry_case_static_shape_summary.md`
- `xtx_diagnostics/expiry_case_joint_candidate_sweep_summary.md`
- `xtx_diagnostics/18_expiry_case_joint_candidate_sweep_metrics.png`

Automatic remediation:

- `xtx_diagnostics/auto_regime/auto_regime_summary.md`
- `xtx_diagnostics/auto_regime/auto_regime_before_after.png`
- `xtx_diagnostics/auto_regime/auto_regime_steps.csv`

Multi-day ETA regime study:

- `xtx_diagnostics/multi_day_eta/multi_day_eta_regime_summary.md`
- `xtx_diagnostics/multi_day_eta/multi_day_eta_regime_charts.png`
- `xtx_diagnostics/multi_day_eta/multi_day_dense_vs_market_improvement.csv`

## Main Findings

For the 2025-06-02 SPXW case, the original joint LP found 21 expiries from 1D
to 88D and produced a generally regular pure call price surface. The main weak
point was not the whole surface, but the 74D expiry.

The 74D expiry showed:

```text
baseline coverage        0.10
baseline mid RMSE        0.00117479
baseline max outside     0.00237702
```

Automatic regime remediation selected dense anchors and smaller ETA:

```text
74D selected ETA         0.05
74D coverage             0.55
74D mid RMSE             0.00028948
74D max outside          0.00062161
```

The 74D market quotes themselves did not show obvious static-arbitrage failure:

```text
same-expiry monotonicity: no major issue
same-expiry convexity:   no major issue
neighbor calendar:       no major ATM issue
```

The better interpretation is:

```text
The 74D problem is a conflict between sparse effective quotes, joint calendar
constraints, and the current model grid/smoothing choice. It is not enough
evidence to call the market quote itself wrong.
```

## Engineering Additions Beyond The Paper

The paper provides the mathematical construction and discusses production model
strikes, ETA smoothness, bid/ask weighting, and outside bid/ask penalties.

This project adds:

```text
1. Normalized pure-price implementation.
2. OTM call-equivalent quote construction.
3. Single-expiry then joint-LP diagnostic workflow.
4. Coverage/outside-error/calendar/q-roughness diagnostic suite.
5. Generic expiry-case forensic diagnostics, using 74D as the default example.
6. Multi-day ETA and anchor regime study.
7. Automatic regime-based dense-anchor remediation.
8. Wing-weighted QP stabilization prototype.
```

The wing-weighted QP stabilization is SANOS-compatible engineering
regularization, not a formula copied directly from the paper.

## Practical Rule

Use this as the production default:

```text
Start with market-only joint LP.
If an expiry has low coverage or high outside error, and especially if it is
sparse or medium/long dated, remediate only that expiry.
For remediation, use dense_full_tail anchors and ETA candidates in [0.01, 0.10].
Accept a remediation only if target coverage improves, global median coverage
does not materially deteriorate, and calendar checks remain clean.
```

