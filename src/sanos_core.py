"""SANOS baseline: one bid/ask-loss variable and discrete convex order.

Equal mass/mean and convex order of the unsmoothed marginals, combined with
nondecreasing common lognormal variance, imply calendar order at every strike.
Solver tolerances are checked explicitly; numerical output is not exact algebra.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import norm

from vol_surface_utils import DEFAULT_MAX_DTE


def call_kernel(anchors, strikes, variance):
    a = np.asarray(anchors, float)[None, :]
    k = np.asarray(strikes, float)[:, None]
    if variance <= 0:
        return np.maximum(a - k, 0.0)
    s = np.sqrt(variance)
    d1 = (np.log(a / k) + 0.5 * variance) / s
    return a * norm.cdf(d1) - k * norm.cdf(d1 - s)


def spread_weights(bid, ask):
    spread = np.asarray(ask) - np.asarray(bid)
    if np.any(~np.isfinite(spread)) or np.any(spread <= 0):
        raise ValueError("Each quote must have a finite positive bid/ask spread.")
    w = 1.0 / np.maximum(spread, np.quantile(spread, 0.10))
    return np.clip(w / np.median(w), 0.01, 100.0)


def paper_spread_weights(bid, ask):
    spread=np.asarray(ask,float)-np.asarray(bid,float)
    if np.any(~np.isfinite(spread)) or np.any(spread<=0):
        raise ValueError('Each quote must have a finite positive bid/ask spread.')
    return 1.0/spread


def build_model_anchor_grid(market_k,*,max_gap=0.01):
    """Retain eligible pre-volume strikes and fill gaps; no anchor-count cap.

    The paper prescribes a maximum spacing but does not specify its value.
    Shared exterior boundary states are added separately by the LP builder.
    """
    a=np.unique(np.asarray(market_k,float))
    if not np.isfinite(max_gap) or max_gap<=0:raise ValueError('max_gap must be positive and finite')
    if not len(a) or np.any(~np.isfinite(a)) or np.any(a<=0):raise ValueError('Invalid model strikes')
    if len(a)==1:return a
    return np.unique(np.concatenate([np.linspace(lo,hi,int(np.ceil((hi-lo)/max_gap))+1)
                                    for lo,hi in zip(a[:-1],a[1:])]))


def add_common_boundary_anchors(inputs):
    """Copy inputs and add shared tail states; do not invent market quotes.

    This is idempotent when the supplied model endpoints are already shared
    and outside all observed market strikes.
    """
    if not inputs:
        raise ValueError("No eligible expiries.")
    items = {pd.Timestamp(k): dict(v) for k, v in sorted(inputs.items(), key=lambda kv: pd.Timestamp(kv[0]))}
    minimum = min(np.min(v['market_k']) for v in items.values())
    maximum = max(np.max(v['market_k']) for v in items.values())
    lows = [float(np.min(v['anchor_k'])) for v in items.values()]
    highs = [float(np.max(v['anchor_k'])) for v in items.values()]
    shared = (np.ptp(lows) < 1e-14 and np.ptp(highs) < 1e-14
              and lows[0] < minimum and highs[0] > maximum)
    left = lows[0] if shared else min(lows) * 0.95
    right = highs[0] if shared else max(highs) * 1.05
    if not 0 < left < 1 < right:
        raise ValueError("Common boundary anchors must bracket the unit forward.")
    for v in items.values():
        v['anchor_k'] = np.unique(np.r_[left, v['anchor_k'], right])
        v['G'] = call_kernel(v['anchor_k'], v['market_k'], v['smooth_variance'])
    return items


def _latent_otm(anchors, strikes):
    a = np.asarray(anchors)[None, :]
    k = np.asarray(strikes)[:, None]
    # Unit mass/mean make put and call convex-order inequalities equivalent.
    return np.where(k < 1, np.maximum(k - a, 0), np.maximum(a - k, 0))


def solve_convex_order_lp(inputs, *, time_limit=120.0, calendar_points=4001,
                          feasibility_tolerance=2e-7, mid_penalty=1e-8,
                          weight_mode='paper'):
    return _solve_convex_order(inputs, time_limit=time_limit, calendar_points=calendar_points,
                               feasibility_tolerance=feasibility_tolerance,
                               mid_penalty=mid_penalty,weight_mode=weight_mode)


def solve_convex_order_qp(inputs, price_reference, *, stability_strength=0.01,
                          time_limit=120.0, calendar_points=4001,
                          feasibility_tolerance=2e-7):
    """Same feasible set as LP; penalize spread-scaled price changes.

    Reference values are (prices, mask) at the current market strikes, keyed
    by expiry. Only masked observations enter the temporal penalty. The loss is
    weighted outside error / sum(weight * spread), plus strength times the
    mean squared reference change in floored-spread units. Strength zero is
    a Clarabel LP control, useful to distinguish objective and backend effects.
    """
    if not np.isfinite(stability_strength) or stability_strength < 0:
        raise ValueError('stability_strength must be finite and nonnegative.')
    return _solve_convex_order(inputs, time_limit=time_limit, calendar_points=calendar_points,
                               feasibility_tolerance=feasibility_tolerance,
                               price_reference=price_reference, stability_strength=stability_strength)


def build_convex_order_problem(inputs, *, weight_mode='paper',mid_penalty=None,calendar_grid='next'):
    """Build the shared LP once; expose matrices for persistent native solvers.

    Next-expiry knots suffice: its latent call is linear between these knots,
    whereas the previous latent call is convex. Endpoint dominance implies
    dominance throughout each interval. Shared outer states handle the tails.
    Full union knots are still checked independently after every solve.
    """
    if mid_penalty is None:mid_penalty=1e-8 if weight_mode=='paper' else 0.
    if calendar_grid not in {'union','next'}:raise ValueError('Unknown calendar grid')
    if not np.isfinite(mid_penalty) or mid_penalty<0:raise ValueError('mid_penalty must be finite and nonnegative')
    started = time.perf_counter()
    items = add_common_boundary_anchors(inputs)
    kernel_seconds = time.perf_counter()-started
    exdates = list(items)
    variances = np.array([v['smooth_variance'] for v in items.values()])
    if np.any(~np.isfinite(variances)) or np.any(variances < 0) or np.any(np.diff(variances) < -1e-14):
        raise ValueError("Convex-order baseline requires nonnegative, nondecreasing smoothing variances.")
    if weight_mode not in {'paper','legacy'}:raise ValueError('Unknown weight mode')
    weight_fn=paper_spread_weights if weight_mode=='paper' else spread_weights
    blocks, eqblocks, rhs, c = [], [], [], []
    q_offsets, z_offsets = {}, {}
    cursor = 0
    for ex, v in items.items():
        if not 1 <= int(v['dte']) <= DEFAULT_MAX_DTE:
            raise ValueError(f"Expiry {ex} is outside 1-{DEFAULT_MAX_DTE}D.")
        a = np.asarray(v['anchor_k'], float)
        k = np.asarray(v['market_k'], float)
        if not len(k) or np.any(~np.isfinite(k)) or np.any(k <= 0) or np.any(~np.isfinite(a)):
            raise ValueError(f"Invalid strikes for {ex}.")
        n, na = len(k), len(a)
        G = v['G']
        blocks.append(sparse.bmat([[sparse.csr_matrix(G), -sparse.eye(n)],
                                   [-sparse.csr_matrix(G), -sparse.eye(n)]], format='csr'))
        eqblocks.append(sparse.csr_matrix(np.c_[np.vstack([np.ones(na), a]), np.zeros((2, n))]))
        rhs.extend(np.r_[v['pure_ask'], -v['pure_bid']])
        c.extend(np.r_[np.zeros(na), weight_fn(v['pure_bid'], v['pure_ask'])])
        q_offsets[ex] = (cursor, cursor + na)
        z_offsets[ex] = (cursor + na, cursor + na + n)
        cursor += na + n
    rr, cc, dd, row = [], [], [], 0
    for prev, nxt in zip(exdates[:-1], exdates[1:]):
        # C_next is linear between its own states and C_prev is convex.
        # Dominance at NEXT states therefore implies dominance everywhere.
        # Union knots remain the independent audit and comparison formulation.
        knots = (np.union1d(items[prev]['anchor_k'], items[nxt]['anchor_k'])
                 if calendar_grid=='union' else items[nxt]['anchor_k'])
        pa, pb = _latent_otm(items[prev]['anchor_k'], knots), _latent_otm(items[nxt]['anchor_k'], knots)
        active = np.any(pa != 0, axis=1) | np.any(pb != 0, axis=1)
        pa, pb = pa[active], pb[active]
        for ex, values in [(prev, pa), (nxt, -pb)]:
            r, col = np.nonzero(values)
            rr.extend((r + row).tolist()); cc.extend((col + q_offsets[ex][0]).tolist())
            dd.extend(values[r, col].tolist())
        row += len(pa)
    market = sparse.block_diag(blocks, format='csr')
    calendar = sparse.coo_matrix((dd, (rr, cc)), shape=(row, cursor)).tocsr()
    A = sparse.vstack([market, calendar], format='csr')
    b = np.r_[rhs, np.zeros(row)]
    E = sparse.block_diag(eqblocks, format='csr')
    mid_offsets={};mid_rows={};ne=E.shape[0];market_rows=sum(2*len(v['market_k']) for v in items.values())
    calendar_rows=(ne+market_rows,ne+market_rows+row)
    if mid_penalty:
        nmid=sum(len(v['market_k']) for v in items.values())
        kernels=[];targets=[];mid_cost=[];offset=cursor;quote_offset=0
        for ex,v in items.items():
            n=len(v['market_k']);q0,q1=q_offsets[ex]
            kernels.append(sparse.hstack([sparse.csr_matrix((n,q0)),sparse.csr_matrix(v['G']),
                                          sparse.csr_matrix((n,cursor-q1))],format='csr'))
            targets.extend(v['pure_mid']);mid_cost.extend(mid_penalty*weight_fn(v['pure_bid'],v['pure_ask']))
            mid_offsets[ex]=(offset,offset+n);offset+=n
            mid_rows[ex]=(ne+A.shape[0]+quote_offset,ne+A.shape[0]+nmid+quote_offset,n)
            quote_offset+=n
        K=sparse.vstack(kernels,format='csr');I=sparse.eye(nmid)
        A=sparse.vstack([sparse.hstack([A,sparse.csr_matrix((A.shape[0],nmid))]),
                         sparse.hstack([K,-I]),sparse.hstack([-K,-I])],format='csr')
        E=sparse.hstack([E,sparse.csr_matrix((E.shape[0],nmid))],format='csr')
        b=np.r_[b,targets,-np.asarray(targets)];c=np.r_[c,mid_cost];cursor+=nmid
    return dict(items=items,exdates=exdates,A=A,b=b,E=E,c=np.asarray(c),calendar_grid=calendar_grid,
                weight_mode=weight_mode,mid_penalty=mid_penalty,mid_offsets=mid_offsets,
                mid_rows=mid_rows,calendar_rows=calendar_rows,
                q_offsets=q_offsets,z_offsets=z_offsets,cursor=cursor,
                calendar_constraint_count=row,kernel_seconds=kernel_seconds,
                assembly_seconds=time.perf_counter()-started-kernel_seconds)


def _solve_convex_order(inputs, *, time_limit, calendar_points, feasibility_tolerance,
                        price_reference=None, stability_strength=0.0,
                        mid_penalty=0.,weight_mode='legacy'):
    if not np.isfinite(mid_penalty) or mid_penalty<0:raise ValueError('mid_penalty must be finite and nonnegative')
    problem = build_convex_order_problem(inputs,weight_mode=weight_mode,mid_penalty=mid_penalty)
    items,exdates = problem['items'],problem['exdates']
    A,b,E,c = (problem[k] for k in ['A','b','E','c'])
    q_offsets,z_offsets = problem['q_offsets'],problem['z_offsets']
    cursor,row = problem['cursor'],problem['calendar_constraint_count']
    weight_fn=paper_spread_weights if weight_mode=='paper' else spread_weights
    mid_offsets=problem['mid_offsets']
    t = time.perf_counter()
    attempts = []
    if price_reference is not None:
        import clarabel
        from types import SimpleNamespace
        normalizer = sum(spread_weights(v['pure_bid'], v['pure_ask']) @
                         (np.asarray(v['pure_ask'])-np.asarray(v['pure_bid'])) for v in items.values())
        linear = np.asarray(c) / normalizer
        selected = sum(np.count_nonzero(mask) for _, mask in price_reference.values())
        if not selected:
            raise ValueError('No observations selected for the price reference.')
        ref_blocks, ref_targets = [], []
        for ex, v in items.items():
            nq = len(v['anchor_k']); nz = len(v['market_k'])
            B = np.zeros((0,nq))
            if ex in price_reference:
                ref, mask = price_reference[ex]
                ref, mask = np.asarray(ref, float), np.asarray(mask, bool)
                if ref.shape != (nz,) or mask.shape != (nz,) or np.any(~np.isfinite(ref[mask])):
                    raise ValueError(f'Invalid price reference for {ex}.')
                spread = np.asarray(v['pure_ask']) - np.asarray(v['pure_bid'])
                scale = np.maximum(spread, np.quantile(spread, .10))[mask]
                # Put parity on the left avoids cancellation of large intrinsic
                # call values. Unit mass/mean make these exact same price changes.
                strikes = v['market_k'][mask]
                kernel = np.where(strikes[:,None] < 1,
                                  v['G'][mask]-v['anchor_k'][None,:]+strikes[:,None],v['G'][mask])
                B = kernel / scale[:, None]
                target = (ref[mask]-np.maximum(1-strikes,0)) / scale
                ref_targets.extend(target)
            ref_blocks.append(sparse.hstack([sparse.csr_matrix(B),sparse.csr_matrix((len(B),nz))]))
        # Explicit free residuals avoid forming ill-conditioned G'WG or a large
        # constant-subtracted quadratic objective. They do not change feasibility
        # of the original q,z variables.
        nr = selected if stability_strength else 0
        if nr:
            R = sparse.block_diag(ref_blocks,format='csr')
            equalities = sparse.vstack([sparse.hstack([E,sparse.csr_matrix((E.shape[0],nr))]),
                                       sparse.hstack([R,-sparse.eye(nr)])],format='csc')
        else:
            equalities = E.tocsc()
        inequalities = sparse.vstack([A,-sparse.eye(cursor)],format='csc')
        inequalities = sparse.hstack([inequalities,sparse.csr_matrix((inequalities.shape[0],nr))],format='csc')
        P = sparse.diags(np.r_[np.zeros(cursor),np.full(nr,2*stability_strength/selected)],format='csc')
        linear = np.r_[linear,np.zeros(nr)]
        constraints = sparse.vstack([equalities,inequalities],format='csc')
        rhs_all = np.r_[np.ones(E.shape[0]),ref_targets if nr else [], b, np.zeros(cursor)]
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = 200
        settings.time_limit = float(time_limit)
        settings.tol_feas = 1e-8
        settings.tol_gap_abs = 1e-6
        settings.tol_gap_rel = 1e-6
        optimizer = clarabel.DefaultSolver(P, linear, constraints, rhs_all,
            [clarabel.ZeroConeT(equalities.shape[0]), clarabel.NonnegativeConeT(A.shape[0]+cursor)], settings)
        raw = optimizer.solve()
        result = SimpleNamespace(success=str(raw.status) in {'Solved','AlmostSolved'}, message=str(raw.status),
                                 status=str(raw.status), x=np.asarray(raw.x)[:cursor], fun=float(raw.obj_val))
        attempts.append({'method': 'clarabel', 'status': str(raw.status), 'seconds': time.perf_counter()-t,
                         'primal_residual':float(raw.r_prim),'dual_residual':float(raw.r_dual),
                         'objective_gap':float(abs(raw.obj_val-raw.obj_val_dual)),
                         'gap_tolerance':1e-6,'feasibility_tolerance':1e-8})
    for method, presolve in ([] if price_reference is not None else [('highs', True), ('highs-ds', False), ('highs-ipm', True)]):
        attempt_start = time.perf_counter()
        result = linprog(np.asarray(c), A_ub=A, b_ub=b, A_eq=E, b_eq=np.ones(E.shape[0]),
                         bounds=[(0, None)] * cursor, method=method,
                         options={'presolve': presolve, 'time_limit': float(time_limit),
                                  'primal_feasibility_tolerance': 1e-9,
                                  'dual_feasibility_tolerance': 1e-9})
        attempts.append({'method': method, 'presolve': presolve, 'status': int(result.status),
                         'message': result.message, 'seconds': time.perf_counter()-attempt_start})
        if result.success or result.status in {2, 3}:
            break
    sol: dict[str, Any] = {
        'success': bool(result.success), 'message': result.message, 'result': result,
        'formulation': 'discrete_convex_order_paper_lp' if mid_penalty else 'discrete_convex_order_one_slack',
        'mid_tiebreak': bool(mid_penalty), 'mid_penalty':mid_penalty,'weight_mode':weight_mode,
        'mid_offsets':mid_offsets,
        'exdates': exdates, 'q_offsets': q_offsets, 'z_offsets': z_offsets,
        'n_vars': cursor, 'n_eq': E.shape[0], 'n_ub': A.shape[0],
        'A_eq_nnz': E.nnz, 'A_ub_nnz': A.nnz, 'calendar_constraint_count': row,
        'solve_seconds': time.perf_counter() - t, 'inputs': items,
        'solver_attempts': attempts, 'validation_tolerance': feasibility_tolerance,
        'stability_strength': stability_strength,
        'kernel_seconds':problem['kernel_seconds'],'assembly_seconds':problem['assembly_seconds'],
    }
    if not result.success:
        return sol
    x = result.x
    # Preserve the raw optimizer output: clipping weights can break the moments.
    residuals = {'max_eq_residual': float(np.max(abs(E @ x - 1))),
                 'min_ub_slack': float(np.min(b - A @ x)), 'min_variable': float(x.min())}
    q_by, fits, per = {}, {}, []
    for ex, v in items.items():
        q0, q1 = q_offsets[ex]; q = x[q0:q1]; fit = v['G'] @ q
        bid, ask, mid = v['pure_bid'], v['pure_ask'], v['pure_mid']
        outside = np.maximum(bid - fit, 0) + np.maximum(fit - ask, 0)
        q_by[ex], fits[ex] = q, fit
        per.append({'exdate': ex, 'dte': v['dte'], 'n_quotes': len(fit),
                    'n_anchor': len(q), 'sum_q': q.sum(), 'mean_q': q @ v['anchor_k'],
                    'nonzero_q_count': int((q > 1e-10).sum()),
                    'q_roughness': float(np.linalg.norm(np.diff(q))),
                    'max_outside_bid_ask': float(outside.max()),
                    'weighted_outside_error': float(weight_fn(bid, ask) @ outside),
                    'weighted_mid_tiebreak_error': float(mid_penalty*(weight_fn(bid,ask)@abs(fit-mid))),
                    'mid_rmse': float(np.sqrt(np.mean((fit - mid) ** 2))),
                    'bid_ask_coverage': float(np.mean(outside <= 1e-9))})
    checks, convex_checks = [], []
    minimum = min(v['market_k'].min() for v in items.values())
    maximum = max(v['market_k'].max() for v in items.values())
    for prev, nxt in zip(exdates[:-1], exdates[1:]):
        a, bitem = items[prev], items[nxt]
        knots = np.union1d(a['anchor_k'], bitem['anchor_k'])
        latent = (np.maximum(bitem['anchor_k'][None, :] - knots[:, None], 0) @ q_by[nxt]
                  - np.maximum(a['anchor_k'][None, :] - knots[:, None], 0) @ q_by[prev])
        convex_checks.append({'prev_dte': a['dte'], 'next_dte': bitem['dte'],
                              'n_knots': len(knots), 'min_convex_order_diff': float(latent.min())})
        grid = np.unique(np.r_[a['market_k'], bitem['market_k'], np.linspace(minimum, maximum, calendar_points)])
        diff = (call_kernel(bitem['anchor_k'], grid, bitem['smooth_variance']) @ q_by[nxt]
                - call_kernel(a['anchor_k'], grid, a['smooth_variance']) @ q_by[prev])
        checks.append({'prev_exdate': prev, 'next_exdate': nxt, 'prev_dte': a['dte'],
                       'next_dte': bitem['dte'], 'n_check_k': len(grid),
                       'min_calendar_diff': float(diff.min()),
                       'violation_count': int((diff < -1e-7).sum())})
    calendar_df = pd.DataFrame(checks, columns=['prev_exdate','next_exdate','prev_dte','next_dte','n_check_k','min_calendar_diff','violation_count'])
    convex_df = pd.DataFrame(convex_checks, columns=['prev_dte','next_dte','n_knots','min_convex_order_diff'])
    errors = [residuals['max_eq_residual'], -residuals['min_ub_slack'], -residuals['min_variable']]
    if len(checks):
        errors += [-calendar_df.min_calendar_diff.min(), -convex_df.min_convex_order_diff.min()]
    valid = max(errors) <= feasibility_tolerance
    sol.update(success=bool(valid), message=result.message if valid else 'Optimizer returned a solution outside validation tolerance.',
               x=x, objective=float(result.fun), residuals=residuals, validated=bool(valid),
               q_by_expiry=q_by, fit_by_expiry=fits, per_expiry=pd.DataFrame(per),
               calendar_checks=calendar_df, convex_order_checks=convex_df)
    if price_reference is None:
        actual=float(sol['per_expiry'].weighted_outside_error.sum()+sol['per_expiry'].weighted_mid_tiebreak_error.sum())
        sol.update(actual_objective=actual,objective_reconstruction_error=abs(actual-float(result.fun)))
    if price_reference is not None:
        # Near-converged candidates are audited, but never labeled successful.
        sol['success'] = sol['success'] and result.status == 'Solved'
        # Recompute the objective from actual prices, independently of residual variables.
        changes = []
        for ex, (ref, mask) in price_reference.items():
            v = items[ex]; mask = np.asarray(mask, bool)
            spread = np.asarray(v['pure_ask'])-np.asarray(v['pure_bid'])
            scale = np.maximum(spread, np.quantile(spread, .10))
            changes.extend(((fits[ex][mask]-np.asarray(ref)[mask])/scale[mask]).tolist())
        loss = float(sol['per_expiry'].weighted_outside_error.sum()) / normalizer
        sol.update(formulation='discrete_convex_order_price_stability',
                   normalized_outside_loss=loss, reference_change_mse=float(np.mean(np.square(changes))),
                   objective=loss+stability_strength*float(np.mean(np.square(changes))))
    return sol
