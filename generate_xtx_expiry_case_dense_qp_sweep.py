from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import clarabel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse as sp


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "option_surface_step1.ipynb"
OUT = ROOT / "xtx_diagnostics"
DEFAULT_DTE = 74


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an expiry-level SANOS/XTX case diagnostic.")
    parser.add_argument("--target-dte", type=int, default=DEFAULT_DTE, help="Nearest DTE to analyze. Defaults to 74 for the project case study.")
    return parser.parse_args()
ETAS = [0.01, 0.05]
LAM_SMOOTHS = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
LAM_RIDGE = 1e-8
MID_TIE_BREAK_EPS = 1e-8


def execute_state(stop_cell: int = 44) -> dict[str, Any]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    env: dict[str, Any] = {"__name__": "__xtx_dense_qp_sweep__", "__file__": str(NOTEBOOK)}
    for idx, cell in enumerate(nb["cells"]):
        if idx > stop_cell:
            break
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if ".show()" in src:
            continue
        print(f"execute cell {idx}")
        exec(compile(src, f"{NOTEBOOK.name}:cell{idx}", "exec"), env)
    return env


def dense_full_tail_anchors(market_k: np.ndarray) -> np.ndarray:
    market_k = np.unique(np.asarray(market_k, dtype=float))
    dense = np.linspace(max(float(market_k.min()) * 0.96, 1e-6), float(market_k.max()) * 1.04, 201)
    return np.unique(np.r_[market_k, dense])


def first_difference_operator(n: int) -> sp.csr_matrix:
    if n < 2:
        return sp.csr_matrix((0, n))
    rows = np.repeat(np.arange(n - 1), 2)
    cols = np.column_stack([np.arange(n - 1), np.arange(1, n)]).ravel()
    data = np.column_stack([-np.ones(n - 1), np.ones(n - 1)]).ravel()
    return sp.coo_matrix((data, (rows, cols)), shape=(n - 1, n)).tocsr()


def solve_single_qp(item: dict[str, Any], lam_smooth: float, lam_ridge: float) -> dict[str, Any]:
    G = sp.csr_matrix(item["G"])
    anchor_k = np.asarray(item["anchor_k"], dtype=float)
    bid = np.asarray(item["pure_bid"], dtype=float)
    mid = np.asarray(item["pure_mid"], dtype=float)
    ask = np.asarray(item["pure_ask"], dtype=float)
    n_quotes, n_anchor = G.shape

    q0 = 0
    u0 = n_anchor
    v0 = u0 + n_quotes
    a0 = v0 + n_quotes
    n_vars = a0 + n_quotes

    spread = np.maximum(ask - bid, 1e-10)
    weights = 1.0 / spread
    c = np.zeros(n_vars)
    c[u0:v0] = weights
    c[v0:a0] = weights
    c[a0:n_vars] = MID_TIE_BREAK_EPS * weights

    # Quadratic penalty only on q.
    D = first_difference_operator(n_anchor)
    Pq = (2.0 * lam_smooth * (D.T @ D) + 2.0 * lam_ridge * sp.identity(n_anchor)).tocoo()
    P = sp.coo_matrix(
        (Pq.data, (Pq.row, Pq.col)),
        shape=(n_vars, n_vars),
    ).tocsc()

    eq_rows = []
    eq_cols = []
    eq_data = []
    b_eq = []
    for r, vals, rhs in [
        (0, np.ones(n_anchor), 1.0),
        (1, anchor_k, 1.0),
    ]:
        eq_rows.extend([r] * n_anchor)
        eq_cols.extend(range(n_anchor))
        eq_data.extend(vals)
        b_eq.append(rhs)
    A_eq = sp.coo_matrix((eq_data, (eq_rows, eq_cols)), shape=(2, n_vars)).tocsc()
    b_eq = np.asarray(b_eq)

    rows = []
    rhs = []
    # Gq - u <= ask
    block = sp.hstack([G, -sp.identity(n_quotes), sp.csr_matrix((n_quotes, 2 * n_quotes))], format="csr")
    rows.append(block)
    rhs.append(ask)
    # -Gq - v <= -bid
    block = sp.hstack([ -G, sp.csr_matrix((n_quotes, n_quotes)), -sp.identity(n_quotes), sp.csr_matrix((n_quotes, n_quotes))], format="csr")
    rows.append(block)
    rhs.append(-bid)
    # Gq - a <= mid
    block = sp.hstack([G, sp.csr_matrix((n_quotes, 2 * n_quotes)), -sp.identity(n_quotes)], format="csr")
    rows.append(block)
    rhs.append(mid)
    # -Gq - a <= -mid
    block = sp.hstack([-G, sp.csr_matrix((n_quotes, 2 * n_quotes)), -sp.identity(n_quotes)], format="csr")
    rows.append(block)
    rhs.append(-mid)
    A_ub = sp.vstack(rows, format="csc")
    b_ub = np.concatenate(rhs)

    A = sp.vstack([A_eq, A_ub, -sp.identity(n_vars, format="csc")], format="csc")
    b = np.concatenate([b_eq, b_ub, np.zeros(n_vars)])
    cones = [clarabel.ZeroConeT(A_eq.shape[0]), clarabel.NonnegativeConeT(A_ub.shape[0] + n_vars)]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 200
    solver = clarabel.DefaultSolver(sp.triu(P, format="csc"), c, A, b, cones, settings)
    res = solver.solve()
    success = str(res.status) in {"Solved", "AlmostSolved"}
    out = {"success": success, "message": str(res.status), "objective": float(res.obj_val)}
    if not success:
        return out
    x = np.maximum(np.asarray(res.x), 0.0)
    q = x[q0:u0]
    fit = item["G"] @ q
    outside = np.maximum(bid - fit, 0.0) + np.maximum(fit - ask, 0.0)
    out.update(
        {
            "q": q,
            "fit": fit,
            "coverage": float(np.mean((fit >= bid - 1e-9) & (fit <= ask + 1e-9))),
            "mid_rmse": float(np.sqrt(np.mean((fit - mid) ** 2))),
            "max_outside_bid_ask": float(outside.max()),
            "mean_outside_bid_ask": float(outside.mean()),
            "weighted_outside_error": float(np.sum(weights * outside)),
            "nonzero_q_count": int((q > 1e-6).sum()),
            "q_roughness": float(np.linalg.norm(np.diff(q))),
        }
    )
    return out


def main() -> None:
    args = parse_args()
    target_dte = args.target_dte
    OUT.mkdir(exist_ok=True)
    env = execute_state()
    sanos_expiry_inputs = env["sanos_expiry_inputs"]
    build_kernel_matrix = env["build_kernel_matrix"]

    target_exdate = min(
        (pd.Timestamp(x) for x in sanos_expiry_inputs),
        key=lambda x: abs(int(sanos_expiry_inputs[x]["dte"]) - target_dte),
    )
    base_item = sanos_expiry_inputs[target_exdate]
    anchors = dense_full_tail_anchors(base_item["market_k"])

    rows = []
    for eta in ETAS:
        item = dict(base_item)
        item["anchor_k"] = anchors
        item["smooth_variance"] = eta * float(base_item["V_j"])
        item["G"] = build_kernel_matrix(anchors, base_item["market_k"], item["smooth_variance"])
        for lam_smooth in LAM_SMOOTHS:
            print(f"solve eta={eta:g} lam_smooth={lam_smooth:g}")
            sol = solve_single_qp(item, lam_smooth=lam_smooth, lam_ridge=LAM_RIDGE)
            row = {
                "eta": eta,
                "lam_smooth": lam_smooth,
                "lam_ridge": LAM_RIDGE,
                "n_anchor": len(anchors),
                "success": sol["success"],
                "message": sol["message"],
            }
            for key in [
                "coverage",
                "mid_rmse",
                "max_outside_bid_ask",
                "mean_outside_bid_ask",
                "weighted_outside_error",
                "nonzero_q_count",
                "q_roughness",
                "objective",
            ]:
                if key in sol:
                    row[key] = sol[key]
            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "expiry_case_dense_qp_sweep_metrics.csv", index=False)
    ok = result[result["success"]].copy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150, sharex=True)
    for eta, group in ok.groupby("eta"):
        group = group.sort_values("lam_smooth")
        x = group["lam_smooth"].replace(0.0, 1e-10)
        axes[0, 0].semilogx(x, group["coverage"], marker="o", linewidth=1.8, label=f"eta={eta:g}")
        axes[0, 1].semilogx(x, group["mid_rmse"], marker="o", linewidth=1.8, label=f"eta={eta:g}")
        axes[1, 0].semilogx(x, group["max_outside_bid_ask"], marker="o", linewidth=1.8, label=f"eta={eta:g}")
        axes[1, 1].semilogx(x, group["q_roughness"], marker="o", linewidth=1.8, label=f"eta={eta:g}")
    axes[0, 0].set_ylabel("coverage")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 1].set_ylabel("mid RMSE")
    axes[1, 0].set_ylabel("max outside")
    axes[1, 0].set_xlabel("lambda smooth")
    axes[1, 1].set_ylabel("q roughness")
    axes[1, 1].set_xlabel("lambda smooth")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Expiry Case Dense-Full-Tail Single-Expiry QP/Ridge Sweep")
    fig.tight_layout()
    fig.savefig(OUT / "20_expiry_case_dense_qp_sweep_metrics.png")
    plt.close(fig)

    lines = [
        "# Expiry Case Dense Anchor QP/Ridge Sweep",
        "",
        f"- Target expiry: {target_exdate.date()}",
        f"- DTE: {base_item['dte']}",
        f"- Anchor grid: dense_full_tail",
        f"- Ridge lambda: {LAM_RIDGE:g}",
        "",
        ok[
            [
                "eta",
                "lam_smooth",
                "coverage",
                "mid_rmse",
                "max_outside_bid_ask",
                "weighted_outside_error",
                "nonzero_q_count",
                "q_roughness",
            ]
        ].to_markdown(index=False),
    ]
    (OUT / "expiry_case_dense_qp_sweep_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote Expiry Case dense QP diagnostics to {OUT}")


if __name__ == "__main__":
    main()
