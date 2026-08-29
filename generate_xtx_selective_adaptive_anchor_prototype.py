from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "option_surface_step1.ipynb"
OUT = ROOT / "xtx_diagnostics"


def execute_state(stop_cell: int = 57) -> dict[str, Any]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    env: dict[str, Any] = {"__name__": "__xtx_selective_adaptive__", "__file__": str(NOTEBOOK)}
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


def dense_full_tail_anchors(market_k: np.ndarray, n_dense: int = 201) -> np.ndarray:
    market_k = np.unique(np.asarray(market_k, dtype=float))
    left = max(float(market_k.min()) * 0.96, 1e-6)
    right = float(market_k.max()) * 1.04
    dense = np.linspace(left, right, n_dense)
    return np.unique(np.r_[market_k, dense])


def should_adapt(item: dict[str, Any]) -> bool:
    dte = int(item["dte"])
    n_quotes = len(item["market_k"])
    return 30 <= dte <= 90 and n_quotes <= 40


def build_selective_inputs(env: dict[str, Any], eta_for_adapted: float) -> tuple[dict[pd.Timestamp, dict[str, Any]], list[int]]:
    build_kernel_matrix = env["build_kernel_matrix"]
    base_joint_inputs = env["joint_inputs"]
    out: dict[pd.Timestamp, dict[str, Any]] = {}
    adapted_dtes = []
    for exdate in sorted(pd.Timestamp(x) for x in base_joint_inputs):
        base = base_joint_inputs[exdate]
        item = copy.copy(base)
        if should_adapt(item):
            anchors = dense_full_tail_anchors(item["market_k"])
            smooth_variance = eta_for_adapted * float(item["V_j"])
            item["anchor_k"] = anchors
            item["smooth_variance"] = smooth_variance
            item["G"] = build_kernel_matrix(anchors, item["market_k"], smooth_variance)
            adapted_dtes.append(int(item["dte"]))
        out[exdate] = item
    return out, adapted_dtes


def summarize(sol: dict[str, Any], joint_inputs: dict[pd.Timestamp, dict[str, Any]], eta: float, adapted_dtes: list[int]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "eta_for_adapted": eta,
        "adapted_dtes": ",".join(str(x) for x in adapted_dtes),
        "success": sol["success"],
        "message": sol["message"],
        "n_vars": sol["n_vars"],
        "n_ub": sol["n_ub"],
        "n_anchor_total": int(sum(len(item["anchor_k"]) for item in joint_inputs.values())),
    }
    if not sol["success"]:
        return row
    per = sol["per_expiry"].copy()
    target = per.loc[per["dte"].eq(74)]
    rough = [float(np.linalg.norm(np.diff(q))) for q in sol["q_by_expiry"].values()]
    row.update(
        {
            "objective": sol["objective"],
            "median_coverage": float(per["bid_ask_coverage"].median()),
            "mean_coverage": float(per["bid_ask_coverage"].mean()),
            "worst_coverage": float(per["bid_ask_coverage"].min()),
            "worst_dte": int(per.loc[per["bid_ask_coverage"].idxmin(), "dte"]),
            "calendar_min": float(sol["calendar_checks"]["min_calendar_diff"].min()),
            "coverage_74d": float(target["bid_ask_coverage"].iloc[0]) if not target.empty else np.nan,
            "mid_rmse_74d": float(target["mid_rmse"].iloc[0]) if not target.empty else np.nan,
            "max_outside_74d": float(target["max_outside_bid_ask"].iloc[0]) if not target.empty else np.nan,
            "median_q_roughness": float(np.median(rough)),
            "max_q_roughness": float(np.max(rough)),
        }
    )
    return row


def main() -> None:
    OUT.mkdir(exist_ok=True)
    env = execute_state()
    solve_joint_expiry_lp = env["solve_joint_expiry_lp"]
    baseline = env["joint_solution"]["per_expiry"].copy()
    eta_candidates = [0.01, 0.025, 0.05, 0.075]

    rows = []
    solutions: dict[float, dict[str, Any]] = {}
    for eta in eta_candidates:
        print(f"solve selective adaptive eta={eta:g}")
        joint_inputs, adapted_dtes = build_selective_inputs(env, eta)
        env["GLOBAL_CALENDAR_K_MIN"] = min(float(np.min(item["market_k"])) for item in joint_inputs.values())
        env["GLOBAL_CALENDAR_K_MAX"] = max(float(np.max(item["market_k"])) for item in joint_inputs.values())
        sol = solve_joint_expiry_lp(joint_inputs)
        rows.append(summarize(sol, joint_inputs, eta, adapted_dtes))
        if sol["success"]:
            solutions[eta] = sol

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "selective_adaptive_anchor_joint_lp_metrics.csv", index=False)
    if not solutions:
        print(f"Wrote failed selective adaptive metrics to {OUT}")
        return

    ok = metrics[metrics["success"]].copy().sort_values("eta_for_adapted")
    best = ok.sort_values(["worst_coverage", "median_coverage", "calendar_min"], ascending=[False, False, False]).iloc[0]
    best_eta = float(best["eta_for_adapted"])
    best_per = solutions[best_eta]["per_expiry"].copy()
    best_per.to_csv(OUT / "selective_adaptive_anchor_joint_lp_best_per_expiry.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150, sharex=True)
    axes[0, 0].plot(ok["eta_for_adapted"], ok["median_coverage"], marker="o", label="selective median", linewidth=1.8)
    axes[0, 0].plot(ok["eta_for_adapted"], ok["coverage_74d"], marker="o", label="selective 74D", linewidth=1.8)
    axes[0, 0].axhline(float(baseline["bid_ask_coverage"].median()), color="gray", linestyle="--", label="baseline median")
    axes[0, 0].axhline(float(baseline.loc[baseline["dte"].eq(74), "bid_ask_coverage"].iloc[0]), color="#b24c00", linestyle="--", label="baseline 74D")
    axes[0, 0].set_ylabel("coverage")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(ok["eta_for_adapted"], ok["max_outside_74d"], marker="o", linewidth=1.8)
    axes[0, 1].axhline(float(baseline.loc[baseline["dte"].eq(74), "max_outside_bid_ask"].iloc[0]), color="gray", linestyle="--")
    axes[0, 1].set_ylabel("74D max outside")

    axes[1, 0].plot(ok["eta_for_adapted"], ok["calendar_min"], marker="o", linewidth=1.8)
    axes[1, 0].axhline(0.0, color="gray", linestyle="--")
    axes[1, 0].set_ylabel("calendar min")
    axes[1, 0].set_xlabel("ETA for adapted expiries")

    axes[1, 1].plot(ok["eta_for_adapted"], ok["median_q_roughness"], marker="o", linewidth=1.8)
    axes[1, 1].set_ylabel("median q roughness")
    axes[1, 1].set_xlabel("ETA for adapted expiries")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    fig.suptitle("Selective Adaptive Anchor Joint LP Prototype")
    fig.tight_layout()
    fig.savefig(OUT / "21_selective_adaptive_anchor_joint_lp_metrics.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True)
    axes[0].plot(baseline["dte"], baseline["bid_ask_coverage"], marker="o", label="baseline", linewidth=1.8)
    axes[0].plot(best_per["dte"], best_per["bid_ask_coverage"], marker="o", label=f"selective eta={best_eta:g}", linewidth=1.8)
    axes[0].set_ylabel("bid/ask coverage")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[1].plot(baseline["dte"], baseline["mid_rmse"], marker="o", label="baseline", linewidth=1.8)
    axes[1].plot(best_per["dte"], best_per["mid_rmse"], marker="o", label=f"selective eta={best_eta:g}", linewidth=1.8)
    axes[1].set_ylabel("mid RMSE")
    axes[1].set_xlabel("DTE")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Baseline vs Best Selective Adaptive Anchor Joint LP")
    fig.tight_layout()
    fig.savefig(OUT / "22_baseline_vs_selective_adaptive_best_per_expiry.png")
    plt.close(fig)

    lines = [
        "# Selective Adaptive Anchor Joint LP Prototype",
        "",
        "Rule: adapt only expiries with `30 <= dte <= 90` and `n_quotes <= 40`.",
        "",
        "## Candidate Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Selected Candidate",
        "",
        best.to_frame().T.to_markdown(index=False),
        "",
        "## Worst Expiries Under Selected Candidate",
        "",
        best_per.sort_values("bid_ask_coverage")
        .head(8)[["exdate", "dte", "n_quotes", "n_anchor", "bid_ask_coverage", "mid_rmse", "max_outside_bid_ask", "nonzero_q_count"]]
        .to_markdown(index=False),
    ]
    (OUT / "selective_adaptive_anchor_joint_lp_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote selective adaptive prototype diagnostics to {OUT}")


if __name__ == "__main__":
    main()
