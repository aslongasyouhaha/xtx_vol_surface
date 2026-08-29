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
ETA_CANDIDATES = [0.01, 0.025, 0.05, 0.075]


def execute_state(stop_cell: int = 57) -> dict[str, Any]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    env: dict[str, Any] = {"__name__": "__xtx_adaptive_anchor__", "__file__": str(NOTEBOOK)}
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


def dense_n_for_dte(dte: int) -> int:
    if dte <= 30:
        return 121
    if dte <= 90:
        return 201
    return 241


def adaptive_anchor_grid(market_k: np.ndarray, dte: int) -> np.ndarray:
    market_k = np.unique(np.asarray(market_k, dtype=float))
    market_k = market_k[np.isfinite(market_k)]
    if len(market_k) == 0:
        return market_k

    left = max(float(market_k.min()) * 0.96, 1e-6)
    right = float(market_k.max()) * 1.04
    full = np.linspace(left, right, dense_n_for_dte(int(dte)))

    # ATM support is included for every expiry; clipping keeps anchors local to
    # the observed market domain plus mild tails.
    atm = np.arange(0.85, 1.10 + 1e-12, 0.002)
    atm = atm[(atm >= left) & (atm <= right)]
    return np.unique(np.r_[market_k, full, atm])


def build_adaptive_joint_inputs(env: dict[str, Any], eta: float) -> dict[pd.Timestamp, dict[str, Any]]:
    sanos_expiry_inputs = env["sanos_expiry_inputs"]
    build_kernel_matrix = env["build_kernel_matrix"]
    out: dict[pd.Timestamp, dict[str, Any]] = {}
    for exdate in sorted(pd.Timestamp(x) for x in sanos_expiry_inputs):
        base = sanos_expiry_inputs[exdate]
        item = copy.copy(base)
        anchors = adaptive_anchor_grid(base["market_k"], base["dte"])
        smooth_variance = float(eta) * float(base["V_j"])
        item["anchor_k"] = anchors
        item["smooth_variance"] = smooth_variance
        item["G"] = build_kernel_matrix(anchors, base["market_k"], smooth_variance)
        out[exdate] = item
    return out


def summarize_solution(sol: dict[str, Any], joint_inputs: dict[pd.Timestamp, dict[str, Any]], eta: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "eta": eta,
        "success": sol["success"],
        "message": sol["message"],
        "n_vars": sol["n_vars"],
        "n_ub": sol["n_ub"],
        "n_anchor_total": int(sum(len(item["anchor_k"]) for item in joint_inputs.values())),
        "n_anchor_median": float(np.median([len(item["anchor_k"]) for item in joint_inputs.values()])),
    }
    if not sol["success"]:
        return row

    per = sol["per_expiry"].copy()
    target = per.loc[per["dte"].eq(74)]
    row.update(
        {
            "objective": sol["objective"],
            "median_coverage": float(per["bid_ask_coverage"].median()),
            "mean_coverage": float(per["bid_ask_coverage"].mean()),
            "worst_coverage": float(per["bid_ask_coverage"].min()),
            "worst_dte": int(per.loc[per["bid_ask_coverage"].idxmin(), "dte"]),
            "max_outside": float(per["max_outside_bid_ask"].max()),
            "calendar_min": float(sol["calendar_checks"]["min_calendar_diff"].min()),
            "coverage_74d": float(target["bid_ask_coverage"].iloc[0]) if not target.empty else np.nan,
            "mid_rmse_74d": float(target["mid_rmse"].iloc[0]) if not target.empty else np.nan,
            "max_outside_74d": float(target["max_outside_bid_ask"].iloc[0]) if not target.empty else np.nan,
            "nonzero_q_74d": int(target["nonzero_q_count"].iloc[0]) if not target.empty else -1,
        }
    )
    rough = []
    for exdate, q in sol["q_by_expiry"].items():
        rough.append(float(np.linalg.norm(np.diff(q))))
    row["median_q_roughness"] = float(np.median(rough))
    row["max_q_roughness"] = float(np.max(rough))
    return row


def save_plots(metrics: pd.DataFrame, baseline: pd.DataFrame) -> None:
    ok = metrics[metrics["success"]].copy().sort_values("eta")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150, sharex=True)

    axes[0, 0].plot(ok["eta"], ok["median_coverage"], marker="o", label="adaptive median", linewidth=1.8)
    axes[0, 0].plot(ok["eta"], ok["coverage_74d"], marker="o", label="adaptive 74D", linewidth=1.8)
    axes[0, 0].axhline(float(baseline["bid_ask_coverage"].median()), color="gray", linestyle="--", label="baseline median")
    axes[0, 0].axhline(float(baseline.loc[baseline["dte"].eq(74), "bid_ask_coverage"].iloc[0]), color="#b24c00", linestyle="--", label="baseline 74D")
    axes[0, 0].set_ylabel("coverage")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(ok["eta"], ok["max_outside_74d"], marker="o", linewidth=1.8)
    axes[0, 1].axhline(float(baseline.loc[baseline["dte"].eq(74), "max_outside_bid_ask"].iloc[0]), color="gray", linestyle="--")
    axes[0, 1].set_ylabel("74D max outside")

    axes[1, 0].plot(ok["eta"], ok["calendar_min"], marker="o", linewidth=1.8)
    axes[1, 0].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1, 0].set_ylabel("calendar min")
    axes[1, 0].set_xlabel("ETA")

    axes[1, 1].plot(ok["eta"], ok["median_q_roughness"], marker="o", linewidth=1.8)
    axes[1, 1].set_ylabel("median q roughness")
    axes[1, 1].set_xlabel("ETA")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    fig.suptitle("Adaptive Anchor Joint LP Prototype")
    fig.tight_layout()
    fig.savefig(OUT / "21_adaptive_anchor_joint_lp_metrics.png")
    plt.close(fig)


def save_best_per_expiry(best_sol: dict[str, Any], baseline: pd.DataFrame, eta: float) -> None:
    per = best_sol["per_expiry"].copy()
    per.to_csv(OUT / "adaptive_anchor_joint_lp_best_per_expiry.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True)
    axes[0].plot(baseline["dte"], baseline["bid_ask_coverage"], marker="o", label="baseline", linewidth=1.8)
    axes[0].plot(per["dte"], per["bid_ask_coverage"], marker="o", label=f"adaptive eta={eta:g}", linewidth=1.8)
    axes[0].set_ylabel("bid/ask coverage")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[1].plot(baseline["dte"], baseline["mid_rmse"], marker="o", label="baseline", linewidth=1.8)
    axes[1].plot(per["dte"], per["mid_rmse"], marker="o", label=f"adaptive eta={eta:g}", linewidth=1.8)
    axes[1].set_ylabel("mid RMSE")
    axes[1].set_xlabel("DTE")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Baseline vs Best Adaptive Anchor Joint LP")
    fig.tight_layout()
    fig.savefig(OUT / "22_baseline_vs_adaptive_best_per_expiry.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    env = execute_state()
    solve_joint_expiry_lp = env["solve_joint_expiry_lp"]
    baseline = env["joint_solution"]["per_expiry"].copy()
    baseline.to_csv(OUT / "baseline_joint_lp_per_expiry_from_prototype.csv", index=False)

    rows = []
    solutions: dict[float, tuple[dict[str, Any], dict[pd.Timestamp, dict[str, Any]]]] = {}
    for eta in ETA_CANDIDATES:
        print(f"solve adaptive joint eta={eta:g}")
        joint_inputs = build_adaptive_joint_inputs(env, eta)
        env["GLOBAL_CALENDAR_K_MIN"] = min(float(np.min(item["market_k"])) for item in joint_inputs.values())
        env["GLOBAL_CALENDAR_K_MAX"] = max(float(np.max(item["market_k"])) for item in joint_inputs.values())
        sol = solve_joint_expiry_lp(joint_inputs)
        rows.append(summarize_solution(sol, joint_inputs, eta))
        if sol["success"]:
            solutions[eta] = (sol, joint_inputs)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "adaptive_anchor_joint_lp_metrics.csv", index=False)
    save_plots(metrics, baseline)

    ok = metrics[metrics["success"]].copy()
    best = ok.sort_values(["worst_coverage", "median_coverage", "calendar_min"], ascending=[False, False, False]).iloc[0]
    best_eta = float(best["eta"])
    best_sol = solutions[best_eta][0]
    save_best_per_expiry(best_sol, baseline, best_eta)

    lines = [
        "# Adaptive Anchor Joint LP Prototype",
        "",
        "## Candidate Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Selected Candidate",
        "",
        best.to_frame().T.to_markdown(index=False),
        "",
        "## Baseline Headline",
        "",
        f"- Baseline median coverage: {float(baseline['bid_ask_coverage'].median()):.3f}",
        f"- Baseline worst coverage: {float(baseline['bid_ask_coverage'].min()):.3f}",
        f"- Baseline 74D coverage: {float(baseline.loc[baseline['dte'].eq(74), 'bid_ask_coverage'].iloc[0]):.3f}",
    ]
    (OUT / "adaptive_anchor_joint_lp_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote adaptive prototype diagnostics to {OUT}")


if __name__ == "__main__":
    main()
