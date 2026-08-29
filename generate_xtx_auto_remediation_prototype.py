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
COVERAGE_GATE = 0.60
ETA_REMEDIATION_CANDIDATES = [0.01, 0.025, 0.05, 0.075]


def execute_state(stop_cell: int = 57) -> dict[str, Any]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    env: dict[str, Any] = {"__name__": "__xtx_auto_remediation__", "__file__": str(NOTEBOOK)}
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


def find_remediation_target(per_expiry: pd.DataFrame) -> pd.Series | None:
    candidates = per_expiry[
        (per_expiry["bid_ask_coverage"] < COVERAGE_GATE)
        & (per_expiry["dte"].between(30, 90))
        & (per_expiry["n_quotes"] <= 40)
    ].copy()
    if candidates.empty:
        return None
    return candidates.sort_values(["bid_ask_coverage", "max_outside_bid_ask"], ascending=[True, False]).iloc[0]


def patch_one_expiry(
    env: dict[str, Any],
    base_joint_inputs: dict[pd.Timestamp, dict[str, Any]],
    target_exdate: pd.Timestamp,
    eta: float,
) -> dict[pd.Timestamp, dict[str, Any]]:
    build_kernel_matrix = env["build_kernel_matrix"]
    out = {pd.Timestamp(k): copy.copy(v) for k, v in base_joint_inputs.items()}
    item = copy.copy(out[target_exdate])
    anchors = dense_full_tail_anchors(item["market_k"])
    item["anchor_k"] = anchors
    item["smooth_variance"] = eta * float(item["V_j"])
    item["G"] = build_kernel_matrix(anchors, item["market_k"], item["smooth_variance"])
    out[target_exdate] = item
    return out


def summary_row(sol: dict[str, Any], eta: float, target_dte: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "eta": eta,
        "success": sol["success"],
        "message": sol["message"],
        "n_vars": sol["n_vars"],
        "n_ub": sol["n_ub"],
    }
    if not sol["success"]:
        return row
    per = sol["per_expiry"]
    target = per.loc[per["dte"].eq(target_dte)].iloc[0]
    row.update(
        {
            "objective": sol["objective"],
            "median_coverage": float(per["bid_ask_coverage"].median()),
            "worst_coverage": float(per["bid_ask_coverage"].min()),
            "worst_dte": int(per.loc[per["bid_ask_coverage"].idxmin(), "dte"]),
            "target_coverage": float(target["bid_ask_coverage"]),
            "target_mid_rmse": float(target["mid_rmse"]),
            "target_max_outside": float(target["max_outside_bid_ask"]),
            "target_weighted_outside": float(target["weighted_outside_error"]),
            "target_n_anchor": int(target["n_anchor"]),
            "target_nonzero_q": int(target["nonzero_q_count"]),
            "calendar_min": float(sol["calendar_checks"]["min_calendar_diff"].min()),
        }
    )
    return row


def main() -> None:
    OUT.mkdir(exist_ok=True)
    env = execute_state()
    solve_joint_expiry_lp = env["solve_joint_expiry_lp"]
    baseline_sol = env["joint_solution"]
    baseline_per = baseline_sol["per_expiry"].copy()
    target = find_remediation_target(baseline_per)

    if target is None:
        (OUT / "auto_remediation_summary.md").write_text(
            "No remediation target found under the current gates.\n",
            encoding="utf-8",
        )
        print("No remediation target found")
        return

    target_dte = int(target["dte"])
    target_exdate = pd.Timestamp(target["exdate"])
    base_joint_inputs = env["joint_inputs"]

    rows = []
    solutions: dict[float, dict[str, Any]] = {}
    for eta in ETA_REMEDIATION_CANDIDATES:
        print(f"solve auto remediation target={target_dte}D eta={eta:g}")
        joint_inputs = patch_one_expiry(env, base_joint_inputs, target_exdate, eta)
        env["GLOBAL_CALENDAR_K_MIN"] = min(float(np.min(item["market_k"])) for item in joint_inputs.values())
        env["GLOBAL_CALENDAR_K_MAX"] = max(float(np.max(item["market_k"])) for item in joint_inputs.values())
        sol = solve_joint_expiry_lp(joint_inputs)
        rows.append(summary_row(sol, eta, target_dte))
        if sol["success"]:
            solutions[eta] = sol

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "auto_remediation_metrics.csv", index=False)
    ok = metrics[metrics["success"]].copy()
    if ok.empty:
        (OUT / "auto_remediation_summary.md").write_text(
            "# Auto Remediation Prototype\n\nAll remediation candidates failed.\n\n"
            + metrics.to_markdown(index=False),
            encoding="utf-8",
        )
        print("All remediation candidates failed")
        return

    best = ok.sort_values(["target_coverage", "median_coverage", "calendar_min"], ascending=[False, False, False]).iloc[0]
    best_eta = float(best["eta"])
    best_per = solutions[best_eta]["per_expiry"].copy()
    best_per.to_csv(OUT / "auto_remediation_best_per_expiry.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True)
    axes[0].plot(baseline_per["dte"], baseline_per["bid_ask_coverage"], marker="o", label="baseline", linewidth=1.8)
    axes[0].plot(best_per["dte"], best_per["bid_ask_coverage"], marker="o", label=f"auto remediation eta={best_eta:g}", linewidth=1.8)
    axes[0].axhline(COVERAGE_GATE, color="gray", linestyle="--", linewidth=1, label="coverage gate")
    axes[0].set_ylabel("bid/ask coverage")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    axes[1].plot(baseline_per["dte"], baseline_per["max_outside_bid_ask"], marker="o", label="baseline", linewidth=1.8)
    axes[1].plot(best_per["dte"], best_per["max_outside_bid_ask"], marker="o", label=f"auto remediation eta={best_eta:g}", linewidth=1.8)
    axes[1].set_ylabel("max outside bid/ask")
    axes[1].set_xlabel("DTE")
    axes[1].legend()

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Automatic Single-Expiry Remediation Prototype")
    fig.tight_layout()
    fig.savefig(OUT / "23_auto_remediation_best_per_expiry.png")
    plt.close(fig)

    lines = [
        "# Auto Remediation Prototype",
        "",
        f"- Coverage gate: {COVERAGE_GATE:.2f}",
        f"- Automatically selected target: {target_dte}D, {target_exdate.date()}",
        f"- Baseline target coverage: {float(target['bid_ask_coverage']):.3f}",
        f"- Selected remediation ETA: {best_eta:g}",
        "",
        "## Candidate Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Selected Candidate",
        "",
        best.to_frame().T.to_markdown(index=False),
        "",
        "## Worst Expiries After Remediation",
        "",
        best_per.sort_values("bid_ask_coverage")
        .head(8)[["exdate", "dte", "n_quotes", "n_anchor", "bid_ask_coverage", "mid_rmse", "max_outside_bid_ask", "nonzero_q_count"]]
        .to_markdown(index=False),
    ]
    (OUT / "auto_remediation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote auto remediation diagnostics to {OUT}")


if __name__ == "__main__":
    main()
