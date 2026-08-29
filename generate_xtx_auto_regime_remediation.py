from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "option_surface_step1.ipynb"
OUT = ROOT / "xtx_diagnostics" / "auto_regime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatic regime-based XTX/SANOS joint LP remediation.")
    parser.add_argument("--trade-date", default=None, help="Optional YYYY-MM-DD override for the notebook TRADE_DATE.")
    parser.add_argument("--coverage-gate", type=float, default=0.80)
    parser.add_argument("--min-coverage-gain", type=float, default=0.05)
    parser.add_argument("--max-median-coverage-drop", type=float, default=0.02)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--dense-points", type=int, default=201)
    return parser.parse_args()


def execute_state(trade_date: str | None, stop_cell: int = 57) -> dict[str, Any]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    env: dict[str, Any] = {"__name__": "__xtx_auto_regime__", "__file__": str(NOTEBOOK)}
    for idx, cell in enumerate(nb["cells"]):
        if idx > stop_cell:
            break
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if ".show()" in src:
            continue
        if trade_date and idx == 2:
            src = re.sub(
                r'TRADE_DATE\s*=\s*pd\.Timestamp\("[0-9-]+"\)',
                f'TRADE_DATE = pd.Timestamp("{trade_date}")',
                src,
            )
        print(f"execute cell {idx}")
        exec(compile(src, f"{NOTEBOOK.name}:cell{idx}", "exec"), env)
    return env


def dense_full_tail_anchors(market_k: np.ndarray, n_dense: int) -> np.ndarray:
    market_k = np.unique(np.asarray(market_k, dtype=float))
    left = max(float(market_k.min()) * 0.96, 1e-6)
    right = float(market_k.max()) * 1.04
    dense = np.linspace(left, right, n_dense)
    return np.unique(np.r_[market_k, dense])


def eta_candidates_for_regime(dte: int, n_quotes: int) -> list[float]:
    if dte <= 7:
        return [0.01, 0.025]
    if dte <= 21:
        return [0.01, 0.025, 0.05]
    if dte <= 45:
        return [0.01, 0.025, 0.05, 0.075]
    if dte <= 90:
        return [0.01, 0.025, 0.05, 0.075, 0.10]
    if n_quotes <= 25:
        return [0.01, 0.025, 0.05, 0.075, 0.10, 0.15]
    return [0.01, 0.025, 0.05, 0.075, 0.10]


def clone_inputs(joint_inputs: dict[pd.Timestamp, dict[str, Any]]) -> dict[pd.Timestamp, dict[str, Any]]:
    return {pd.Timestamp(k): copy.copy(v) for k, v in joint_inputs.items()}


def patch_one_expiry(
    env: dict[str, Any],
    joint_inputs: dict[pd.Timestamp, dict[str, Any]],
    target_exdate: pd.Timestamp,
    eta: float,
    dense_points: int,
) -> dict[pd.Timestamp, dict[str, Any]]:
    build_kernel_matrix = env["build_kernel_matrix"]
    out = clone_inputs(joint_inputs)
    item = copy.copy(out[target_exdate])
    anchors = dense_full_tail_anchors(item["market_k"], dense_points)
    item["anchor_k"] = anchors
    item["smooth_variance"] = eta * float(item["V_j"])
    item["G"] = build_kernel_matrix(anchors, item["market_k"], item["smooth_variance"])
    out[target_exdate] = item
    return out


def solve_joint(env: dict[str, Any], joint_inputs: dict[pd.Timestamp, dict[str, Any]]) -> dict[str, Any]:
    env["GLOBAL_CALENDAR_K_MIN"] = min(float(np.min(item["market_k"])) for item in joint_inputs.values())
    env["GLOBAL_CALENDAR_K_MAX"] = max(float(np.max(item["market_k"])) for item in joint_inputs.values())
    return env["solve_joint_expiry_lp"](joint_inputs)


def per_expiry_by_exdate(sol: dict[str, Any]) -> pd.DataFrame:
    per = sol["per_expiry"].copy()
    per["exdate_ts"] = pd.to_datetime(per["exdate"])
    if "q_roughness" not in per.columns and "q_by_expiry" in sol:
        roughness = {
            pd.Timestamp(exdate): float(np.linalg.norm(np.diff(np.asarray(q, dtype=float))))
            for exdate, q in sol["q_by_expiry"].items()
        }
        per["q_roughness"] = per["exdate_ts"].map(roughness)
    return per


def target_row(per: pd.DataFrame, target_exdate: pd.Timestamp) -> pd.Series:
    return per.loc[per["exdate_ts"].eq(target_exdate)].iloc[0]


def find_targets(
    per: pd.DataFrame,
    already_patched: set[pd.Timestamp],
    rejected: set[pd.Timestamp],
    coverage_gate: float,
) -> pd.DataFrame:
    candidates = per[
        (per["bid_ask_coverage"] < coverage_gate)
        & (~per["exdate_ts"].isin(already_patched))
        & (~per["exdate_ts"].isin(rejected))
        & ((per["n_quotes"] <= 40) | (per["dte"] >= 30))
    ].copy()
    if candidates.empty:
        return candidates
    return candidates.sort_values(["bid_ask_coverage", "max_outside_bid_ask", "dte"], ascending=[True, False, False])


def candidate_row(
    sol: dict[str, Any],
    eta: float,
    target_exdate: pd.Timestamp,
    target_before: pd.Series,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "eta": eta,
        "success": bool(sol["success"]),
        "message": sol["message"],
        "target_exdate": str(target_exdate.date()),
        "target_dte": int(target_before["dte"]),
        "target_coverage_before": float(target_before["bid_ask_coverage"]),
    }
    if not sol["success"]:
        return row
    per = per_expiry_by_exdate(sol)
    target_after = target_row(per, target_exdate)
    calendar_min = float(sol["calendar_checks"]["min_calendar_diff"].min())
    row.update(
        {
            "objective": float(sol["objective"]),
            "median_coverage": float(per["bid_ask_coverage"].median()),
            "worst_coverage": float(per["bid_ask_coverage"].min()),
            "target_coverage": float(target_after["bid_ask_coverage"]),
            "target_coverage_gain": float(target_after["bid_ask_coverage"] - target_before["bid_ask_coverage"]),
            "target_mid_rmse": float(target_after["mid_rmse"]),
            "target_max_outside": float(target_after["max_outside_bid_ask"]),
            "target_weighted_outside": float(target_after["weighted_outside_error"]),
            "target_n_anchor": int(target_after["n_anchor"]),
            "target_nonzero_q": int(target_after["nonzero_q_count"]),
            "target_q_roughness": float(target_after["q_roughness"]),
            "calendar_min": calendar_min,
            "n_vars": int(sol["n_vars"]),
            "n_ub": int(sol["n_ub"]),
        }
    )
    return row


def choose_candidate(
    metrics: pd.DataFrame,
    current_median_coverage: float,
    args: argparse.Namespace,
) -> pd.Series | None:
    ok = metrics[metrics["success"].astype(bool)].copy()
    if ok.empty:
        return None
    ok = ok[
        (ok["target_coverage_gain"] >= args.min_coverage_gain)
        & (ok["median_coverage"] >= current_median_coverage - args.max_median_coverage_drop)
        & (ok["calendar_min"] >= -1e-5)
    ].copy()
    if ok.empty:
        return None

    max_cov = float(ok["target_coverage"].max())
    near_best = ok[ok["target_coverage"] >= max_cov - 1e-12].copy()
    loss_floor = float(near_best["target_weighted_outside"].min())
    near_best = near_best[near_best["target_weighted_outside"] <= loss_floor * 1.10 + 1e-12]
    return near_best.sort_values(
        ["median_coverage", "target_q_roughness", "target_mid_rmse", "eta"],
        ascending=[False, True, True, True],
    ).iloc[0]


def plot_before_after(baseline_per: pd.DataFrame, final_per: pd.DataFrame, steps: pd.DataFrame, args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True)
    axes[0].plot(baseline_per["dte"], baseline_per["bid_ask_coverage"], marker="o", label="baseline", linewidth=1.8)
    axes[0].plot(final_per["dte"], final_per["bid_ask_coverage"], marker="o", label="auto regime", linewidth=1.8)
    axes[0].axhline(args.coverage_gate, color="gray", linestyle="--", linewidth=1, label="coverage gate")
    axes[0].set_ylabel("bid/ask coverage")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    axes[1].plot(baseline_per["dte"], baseline_per["max_outside_bid_ask"], marker="o", label="baseline", linewidth=1.8)
    axes[1].plot(final_per["dte"], final_per["max_outside_bid_ask"], marker="o", label="auto regime", linewidth=1.8)
    axes[1].set_ylabel("max outside bid/ask")
    axes[1].set_xlabel("DTE")
    axes[1].legend()

    for _, step in steps[steps["accepted"].astype(bool)].iterrows():
        for ax in axes:
            ax.axvline(float(step["target_dte"]), color="#6a5acd", alpha=0.18, linewidth=2)

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Automatic Regime-Based Remediation")
    fig.tight_layout()
    fig.savefig(OUT / "auto_regime_before_after.png")
    plt.close(fig)


def save_summary(
    baseline_per: pd.DataFrame,
    final_sol: dict[str, Any],
    steps: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    final_per = per_expiry_by_exdate(final_sol)
    final_per.to_csv(OUT / "auto_regime_final_per_expiry.csv", index=False)
    steps.to_csv(OUT / "auto_regime_steps.csv", index=False)
    candidate_metrics.to_csv(OUT / "auto_regime_candidate_metrics.csv", index=False)
    plot_before_after(baseline_per, final_per, steps, args)

    base_median = float(baseline_per["bid_ask_coverage"].median())
    final_median = float(final_per["bid_ask_coverage"].median())
    base_worst = baseline_per.sort_values("bid_ask_coverage").head(8)
    final_worst = final_per.sort_values("bid_ask_coverage").head(8)
    accepted = steps[steps["accepted"].astype(bool)].copy()

    lines = [
        "# Automatic Regime-Based Remediation",
        "",
        f"- Coverage gate: {args.coverage_gate:.2f}",
        f"- Max rounds: {args.max_rounds}",
        f"- Accepted remediations: {len(accepted)}",
        f"- Baseline median coverage: {base_median:.6g}",
        f"- Final median coverage: {final_median:.6g}",
        f"- Baseline worst coverage: {float(baseline_per['bid_ask_coverage'].min()):.6g}",
        f"- Final worst coverage: {float(final_per['bid_ask_coverage'].min()):.6g}",
        f"- Final calendar min: {float(final_sol['calendar_checks']['min_calendar_diff'].min()):.6g}",
        "",
        "## Accepted Steps",
        "",
        accepted.to_markdown(index=False) if not accepted.empty else "No accepted remediation steps.",
        "",
        "## Baseline Worst Expiries",
        "",
        base_worst[
            ["exdate", "dte", "n_quotes", "n_anchor", "bid_ask_coverage", "mid_rmse", "max_outside_bid_ask", "q_roughness"]
        ].to_markdown(index=False),
        "",
        "## Final Worst Expiries",
        "",
        final_worst[
            ["exdate", "dte", "n_quotes", "n_anchor", "bid_ask_coverage", "mid_rmse", "max_outside_bid_ask", "q_roughness"]
        ].to_markdown(index=False),
    ]
    (OUT / "auto_regime_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    env = execute_state(args.trade_date)
    current_inputs = clone_inputs(env["joint_inputs"])
    current_sol = env["joint_solution"]
    baseline_per = per_expiry_by_exdate(current_sol)
    current_median = float(baseline_per["bid_ask_coverage"].median())
    patched: set[pd.Timestamp] = set()
    rejected: set[pd.Timestamp] = set()
    steps = []
    all_candidate_rows = []

    for round_idx in range(1, args.max_rounds + 1):
        current_per = per_expiry_by_exdate(current_sol)
        targets = find_targets(current_per, patched, rejected, args.coverage_gate)
        if targets.empty:
            break

        target = targets.iloc[0]
        target_exdate = pd.Timestamp(target["exdate_ts"])
        target_dte = int(target["dte"])
        target_quotes = int(target["n_quotes"])
        etas = eta_candidates_for_regime(target_dte, target_quotes)
        print(f"round {round_idx}: target {target_exdate.date()} {target_dte}D, etas={etas}")

        rows = []
        solutions: dict[float, tuple[dict[str, Any], dict[pd.Timestamp, dict[str, Any]]]] = {}
        for eta in etas:
            print(f"  solve eta={eta:g}")
            patched_inputs = patch_one_expiry(env, current_inputs, target_exdate, eta, args.dense_points)
            sol = solve_joint(env, patched_inputs)
            row = candidate_row(sol, eta, target_exdate, target)
            row["round"] = round_idx
            rows.append(row)
            all_candidate_rows.append(row)
            if sol["success"]:
                solutions[eta] = (sol, patched_inputs)

        metrics = pd.DataFrame(rows)
        best = choose_candidate(metrics, current_median, args)
        if best is None:
            rejected.add(target_exdate)
            steps.append(
                {
                    "round": round_idx,
                    "accepted": False,
                    "target_exdate": str(target_exdate.date()),
                    "target_dte": target_dte,
                    "reason": "no candidate passed acceptance gates",
                }
            )
            continue

        best_eta = float(best["eta"])
        current_sol, current_inputs = solutions[best_eta]
        current_median = float(per_expiry_by_exdate(current_sol)["bid_ask_coverage"].median())
        patched.add(target_exdate)
        step = best.to_dict()
        step["round"] = round_idx
        step["accepted"] = True
        step["reason"] = "accepted"
        steps.append(step)

    steps_df = pd.DataFrame(steps)
    candidate_df = pd.DataFrame(all_candidate_rows)
    save_summary(baseline_per, current_sol, steps_df, candidate_df, args)
    print(f"Wrote automatic regime remediation diagnostics to {OUT}")


if __name__ == "__main__":
    main()
