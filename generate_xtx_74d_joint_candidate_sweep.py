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
TARGET_DTE = 74


def execute_state(stop_cell: int = 56) -> dict[str, Any]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    env: dict[str, Any] = {"__name__": "__xtx_joint_candidate_sweep__", "__file__": str(NOTEBOOK)}
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


def dense_full_anchors(market_k: np.ndarray) -> np.ndarray:
    market_k = np.unique(np.asarray(market_k, dtype=float))
    dense = np.linspace(float(market_k.min()), float(market_k.max()), 161)
    return np.unique(np.r_[market_k, dense])


def dense_atm_anchors(market_k: np.ndarray) -> np.ndarray:
    market_k = np.unique(np.asarray(market_k, dtype=float))
    atm = np.arange(0.900, 1.051 + 1e-12, 0.002)
    return np.unique(np.r_[market_k, atm])


def patch_74d_item(env: dict[str, Any], base_joint_inputs: dict[pd.Timestamp, dict[str, Any]], grid: str, eta: float | None) -> dict[pd.Timestamp, dict[str, Any]]:
    build_kernel_matrix = env["build_kernel_matrix"]
    out = {pd.Timestamp(k): copy.copy(v) for k, v in base_joint_inputs.items()}
    target_exdate = min(out, key=lambda x: abs(int(out[x]["dte"]) - TARGET_DTE))
    item = copy.copy(out[target_exdate])

    if grid == "baseline":
        anchors = item["anchor_k"]
    elif grid == "dense_full_tail":
        anchors = dense_full_tail_anchors(item["market_k"])
    elif grid == "dense_full":
        anchors = dense_full_anchors(item["market_k"])
    elif grid == "dense_atm_0_90_1_05":
        anchors = dense_atm_anchors(item["market_k"])
    else:
        raise ValueError(grid)

    smooth_variance = item["smooth_variance"] if eta is None else float(eta) * float(item["V_j"])
    item["anchor_k"] = anchors
    item["smooth_variance"] = smooth_variance
    item["G"] = build_kernel_matrix(anchors, item["market_k"], smooth_variance)
    out[target_exdate] = item
    return out


def model_metrics(item: dict[str, Any], fit: np.ndarray) -> dict[str, float]:
    bid = item["pure_bid"]
    ask = item["pure_ask"]
    mid = item["pure_mid"]
    err = np.maximum(bid - fit, 0.0) + np.maximum(fit - ask, 0.0)
    return {
        "coverage": float(np.mean((fit >= bid - 1e-9) & (fit <= ask + 1e-9))),
        "mid_rmse": float(np.sqrt(np.mean((fit - mid) ** 2))),
        "max_outside_bid_ask": float(err.max()),
        "mean_outside_bid_ask": float(err.mean()),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    env = execute_state()
    solve_joint_expiry_lp = env["solve_joint_expiry_lp"]
    base_joint_inputs = env["joint_inputs"]
    target_exdate = min(base_joint_inputs, key=lambda x: abs(int(base_joint_inputs[x]["dte"]) - TARGET_DTE))

    configs = [
        {"name": "baseline_eta_0_25", "grid": "baseline", "eta": None},
        {"name": "dense_full_tail_eta_0_01", "grid": "dense_full_tail", "eta": 0.01},
        {"name": "dense_full_tail_eta_0_05", "grid": "dense_full_tail", "eta": 0.05},
        {"name": "dense_full_eta_0_05", "grid": "dense_full", "eta": 0.05},
        {"name": "dense_atm_eta_0_01", "grid": "dense_atm_0_90_1_05", "eta": 0.01},
    ]

    rows = []
    fit_store: dict[str, np.ndarray] = {}
    q_store: dict[str, np.ndarray] = {}
    input_store: dict[str, dict[str, Any]] = {}

    for cfg in configs:
        print(f"solve joint {cfg['name']}")
        joint_inputs = patch_74d_item(env, base_joint_inputs, cfg["grid"], cfg["eta"])
        sol = solve_joint_expiry_lp(joint_inputs)
        item = joint_inputs[target_exdate]
        row = {
            "config": cfg["name"],
            "grid": cfg["grid"],
            "eta": cfg["eta"] if cfg["eta"] is not None else 0.25,
            "success": sol["success"],
            "message": sol["message"],
            "n_vars": sol["n_vars"],
            "n_ub": sol["n_ub"],
            "n_anchor_74d": len(item["anchor_k"]),
            "smooth_variance_74d": item["smooth_variance"],
        }
        if sol["success"]:
            per = sol["per_expiry"].copy()
            target_row = per.loc[per["exdate"].eq(target_exdate)].iloc[0]
            fit = sol["fit_by_expiry"][target_exdate]
            q = sol["q_by_expiry"][target_exdate]
            row.update(
                {
                    "objective": sol["objective"],
                    "median_coverage_all": float(per["bid_ask_coverage"].median()),
                    "worst_coverage_all": float(per["bid_ask_coverage"].min()),
                    "coverage_74d": float(target_row["bid_ask_coverage"]),
                    "mid_rmse_74d": float(target_row["mid_rmse"]),
                    "max_outside_74d": float(target_row["max_outside_bid_ask"]),
                    "weighted_outside_74d": float(target_row["weighted_outside_error"]),
                    "nonzero_q_74d": int(target_row["nonzero_q_count"]),
                    "q_roughness_74d": float(np.linalg.norm(np.diff(q))),
                    "calendar_min": float(sol["calendar_checks"]["min_calendar_diff"].min()),
                    **{f"direct_{k}": v for k, v in model_metrics(item, fit).items()},
                }
            )
            fit_store[cfg["name"]] = fit
            q_store[cfg["name"]] = q
            input_store[cfg["name"]] = item
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "74d_joint_candidate_sweep_metrics.csv", index=False)

    ok = result[result["success"]].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    x = np.arange(len(ok))
    labels = ok["config"].to_list()
    axes[0, 0].bar(x, ok["coverage_74d"])
    axes[0, 0].set_ylabel("74D coverage")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 1].bar(x, ok["mid_rmse_74d"], color="#b24c00")
    axes[0, 1].set_ylabel("74D mid RMSE")
    axes[1, 0].bar(x, ok["max_outside_74d"], color="#7a3db8")
    axes[1, 0].set_ylabel("74D max outside")
    axes[1, 1].bar(x, ok["q_roughness_74d"], color="#2f7f73")
    axes[1, 1].set_ylabel("74D q roughness")
    for ax in axes.ravel():
        ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Joint LP Candidate Sweep: 74D Patch Only")
    fig.tight_layout()
    fig.savefig(OUT / "18_74d_joint_candidate_sweep_metrics.png")
    plt.close(fig)

    base_item = input_store.get("baseline_eta_0_25") or base_joint_inputs[target_exdate]
    qframe = base_item["quotes"].copy().reset_index(drop=True)
    qframe["_order"] = np.arange(len(qframe))
    qsorted = qframe.sort_values("k")
    order = qsorted["_order"].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.fill_between(qsorted["k"], qsorted["pure_bid"], qsorted["pure_ask"], color="#b9d7f0", alpha=0.45, label="74D bid/ask")
    ax.scatter(qsorted["k"], qsorted["pure_mid"], s=18, color="#1f77b4", label="74D market mid", zorder=3)
    for name in labels:
        fit = fit_store[name][order]
        cov = float(ok.loc[ok["config"].eq(name), "coverage_74d"].iloc[0])
        ax.plot(qsorted["k"], fit, linewidth=1.7, label=f"{name}, cov={cov:.2f}")
    ax.set_title("74D Joint LP Fits Under Candidate Patches")
    ax.set_xlabel("normalized strike k")
    ax.set_ylabel("pure call price")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "19_74d_joint_candidate_fit_curves.png")
    plt.close(fig)

    lines = [
        "# 74D Joint Candidate Sweep",
        "",
        f"- Target expiry: {target_exdate.date()}",
        f"- DTE: {base_joint_inputs[target_exdate]['dte']}",
        "",
        ok[
            [
                "config",
                "grid",
                "eta",
                "n_anchor_74d",
                "coverage_74d",
                "mid_rmse_74d",
                "max_outside_74d",
                "weighted_outside_74d",
                "median_coverage_all",
                "worst_coverage_all",
                "calendar_min",
                "nonzero_q_74d",
                "q_roughness_74d",
            ]
        ].to_markdown(index=False),
    ]
    (OUT / "74d_joint_candidate_sweep_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote 74D joint candidate sweep diagnostics to {OUT}")


if __name__ == "__main__":
    main()
