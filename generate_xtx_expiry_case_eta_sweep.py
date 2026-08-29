from __future__ import annotations

import argparse
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
DEFAULT_DTE = 74


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an expiry-level SANOS/XTX case diagnostic.")
    parser.add_argument("--target-dte", type=int, default=DEFAULT_DTE, help="Nearest DTE to analyze. Defaults to 74 for the project case study.")
    return parser.parse_args()
ETAS = [0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00]


def execute_state(stop_cell: int = 44) -> dict[str, Any]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    env: dict[str, Any] = {"__name__": "__xtx_eta_sweep__", "__file__": str(NOTEBOOK)}
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


def outside_errors(bid: np.ndarray, ask: np.ndarray, fit: np.ndarray) -> np.ndarray:
    return np.maximum(bid - fit, 0.0) + np.maximum(fit - ask, 0.0)


def main() -> None:
    args = parse_args()
    target_dte = args.target_dte
    OUT.mkdir(exist_ok=True)
    env = execute_state()
    sanos_expiry_inputs = env["sanos_expiry_inputs"]
    build_kernel_matrix = env["build_kernel_matrix"]
    solve_single_expiry_lp = env["solve_single_expiry_lp"]

    target_exdate = min(
        (pd.Timestamp(x) for x in sanos_expiry_inputs),
        key=lambda x: abs(int(sanos_expiry_inputs[x]["dte"]) - target_dte),
    )
    base_item = sanos_expiry_inputs[target_exdate]
    quotes = base_item["quotes"].copy().sort_values("k").reset_index(drop=True)

    rows = []
    fits: dict[float, np.ndarray] = {}
    qs: dict[float, np.ndarray] = {}
    for eta in ETAS:
        item = copy.copy(base_item)
        smooth_variance = float(eta) * float(base_item["V_j"])
        item["smooth_variance"] = smooth_variance
        item["G"] = build_kernel_matrix(
            anchor_k=base_item["anchor_k"],
            market_k=base_item["market_k"],
            total_variance=smooth_variance,
        )
        sol = solve_single_expiry_lp(item)
        if not sol["success"]:
            rows.append(
                {
                    "eta": eta,
                    "success": False,
                    "message": sol["message"],
                    "smooth_variance": smooth_variance,
                }
            )
            continue

        fit = sol["fit"]
        err = outside_errors(base_item["pure_bid"], base_item["pure_ask"], fit)
        fits[eta] = fit
        qs[eta] = sol["q"]
        rows.append(
            {
                "eta": eta,
                "success": True,
                "message": sol["message"],
                "smooth_variance": smooth_variance,
                "coverage": sol["bid_ask_coverage"],
                "mid_rmse": sol["mid_rmse"],
                "max_outside_bid_ask": float(err.max()),
                "mean_outside_bid_ask": float(err.mean()),
                "weighted_outside_error": sol["weighted_outside_error"],
                "nonzero_q_count": int((sol["q"] > 1e-6).sum()),
                "q_roughness": float(np.linalg.norm(np.diff(sol["q"]))),
            }
        )

    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "expiry_case_eta_sweep_metrics.csv", index=False)

    ok = sweep[sweep["success"]].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150, sharex=True)
    axes[0, 0].plot(ok["eta"], ok["coverage"], marker="o", linewidth=1.8)
    axes[0, 0].axvline(0.25, color="gray", linestyle="--", linewidth=1, label="current")
    axes[0, 0].set_ylabel("bid/ask coverage")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].legend()

    axes[0, 1].plot(ok["eta"], ok["mid_rmse"], marker="o", color="#b24c00", linewidth=1.8)
    axes[0, 1].axvline(0.25, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].set_ylabel("mid RMSE")

    axes[1, 0].plot(ok["eta"], ok["max_outside_bid_ask"], marker="o", color="#7a3db8", linewidth=1.8)
    axes[1, 0].axvline(0.25, color="gray", linestyle="--", linewidth=1)
    axes[1, 0].set_ylabel("max outside bid/ask")
    axes[1, 0].set_xlabel("ETA")

    axes[1, 1].plot(ok["eta"], ok["q_roughness"], marker="o", color="#2f7f73", linewidth=1.8)
    axes[1, 1].axvline(0.25, color="gray", linestyle="--", linewidth=1)
    axes[1, 1].set_ylabel("q roughness")
    axes[1, 1].set_xlabel("ETA")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    fig.suptitle("Expiry Case Single-Expiry ETA Sweep")
    fig.tight_layout()
    fig.savefig(OUT / "14_expiry_case_eta_sweep_metrics.png")
    plt.close(fig)

    selected = [eta for eta in [0.0, 0.025, 0.05, 0.10, 0.25, 0.50, 1.00] if eta in fits]
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    q = base_item["quotes"].copy().reset_index(drop=True)
    q["fit_current_order_marker"] = np.arange(len(q))
    q_sorted = q.sort_values("k")
    order = q_sorted["fit_current_order_marker"].to_numpy(dtype=int)
    ax.fill_between(q_sorted["k"], q_sorted["pure_bid"], q_sorted["pure_ask"], color="#b9d7f0", alpha=0.45, label="market bid/ask")
    ax.scatter(q_sorted["k"], q_sorted["pure_mid"], s=18, color="#1f77b4", label="market mid", zorder=3)
    for eta in selected:
        ax.plot(q_sorted["k"], fits[eta][order], linewidth=1.7, label=f"eta={eta:g}")
    ax.set_title("Expiry Case Pure Call Fit Under Different ETA")
    ax.set_xlabel("normalized strike k")
    ax.set_ylabel("pure call price")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "15_expiry_case_eta_sweep_fit_curves.png")
    plt.close(fig)

    best_cov = ok.sort_values(["coverage", "mid_rmse"], ascending=[False, True]).iloc[0]
    best_rmse = ok.sort_values("mid_rmse").iloc[0]
    lines = [
        "# Expiry Case ETA Sweep",
        "",
        f"- Target expiry: {target_exdate.date()}",
        f"- DTE: {base_item['dte']}",
        f"- Current ETA: 0.25",
        f"- Current coverage: {float(ok.loc[ok['eta'].eq(0.25), 'coverage'].iloc[0]):.3f}",
        f"- Best coverage ETA: {best_cov['eta']:.3g}, coverage={best_cov['coverage']:.3f}, mid_rmse={best_cov['mid_rmse']:.6g}",
        f"- Best RMSE ETA: {best_rmse['eta']:.3g}, coverage={best_rmse['coverage']:.3f}, mid_rmse={best_rmse['mid_rmse']:.6g}",
        "",
        ok[
            [
                "eta",
                "coverage",
                "mid_rmse",
                "max_outside_bid_ask",
                "mean_outside_bid_ask",
                "weighted_outside_error",
                "nonzero_q_count",
                "q_roughness",
            ]
        ].to_markdown(index=False),
    ]
    (OUT / "expiry_case_eta_sweep_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote Expiry Case ETA sweep diagnostics to {OUT}")


if __name__ == "__main__":
    main()
