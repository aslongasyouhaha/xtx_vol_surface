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
ETAS = [0.0, 0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.25]


def execute_state(stop_cell: int = 44) -> dict[str, Any]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    env: dict[str, Any] = {"__name__": "__xtx_anchor_eta_sweep__", "__file__": str(NOTEBOOK)}
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


def anchor_grids(market_k: np.ndarray) -> dict[str, np.ndarray]:
    market_k = np.unique(np.asarray(market_k, dtype=float))
    local_right = np.arange(0.990, 1.036 + 1e-12, 0.001)
    atm = np.arange(0.900, 1.051 + 1e-12, 0.002)
    full = np.linspace(float(market_k.min()), float(market_k.max()), 161)
    full_plus_tail = np.linspace(max(float(market_k.min()) * 0.96, 1e-6), float(market_k.max()) * 1.04, 201)
    return {
        "market_only": market_k,
        "dense_1_00_1_03": np.unique(np.r_[market_k, local_right]),
        "dense_atm_0_90_1_05": np.unique(np.r_[market_k, atm]),
        "dense_full": np.unique(np.r_[market_k, full]),
        "dense_full_tail": np.unique(np.r_[market_k, full_plus_tail]),
    }


def outside_errors(bid: np.ndarray, ask: np.ndarray, fit: np.ndarray) -> np.ndarray:
    return np.maximum(bid - fit, 0.0) + np.maximum(fit - ask, 0.0)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    env = execute_state()
    sanos_expiry_inputs = env["sanos_expiry_inputs"]
    build_kernel_matrix = env["build_kernel_matrix"]
    solve_single_expiry_lp = env["solve_single_expiry_lp"]

    target_exdate = min(
        (pd.Timestamp(x) for x in sanos_expiry_inputs),
        key=lambda x: abs(int(sanos_expiry_inputs[x]["dte"]) - TARGET_DTE),
    )
    base_item = sanos_expiry_inputs[target_exdate]
    grids = anchor_grids(base_item["market_k"])

    rows = []
    fit_store: dict[tuple[str, float], np.ndarray] = {}
    q_store: dict[tuple[str, float], np.ndarray] = {}
    for grid_name, anchors in grids.items():
        for eta in ETAS:
            item = copy.copy(base_item)
            smooth_variance = float(eta) * float(base_item["V_j"])
            item["anchor_k"] = anchors
            item["smooth_variance"] = smooth_variance
            item["G"] = build_kernel_matrix(anchors, base_item["market_k"], smooth_variance)
            sol = solve_single_expiry_lp(item)
            row = {
                "grid": grid_name,
                "eta": eta,
                "n_anchor": len(anchors),
                "smooth_variance": smooth_variance,
                "success": sol["success"],
                "message": sol["message"],
            }
            if sol["success"]:
                err = outside_errors(base_item["pure_bid"], base_item["pure_ask"], sol["fit"])
                row.update(
                    {
                        "coverage": sol["bid_ask_coverage"],
                        "mid_rmse": sol["mid_rmse"],
                        "max_outside_bid_ask": float(err.max()),
                        "mean_outside_bid_ask": float(err.mean()),
                        "weighted_outside_error": sol["weighted_outside_error"],
                        "nonzero_q_count": int((sol["q"] > 1e-6).sum()),
                        "q_roughness": float(np.linalg.norm(np.diff(sol["q"]))),
                    }
                )
                fit_store[(grid_name, eta)] = sol["fit"]
                q_store[(grid_name, eta)] = sol["q"]
            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "74d_anchor_eta_sweep_metrics.csv", index=False)
    ok = result[result["success"]].copy()
    best = ok.sort_values(["coverage", "mid_rmse"], ascending=[False, True]).head(12)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150, sharex=True)
    for grid_name, group in ok.groupby("grid", sort=False):
        group = group.sort_values("eta")
        axes[0, 0].plot(group["eta"], group["coverage"], marker="o", linewidth=1.6, label=grid_name)
        axes[0, 1].plot(group["eta"], group["mid_rmse"], marker="o", linewidth=1.6, label=grid_name)
        axes[1, 0].plot(group["eta"], group["max_outside_bid_ask"], marker="o", linewidth=1.6, label=grid_name)
        axes[1, 1].plot(group["eta"], group["q_roughness"], marker="o", linewidth=1.6, label=grid_name)

    axes[0, 0].set_ylabel("bid/ask coverage")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 1].set_ylabel("mid RMSE")
    axes[1, 0].set_ylabel("max outside bid/ask")
    axes[1, 0].set_xlabel("ETA")
    axes[1, 1].set_ylabel("q roughness")
    axes[1, 1].set_xlabel("ETA")
    axes[0, 0].legend(fontsize=8)
    for ax in axes.ravel():
        ax.axvspan(0.01, 0.05, color="gray", alpha=0.08)
        ax.grid(True, alpha=0.25)
    fig.suptitle("74D Single-Expiry Anchor Grid + ETA Sweep")
    fig.tight_layout()
    fig.savefig(OUT / "16_74d_anchor_eta_sweep_metrics.png")
    plt.close(fig)

    quote_order = np.arange(len(base_item["market_k"]))
    qframe = base_item["quotes"].copy().reset_index(drop=True)
    qframe["_order"] = quote_order
    qsorted = qframe.sort_values("k")
    order = qsorted["_order"].to_numpy(dtype=int)

    selected_rows = best.head(6)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.fill_between(qsorted["k"], qsorted["pure_bid"], qsorted["pure_ask"], color="#b9d7f0", alpha=0.45, label="market bid/ask")
    ax.scatter(qsorted["k"], qsorted["pure_mid"], s=18, color="#1f77b4", label="market mid", zorder=3)
    for _, row in selected_rows.iterrows():
        key = (row["grid"], float(row["eta"]))
        fit = fit_store[key][order]
        label = f"{row['grid']}, eta={row['eta']:g}, cov={row['coverage']:.2f}"
        ax.plot(qsorted["k"], fit, linewidth=1.6, label=label)
    ax.set_title("74D Best Single-Expiry Fits Across Anchor/ETA Sweep")
    ax.set_xlabel("normalized strike k")
    ax.set_ylabel("pure call price")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "17_74d_anchor_eta_best_fit_curves.png")
    plt.close(fig)

    lines = [
        "# 74D Anchor + ETA Sweep",
        "",
        f"- Target expiry: {target_exdate.date()}",
        f"- DTE: {base_item['dte']}",
        f"- Baseline grid: market_only",
        f"- Current baseline: market_only, ETA=0.25",
        "",
        "## Best Configurations",
        "",
        best[
            [
                "grid",
                "eta",
                "n_anchor",
                "coverage",
                "mid_rmse",
                "max_outside_bid_ask",
                "mean_outside_bid_ask",
                "weighted_outside_error",
                "nonzero_q_count",
                "q_roughness",
            ]
        ].to_markdown(index=False),
        "",
        "## ETA 0.01-0.05 Window",
        "",
        ok[ok["eta"].between(0.01, 0.05)][
            [
                "grid",
                "eta",
                "n_anchor",
                "coverage",
                "mid_rmse",
                "max_outside_bid_ask",
                "weighted_outside_error",
                "nonzero_q_count",
                "q_roughness",
            ]
        ].sort_values(["coverage", "mid_rmse"], ascending=[False, True]).to_markdown(index=False),
    ]
    (OUT / "74d_anchor_eta_sweep_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote 74D anchor+ETA sweep diagnostics to {OUT}")


if __name__ == "__main__":
    main()
