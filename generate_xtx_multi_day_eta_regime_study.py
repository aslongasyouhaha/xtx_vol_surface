from __future__ import annotations

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
OUT = ROOT / "xtx_diagnostics" / "multi_day_eta"

TRADE_DATES = [
    "2025-01-02",
    "2025-02-03",
    "2025-03-03",
    "2025-04-01",
    "2025-05-01",
    "2025-06-02",
    "2025-07-01",
    "2025-08-01",
]

ETA_CANDIDATES = [0.0, 0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.25, 0.50]
DTE_BINS = [0, 7, 21, 45, 90, 180, 10000]
DTE_LABELS = ["1-7D", "8-21D", "22-45D", "46-90D", "91-180D", "181D+"]


def execute_state_for_date(trade_date: str, stop_cell: int = 44) -> dict[str, Any]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    env: dict[str, Any] = {"__name__": "__xtx_multi_day_eta__", "__file__": str(NOTEBOOK)}
    for idx, cell in enumerate(nb["cells"]):
        if idx > stop_cell:
            break
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if ".show()" in src:
            continue
        if idx == 2:
            src = re.sub(
                r'TRADE_DATE\s*=\s*pd\.Timestamp\("[0-9-]+"\)',
                f'TRADE_DATE = pd.Timestamp("{trade_date}")',
                src,
            )
        exec(compile(src, f"{NOTEBOOK.name}:cell{idx}", "exec"), env)
    return env


def dense_full_tail_anchors(market_k: np.ndarray, n_dense: int = 201) -> np.ndarray:
    market_k = np.unique(np.asarray(market_k, dtype=float))
    left = max(float(market_k.min()) * 0.96, 1e-6)
    right = float(market_k.max()) * 1.04
    dense = np.linspace(left, right, n_dense)
    return np.unique(np.r_[market_k, dense])


def make_item(
    base_item: dict[str, Any],
    anchors: np.ndarray,
    eta: float,
    build_kernel_matrix,
) -> dict[str, Any]:
    item = copy.copy(base_item)
    item["anchor_k"] = anchors
    item["smooth_variance"] = float(eta) * float(base_item["V_j"])
    item["G"] = build_kernel_matrix(anchors, base_item["market_k"], item["smooth_variance"])
    return item


def outside_errors(bid: np.ndarray, ask: np.ndarray, fit: np.ndarray) -> np.ndarray:
    return np.maximum(bid - fit, 0.0) + np.maximum(fit - ask, 0.0)


def run_single_sweeps_for_date(trade_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"date {trade_date}: build state")
    env = execute_state_for_date(trade_date)
    inputs = env["sanos_expiry_inputs"]
    build_kernel_matrix = env["build_kernel_matrix"]
    solve_single_expiry_lp = env["solve_single_expiry_lp"]

    rows = []
    date_summary = {
        "trade_date": trade_date,
        "n_expiries": len(inputs),
        "n_quotes_total": int(sum(len(item["market_k"]) for item in inputs.values())),
    }

    for exdate in sorted(pd.Timestamp(x) for x in inputs):
        base = inputs[exdate]
        grids = {
            "market_only": np.unique(base["market_k"]),
            "dense_full_tail": dense_full_tail_anchors(base["market_k"]),
        }
        for grid_name, anchors in grids.items():
            for eta in ETA_CANDIDATES:
                item = make_item(base, anchors, eta, build_kernel_matrix)
                sol = solve_single_expiry_lp(item)
                row: dict[str, Any] = {
                    "trade_date": trade_date,
                    "exdate": exdate.date().isoformat(),
                    "dte": int(base["dte"]),
                    "T": float(base["T"]),
                    "n_quotes": len(base["market_k"]),
                    "min_k": float(np.min(base["market_k"])),
                    "max_k": float(np.max(base["market_k"])),
                    "grid": grid_name,
                    "n_anchor": len(anchors),
                    "eta": eta,
                    "success": sol["success"],
                    "message": sol["message"],
                }
                if sol["success"]:
                    err = outside_errors(base["pure_bid"], base["pure_ask"], sol["fit"])
                    spread = np.maximum(base["pure_ask"] - base["pure_bid"], 1e-12)
                    row.update(
                        {
                            "coverage": float(sol["bid_ask_coverage"]),
                            "mid_rmse": float(sol["mid_rmse"]),
                            "max_outside": float(err.max()),
                            "mean_outside": float(err.mean()),
                            "weighted_outside": float(sol["weighted_outside_error"]),
                            "spread_unit_outside_mean": float(np.mean(err / spread)),
                            "spread_unit_outside_max": float(np.max(err / spread)),
                            "nonzero_q": int((sol["q"] > 1e-6).sum()),
                            "q_roughness": float(np.linalg.norm(np.diff(sol["q"]))),
                        }
                    )
                rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame([date_summary])


def choose_winners(sweeps: pd.DataFrame) -> pd.DataFrame:
    ok = sweeps[sweeps["success"]].copy()
    winners = []
    keys = ["trade_date", "exdate", "dte", "n_quotes", "grid"]
    for key, group in ok.groupby(keys, sort=False):
        best_cov = group.sort_values(
            ["coverage", "weighted_outside", "mid_rmse", "q_roughness"],
            ascending=[False, True, True, True],
        ).iloc[0]
        best_loss = group.sort_values(
            ["weighted_outside", "mid_rmse", "q_roughness"],
            ascending=[True, True, True],
        ).iloc[0]
        # A practical choice: among configurations within 10% of the best weighted
        # outside loss, pick the smoothest q. This avoids selecting ETA=0 just
        # because it wins one extra quote by a hair.
        loss_floor = float(best_loss["weighted_outside"])
        near = group[group["weighted_outside"] <= loss_floor * 1.10 + 1e-12]
        practical = near.sort_values(["q_roughness", "mid_rmse", "coverage"], ascending=[True, True, False]).iloc[0]
        for label, row in [("best_coverage", best_cov), ("best_loss", best_loss), ("practical_smooth", practical)]:
            winners.append(
                {
                    "selection": label,
                    **{k: row[k] for k in keys},
                    "eta": row["eta"],
                    "coverage": row["coverage"],
                    "weighted_outside": row["weighted_outside"],
                    "mid_rmse": row["mid_rmse"],
                    "q_roughness": row["q_roughness"],
                    "n_anchor": row["n_anchor"],
                }
            )
    out = pd.DataFrame(winners)
    out["dte_bucket"] = pd.cut(out["dte"], bins=DTE_BINS, labels=DTE_LABELS, right=True, include_lowest=True)
    out["quote_bucket"] = pd.cut(
        out["n_quotes"],
        bins=[0, 25, 40, 80, 10000],
        labels=["<=25q", "26-40q", "41-80q", "81q+"],
        right=True,
        include_lowest=True,
    )
    return out


def save_summary(sweeps: pd.DataFrame, winners: pd.DataFrame, date_summary: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sweeps.to_csv(OUT / "multi_day_eta_sweeps.csv", index=False)
    winners.to_csv(OUT / "multi_day_eta_winners.csv", index=False)
    date_summary.to_csv(OUT / "multi_day_date_summary.csv", index=False)

    practical = winners[winners["selection"].eq("practical_smooth")].copy()
    dense = practical[practical["grid"].eq("dense_full_tail")]
    market = practical[practical["grid"].eq("market_only")]

    def pivot_eta(df: pd.DataFrame, by: str) -> pd.DataFrame:
        return (
            df.pivot_table(index=by, columns="eta", values="exdate", aggfunc="count", fill_value=0)
            .sort_index()
        )

    eta_by_dte_dense = pivot_eta(dense, "dte_bucket")
    eta_by_dte_market = pivot_eta(market, "dte_bucket")
    eta_by_quote_dense = pivot_eta(dense, "quote_bucket")

    improvement = (
        practical.pivot_table(
            index=["trade_date", "exdate", "dte", "n_quotes"],
            columns="grid",
            values=["coverage", "weighted_outside", "mid_rmse", "eta"],
            aggfunc="first",
        )
    )
    improvement.columns = [f"{a}_{b}" for a, b in improvement.columns]
    improvement = improvement.reset_index()
    if {"coverage_dense_full_tail", "coverage_market_only"}.issubset(improvement.columns):
        improvement["coverage_gain_dense_vs_market"] = (
            improvement["coverage_dense_full_tail"] - improvement["coverage_market_only"]
        )
        improvement["loss_ratio_dense_vs_market"] = (
            improvement["weighted_outside_dense_full_tail"] / improvement["weighted_outside_market_only"].replace(0, np.nan)
        )
    improvement.to_csv(OUT / "multi_day_dense_vs_market_improvement.csv", index=False)

    hard = improvement.sort_values("coverage_gain_dense_vs_market", ascending=False).head(20)
    bad_dense = improvement.sort_values("coverage_dense_full_tail").head(20)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    dense.boxplot(column="eta", by="dte_bucket", ax=axes[0, 0])
    axes[0, 0].set_title("Practical ETA by DTE Bucket (Dense Anchors)")
    axes[0, 0].set_xlabel("DTE bucket")
    axes[0, 0].set_ylabel("ETA")

    dense.boxplot(column="eta", by="quote_bucket", ax=axes[0, 1])
    axes[0, 1].set_title("Practical ETA by Quote Count (Dense Anchors)")
    axes[0, 1].set_xlabel("quote bucket")
    axes[0, 1].set_ylabel("ETA")

    if "coverage_gain_dense_vs_market" in improvement:
        axes[1, 0].scatter(improvement["dte"], improvement["coverage_gain_dense_vs_market"], s=18, alpha=0.75)
        axes[1, 0].axhline(0.0, color="gray", linestyle="--")
    axes[1, 0].set_title("Dense Anchor Coverage Gain vs DTE")
    axes[1, 0].set_xlabel("DTE")
    axes[1, 0].set_ylabel("coverage gain")

    if "loss_ratio_dense_vs_market" in improvement:
        ratio = improvement["loss_ratio_dense_vs_market"].replace([np.inf, -np.inf], np.nan)
        axes[1, 1].scatter(improvement["dte"], ratio, s=18, alpha=0.75)
        axes[1, 1].axhline(1.0, color="gray", linestyle="--")
        axes[1, 1].set_ylim(0, np.nanquantile(ratio, 0.95) * 1.1)
    axes[1, 1].set_title("Dense Anchor Loss Ratio vs DTE")
    axes[1, 1].set_xlabel("DTE")
    axes[1, 1].set_ylabel("dense / market weighted outside")

    plt.suptitle("")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "multi_day_eta_regime_charts.png")
    plt.close(fig)

    lines = [
        "# Multi-Day ETA Regime Study",
        "",
        f"- Trade dates: {', '.join(TRADE_DATES)}",
        f"- ETA candidates: {ETA_CANDIDATES}",
        f"- Expiry/grid cases: {len(practical)} practical selections",
        "",
        "## Practical ETA Counts by DTE Bucket, Dense Anchors",
        "",
        eta_by_dte_dense.to_markdown(),
        "",
        "## Practical ETA Counts by DTE Bucket, Market-Only Anchors",
        "",
        eta_by_dte_market.to_markdown(),
        "",
        "## Practical ETA Counts by Quote Bucket, Dense Anchors",
        "",
        eta_by_quote_dense.to_markdown(),
        "",
        "## Largest Dense-Anchor Coverage Gains",
        "",
        hard[
            [
                "trade_date",
                "exdate",
                "dte",
                "n_quotes",
                "eta_market_only",
                "coverage_market_only",
                "eta_dense_full_tail",
                "coverage_dense_full_tail",
                "coverage_gain_dense_vs_market",
                "loss_ratio_dense_vs_market",
            ]
        ].to_markdown(index=False),
        "",
        "## Worst Dense-Anchor Practical Fits",
        "",
        bad_dense[
            [
                "trade_date",
                "exdate",
                "dte",
                "n_quotes",
                "eta_dense_full_tail",
                "coverage_dense_full_tail",
                "weighted_outside_dense_full_tail",
                "mid_rmse_dense_full_tail",
            ]
        ].to_markdown(index=False),
    ]
    (OUT / "multi_day_eta_regime_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_sweeps = []
    summaries = []
    for trade_date in TRADE_DATES:
        sweeps, summary = run_single_sweeps_for_date(trade_date)
        all_sweeps.append(sweeps)
        summaries.append(summary)
        # Incremental save so a later date does not lose earlier work.
        pd.concat(all_sweeps, ignore_index=True).to_csv(OUT / "multi_day_eta_sweeps.partial.csv", index=False)
    sweeps = pd.concat(all_sweeps, ignore_index=True)
    date_summary = pd.concat(summaries, ignore_index=True)
    winners = choose_winners(sweeps)
    save_summary(sweeps, winners, date_summary)
    print(f"Wrote multi-day ETA regime study to {OUT}")


if __name__ == "__main__":
    main()
