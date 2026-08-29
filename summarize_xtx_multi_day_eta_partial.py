from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path("xtx_diagnostics/multi_day_eta")
PARTIAL = OUT / "multi_day_eta_sweeps.partial.csv"
DTE_BINS = [0, 7, 21, 45, 90, 180, 10000]
DTE_LABELS = ["1-7D", "8-21D", "22-45D", "46-90D", "91-180D", "181D+"]


def choose_winners(sweeps: pd.DataFrame) -> pd.DataFrame:
    ok = sweeps[sweeps["success"].astype(bool)].copy()
    winners = []
    keys = ["trade_date", "exdate", "dte", "n_quotes", "grid"]
    for _, group in ok.groupby(keys, sort=False):
        best_cov = group.sort_values(
            ["coverage", "weighted_outside", "mid_rmse", "q_roughness"],
            ascending=[False, True, True, True],
        ).iloc[0]
        best_loss = group.sort_values(
            ["weighted_outside", "mid_rmse", "q_roughness"],
            ascending=[True, True, True],
        ).iloc[0]
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


def pivot_counts(df: pd.DataFrame, index: str, columns: str) -> pd.DataFrame:
    return (
        df.pivot_table(
            index=index,
            columns=columns,
            values="exdate",
            aggfunc="count",
            fill_value=0,
            observed=False,
        )
        .sort_index()
    )


def summarize_success(sweeps: pd.DataFrame) -> pd.DataFrame:
    ok = sweeps.copy()
    ok["dte_bucket"] = pd.cut(ok["dte"], bins=DTE_BINS, labels=DTE_LABELS, right=True, include_lowest=True)
    return (
        ok.groupby(["grid", "eta", "dte_bucket"], observed=True)
        .agg(
            n=("exdate", "count"),
            success_rate=("success", "mean"),
            median_coverage=("coverage", "median"),
            median_weighted_outside=("weighted_outside", "median"),
            median_q_roughness=("q_roughness", "median"),
        )
        .reset_index()
    )


def main() -> None:
    sweeps = pd.read_csv(PARTIAL)
    complete_dates = sweeps.groupby("trade_date")["exdate"].nunique().sort_index()
    winners = choose_winners(sweeps)
    practical = winners[winners["selection"].eq("practical_smooth")].copy()
    dense = practical[practical["grid"].eq("dense_full_tail")]
    market = practical[practical["grid"].eq("market_only")]

    improvement = practical.pivot_table(
        index=["trade_date", "exdate", "dte", "n_quotes"],
        columns="grid",
        values=["coverage", "weighted_outside", "mid_rmse", "eta", "q_roughness"],
        aggfunc="first",
    )
    improvement.columns = [f"{a}_{b}" for a, b in improvement.columns]
    improvement = improvement.reset_index()
    improvement["coverage_gain_dense_vs_market"] = (
        improvement["coverage_dense_full_tail"] - improvement["coverage_market_only"]
    )
    improvement["loss_ratio_dense_vs_market"] = (
        improvement["weighted_outside_dense_full_tail"]
        / improvement["weighted_outside_market_only"].replace(0, np.nan)
    )
    improvement["roughness_ratio_dense_vs_market"] = (
        improvement["q_roughness_dense_full_tail"]
        / improvement["q_roughness_market_only"].replace(0, np.nan)
    )

    success = summarize_success(sweeps)
    date_summary = (
        sweeps.groupby("trade_date")
        .agg(
            n_expiries=("exdate", "nunique"),
            n_sweep_rows=("exdate", "size"),
            median_quotes=("n_quotes", "median"),
            min_dte=("dte", "min"),
            max_dte=("dte", "max"),
        )
        .reset_index()
    )

    sweeps.to_csv(OUT / "multi_day_eta_sweeps_7_complete_dates.csv", index=False)
    winners.to_csv(OUT / "multi_day_eta_winners.csv", index=False)
    improvement.to_csv(OUT / "multi_day_dense_vs_market_improvement.csv", index=False)
    success.to_csv(OUT / "multi_day_eta_success_by_regime.csv", index=False)
    date_summary.to_csv(OUT / "multi_day_date_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    dense.boxplot(column="eta", by="dte_bucket", ax=axes[0, 0])
    axes[0, 0].set_title("Practical ETA by DTE Bucket (Dense Anchors)")
    axes[0, 0].set_xlabel("DTE bucket")
    axes[0, 0].set_ylabel("ETA")

    dense.boxplot(column="eta", by="quote_bucket", ax=axes[0, 1])
    axes[0, 1].set_title("Practical ETA by Quote Count (Dense Anchors)")
    axes[0, 1].set_xlabel("quote bucket")
    axes[0, 1].set_ylabel("ETA")

    axes[1, 0].scatter(improvement["dte"], improvement["coverage_gain_dense_vs_market"], s=18, alpha=0.75)
    axes[1, 0].axhline(0.0, color="gray", linestyle="--")
    axes[1, 0].set_title("Dense Anchor Coverage Gain vs DTE")
    axes[1, 0].set_xlabel("DTE")
    axes[1, 0].set_ylabel("coverage gain")

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

    eta_by_dte_dense = pivot_counts(dense, "dte_bucket", "eta")
    eta_by_dte_market = pivot_counts(market, "dte_bucket", "eta")
    eta_by_quote_dense = pivot_counts(dense, "quote_bucket", "eta")

    top_gain = improvement.sort_values("coverage_gain_dense_vs_market", ascending=False).head(15)
    worst_dense = improvement.sort_values("coverage_dense_full_tail").head(15)
    eta_medians = (
        practical.groupby(["grid", "dte_bucket"], observed=True)
        .agg(
            n=("eta", "count"),
            median_eta=("eta", "median"),
            median_coverage=("coverage", "median"),
            median_weighted_outside=("weighted_outside", "median"),
            median_q_roughness=("q_roughness", "median"),
        )
        .reset_index()
    )

    lines = [
        "# Multi-Day ETA Regime Study",
        "",
        f"- Completed trade dates: {', '.join(complete_dates.index.astype(str))}",
        f"- Expiries: {int(complete_dates.sum())} expiry-days; practical selections: {len(practical)}",
        "- Note: 2025-08-01 full sweep was interrupted because one dense long-tail case was too slow; the saved sample contains 7 complete dates.",
        "",
        "## Date Coverage",
        "",
        date_summary.to_markdown(index=False),
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
        "## Practical Median Metrics by Bucket",
        "",
        eta_medians.to_markdown(index=False),
        "",
        "## Largest Dense-Anchor Coverage Gains",
        "",
        top_gain[
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
        worst_dense[
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
    print(f"Wrote summary from {PARTIAL}")
    print(date_summary.to_string(index=False))


if __name__ == "__main__":
    main()
