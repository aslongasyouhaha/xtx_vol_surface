from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "xtx_diagnostics"
QUOTE_CSV = OUT / "74d_quote_level_diagnostics.csv"


def convex_decreasing_fit(k: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> tuple[bool, np.ndarray | None, str]:
    n = len(k)
    aub = []
    bub = []

    # Decreasing call price: c[i + 1] - c[i] <= 0.
    for i in range(n - 1):
        row = np.zeros(n)
        row[i + 1] = 1.0
        row[i] = -1.0
        aub.append(row)
        bub.append(0.0)

    # Convexity on irregular k grid: consecutive slopes must be nondecreasing.
    for i in range(n - 2):
        h0 = k[i + 1] - k[i]
        h1 = k[i + 2] - k[i + 1]
        row = np.zeros(n)
        row[i] = 1.0 / h0
        row[i + 1] = -1.0 / h0 - 1.0 / h1
        row[i + 2] = 1.0 / h1
        aub.append(-row)
        bub.append(0.0)

    res = linprog(
        np.zeros(n),
        A_ub=np.asarray(aub),
        b_ub=np.asarray(bub),
        bounds=list(zip(bid, ask)),
        method="highs",
    )
    if not res.success:
        return False, None, res.message
    return True, np.asarray(res.x, dtype=float), res.message


def slope_table(k: np.ndarray, mid: np.ndarray) -> pd.DataFrame:
    slopes = np.diff(mid) / np.diff(k)
    rows = []
    for i, slope in enumerate(slopes):
        rows.append(
            {
                "left_k": k[i],
                "right_k": k[i + 1],
                "mid_k": 0.5 * (k[i] + k[i + 1]),
                "slope": slope,
                "slope_change_from_prev": np.nan if i == 0 else slope - slopes[i - 1],
            }
        )
    return pd.DataFrame(rows)


def save_plots(q: pd.DataFrame, convex_curve: np.ndarray | None) -> None:
    k = q["k"].to_numpy(dtype=float)
    intrinsic = np.maximum(1.0 - k, 0.0)
    extrinsic = q["pure_mid"].to_numpy(dtype=float) - intrinsic
    slopes = np.diff(q["pure_mid"].to_numpy(dtype=float)) / np.diff(k)
    slope_changes = np.diff(slopes)
    slope_x = 0.5 * (k[:-1] + k[1:])
    change_x = k[1:-1]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    axes[0, 0].fill_between(q["k"], q["pure_bid"], q["pure_ask"], alpha=0.25, label="bid/ask")
    axes[0, 0].plot(q["k"], q["pure_mid"], marker="o", linewidth=1.8, label="mid")
    axes[0, 0].plot(q["k"], intrinsic, linestyle="--", linewidth=1.4, label="intrinsic")
    if convex_curve is not None:
        axes[0, 0].plot(q["k"], convex_curve, linewidth=1.8, label="feasible convex curve")
    axes[0, 0].set_title("74D Pure Call Shape")
    axes[0, 0].set_xlabel("normalized strike k")
    axes[0, 0].set_ylabel("pure call price")
    axes[0, 0].legend()

    axes[0, 1].plot(q["k"], extrinsic, marker="o", linewidth=1.8)
    axes[0, 1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].set_title("74D Pure Call Extrinsic Value")
    axes[0, 1].set_xlabel("normalized strike k")
    axes[0, 1].set_ylabel("pure_mid - intrinsic")

    axes[1, 0].plot(slope_x, slopes, marker="o", linewidth=1.8)
    axes[1, 0].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1, 0].set_title("74D Mid Slopes")
    axes[1, 0].set_xlabel("normalized strike k midpoint")
    axes[1, 0].set_ylabel("dC / dk")

    axes[1, 1].bar(change_x, slope_changes, width=0.006)
    axes[1, 1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1, 1].set_title("74D Slope Changes")
    axes[1, 1].set_xlabel("normalized strike k")
    axes[1, 1].set_ylabel("next slope - previous slope")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "13_74d_static_shape_diagnostics.png")
    plt.close(fig)


def main() -> None:
    q = pd.read_csv(QUOTE_CSV).sort_values("k").reset_index(drop=True)
    k = q["k"].to_numpy(dtype=float)
    bid = q["pure_bid"].to_numpy(dtype=float)
    mid = q["pure_mid"].to_numpy(dtype=float)
    ask = q["pure_ask"].to_numpy(dtype=float)

    slopes = np.diff(mid) / np.diff(k)
    slope_changes = np.diff(slopes)
    feasible, convex_curve, message = convex_decreasing_fit(k, bid, ask)

    slopes_df = slope_table(k, mid)
    slopes_df.to_csv(OUT / "74d_mid_slope_diagnostics.csv", index=False)
    if convex_curve is not None:
        curve = q[["exdate", "dte", "k", "pure_bid", "pure_mid", "pure_ask"]].copy()
        curve["convex_feasible_price"] = convex_curve
        curve["move_from_mid"] = convex_curve - mid
        curve.to_csv(OUT / "74d_convex_feasible_curve.csv", index=False)

    save_plots(q, convex_curve)

    intrinsic = np.maximum(1.0 - k, 0.0)
    extrinsic = mid - intrinsic
    lines = [
        "# 74D Static Shape Diagnostics",
        "",
        f"- Quotes: {len(q)}",
        f"- Mid monotone decreasing violations: {int(np.sum(slopes > 1e-12))}",
        f"- Mid convex slope violations: {int(np.sum(slope_changes < -1e-12))}",
        f"- Minimum slope change: {float(np.min(slope_changes)):.6g}",
        f"- Arbitrary convex decreasing curve inside bid/ask feasible: {feasible}",
        f"- Feasibility message: {message}",
        f"- Minimum extrinsic value: {float(np.min(extrinsic)):.6g}",
        f"- Maximum extrinsic value: {float(np.max(extrinsic)):.6g}",
    ]
    if convex_curve is not None:
        lines.append(f"- Max absolute adjustment from mid to feasible curve: {float(np.max(np.abs(convex_curve - mid))):.6g}")
    lines.extend(
        [
            "",
            "## Largest Slope Changes",
            "",
            slopes_df.sort_values("slope_change_from_prev", ascending=False)
            .head(8)
            .to_markdown(index=False),
            "",
            "## Smallest Slope Changes",
            "",
            slopes_df.sort_values("slope_change_from_prev", ascending=True)
            .head(8)
            .to_markdown(index=False),
        ]
    )
    (OUT / "74d_static_shape_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote 74D static shape diagnostics to {OUT}")


if __name__ == "__main__":
    main()
