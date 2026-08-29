from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "option_surface_step1.ipynb"
OUT = ROOT / "xtx_diagnostics"
TARGET_DTES = [59, 74, 88]
ZOOM_MIN = 0.94
ZOOM_MAX = 1.06


def execute_state(stop_cell: int = 57) -> dict[str, Any]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    env: dict[str, Any] = {"__name__": "__xtx_market_calendar__", "__file__": str(NOTEBOOK)}
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


def select_expiries(joint_inputs: dict[pd.Timestamp, dict[str, Any]]) -> list[pd.Timestamp]:
    exdates = []
    for target in TARGET_DTES:
        exdate = min(
            (pd.Timestamp(x) for x in joint_inputs),
            key=lambda x: abs(int(joint_inputs[x]["dte"]) - target),
        )
        if exdate not in exdates:
            exdates.append(exdate)
    return exdates


def quote_frame(joint_inputs: dict[pd.Timestamp, dict[str, Any]], exdate: pd.Timestamp) -> pd.DataFrame:
    item = joint_inputs[exdate]
    q = item["quotes"].copy().sort_values("k").reset_index(drop=True)
    q["exdate"] = pd.Timestamp(exdate).date()
    q["dte"] = int(item["dte"])
    return q


def interp_curve(q: pd.DataFrame, grid: np.ndarray, col: str) -> np.ndarray:
    x = q["k"].to_numpy(dtype=float)
    y = q[col].to_numpy(dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    uniq = pd.DataFrame({"k": x, col: y}).groupby("k", as_index=False)[col].median()
    return np.interp(grid, uniq["k"], uniq[col], left=np.nan, right=np.nan)


def save_market_curves(frames: list[pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    for q in frames:
        dte = int(q["dte"].iloc[0])
        ax.fill_between(q["k"], q["pure_bid"], q["pure_ask"], alpha=0.18)
        ax.plot(q["k"], q["pure_mid"], marker="o", markersize=3, linewidth=1.6, label=f"{dte}D mid")
    ax.axvspan(1.00, 1.03, color="gray", alpha=0.10, label="k 1.00-1.03")
    ax.set_xlim(ZOOM_MIN, ZOOM_MAX)
    ax.set_title("Market Pure Call Curves Around ATM")
    ax.set_xlabel("normalized strike k")
    ax.set_ylabel("pure call price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "11_market_pure_call_59d_74d_88d_zoom.png")
    plt.close(fig)


def save_market_calendar(frames: list[pd.DataFrame]) -> pd.DataFrame:
    left = max(float(q["k"].min()) for q in frames)
    right = min(float(q["k"].max()) for q in frames)
    grid = np.linspace(max(left, ZOOM_MIN), min(right, ZOOM_MAX), 300)

    rows = []
    curves = {}
    for q in frames:
        dte = int(q["dte"].iloc[0])
        curves[dte] = {
            "bid": interp_curve(q, grid, "pure_bid"),
            "mid": interp_curve(q, grid, "pure_mid"),
            "ask": interp_curve(q, grid, "pure_ask"),
        }

    dtes = sorted(curves)
    fig, axes = plt.subplots(len(dtes) - 1, 1, figsize=(12, 7), dpi=150, sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, prev_dte, next_dte in zip(axes, dtes[:-1], dtes[1:]):
        mid_diff = curves[next_dte]["mid"] - curves[prev_dte]["mid"]
        bid_ask_gap = curves[next_dte]["bid"] - curves[prev_dte]["ask"]
        ax.plot(grid, mid_diff, linewidth=1.8, label=f"mid {next_dte}D - {prev_dte}D")
        ax.plot(grid, bid_ask_gap, linewidth=1.8, label=f"bid_next - ask_prev")
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.axvspan(1.00, 1.03, color="gray", alpha=0.10)
        ax.set_ylabel("calendar diff")
        ax.grid(True, alpha=0.25)
        ax.legend()

        mask = np.isfinite(mid_diff) & np.isfinite(bid_ask_gap)
        zoom = mask & (grid >= 1.00) & (grid <= 1.03)
        rows.append(
            {
                "pair": f"{prev_dte}D->{next_dte}D",
                "grid_min": float(grid[mask].min()) if mask.any() else np.nan,
                "grid_max": float(grid[mask].max()) if mask.any() else np.nan,
                "min_mid_diff": float(np.nanmin(mid_diff[mask])) if mask.any() else np.nan,
                "min_bid_next_minus_ask_prev": float(np.nanmin(bid_ask_gap[mask])) if mask.any() else np.nan,
                "min_mid_diff_k_1_00_1_03": float(np.nanmin(mid_diff[zoom])) if zoom.any() else np.nan,
                "min_bid_next_minus_ask_prev_k_1_00_1_03": float(np.nanmin(bid_ask_gap[zoom])) if zoom.any() else np.nan,
            }
        )

    axes[-1].set_xlabel("normalized strike k")
    fig.suptitle("Market Calendar Checks Around 74D")
    fig.tight_layout()
    fig.savefig(OUT / "12_market_calendar_around_74d_zoom.png")
    plt.close(fig)
    return pd.DataFrame(rows)


def save_atm_table(frames: list[pd.DataFrame], calendar_summary: pd.DataFrame) -> None:
    all_quotes = pd.concat(frames, ignore_index=True)
    zoom = all_quotes[(all_quotes["k"] >= 0.99) & (all_quotes["k"] <= 1.04)].copy()
    zoom = zoom.sort_values(["dte", "k"]).reset_index(drop=True)
    cols = [
        "exdate",
        "dte",
        "k",
        "source_cp_flag",
        "pure_bid",
        "pure_mid",
        "pure_ask",
        "volume",
        "open_interest",
        "source_vega_over_sqrt_t",
    ]
    cols = [c for c in cols if c in zoom.columns]
    zoom[cols].to_csv(OUT / "market_quotes_59d_74d_88d_atm_zoom.csv", index=False)
    calendar_summary.to_csv(OUT / "market_calendar_around_74d_summary.csv", index=False)

    lines = [
        "# Market Calendar Around 74D",
        "",
        "## Calendar Summary",
        "",
        calendar_summary.to_markdown(index=False),
        "",
        "## ATM / Light-Right Quotes",
        "",
        zoom[cols].to_markdown(index=False),
    ]
    (OUT / "market_calendar_around_74d_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    env = execute_state()
    joint_inputs = env["joint_inputs"]
    exdates = select_expiries(joint_inputs)
    frames = [quote_frame(joint_inputs, x) for x in exdates]

    save_market_curves(frames)
    calendar_summary = save_market_calendar(frames)
    save_atm_table(frames, calendar_summary)
    print(f"Wrote market calendar diagnostics to {OUT}")


if __name__ == "__main__":
    main()
