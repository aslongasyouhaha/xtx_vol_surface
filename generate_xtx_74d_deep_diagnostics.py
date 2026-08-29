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
TARGET_DTE = 74


def execute_notebook_state(stop_cell: int = 75) -> dict[str, Any]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    env: dict[str, Any] = {
        "__name__": "__xtx_diagnostics__",
        "__file__": str(NOTEBOOK),
    }
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


def coverage_metrics(bid: np.ndarray, mid: np.ndarray, ask: np.ndarray, fit: np.ndarray) -> dict[str, float]:
    inside = (fit >= bid - 1e-9) & (fit <= ask + 1e-9)
    outside = np.maximum(bid - fit, 0.0) + np.maximum(fit - ask, 0.0)
    return {
        "coverage": float(np.mean(inside)),
        "mid_rmse": float(np.sqrt(np.mean((fit - mid) ** 2))),
        "max_outside_bid_ask": float(np.max(outside)),
        "mean_outside_bid_ask": float(np.mean(outside)),
    }


def add_fit_columns(quotes: pd.DataFrame, name: str, fit: np.ndarray) -> None:
    bid = quotes["pure_bid"].to_numpy(dtype=float)
    ask = quotes["pure_ask"].to_numpy(dtype=float)
    quotes[f"fit_{name}"] = fit
    quotes[f"outside_{name}"] = np.maximum(bid - fit, 0.0) + np.maximum(fit - ask, 0.0)
    quotes[f"inside_{name}"] = quotes[f"outside_{name}"] <= 1e-9


def save_74d_price_fit(quotes: pd.DataFrame, target_dte: int) -> None:
    title_tail = "Single vs Joint vs QP" if "fit_qp" in quotes.columns else "Single vs Joint"
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.fill_between(
        quotes["k"],
        quotes["pure_bid"],
        quotes["pure_ask"],
        color="#b9d7f0",
        alpha=0.45,
        label="market bid/ask",
    )
    ax.scatter(quotes["k"], quotes["pure_mid"], s=18, color="#1f77b4", label="market mid", zorder=3)
    ax.plot(quotes["k"], quotes["fit_single"], linewidth=2.0, label="single-expiry LP")
    ax.plot(quotes["k"], quotes["fit_joint"], linewidth=2.0, label="joint LP")
    if "fit_qp" in quotes.columns:
        ax.plot(quotes["k"], quotes["fit_qp"], linewidth=2.0, label="wing-stabilized QP")
    ax.set_title(f"{target_dte}D Pure Call Fit: {title_tail}")
    ax.set_xlabel("normalized strike k")
    ax.set_ylabel("pure call price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "08_74d_pure_call_fit.png")
    plt.close(fig)


def save_74d_outside(quotes: pd.DataFrame, target_dte: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    width = 0.0025
    ax.bar(quotes["k"] - width, quotes["outside_single"], width=width, label="single-expiry LP")
    ax.bar(quotes["k"], quotes["outside_joint"], width=width, label="joint LP")
    if "outside_qp" in quotes.columns:
        ax.bar(quotes["k"] + width, quotes["outside_qp"], width=width, label="QP")
    ax.set_title(f"{target_dte}D Outside Bid/Ask Error by Strike")
    ax.set_xlabel("normalized strike k")
    ax.set_ylabel("outside bid/ask amount")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "09_74d_outside_error_by_strike.png")
    plt.close(fig)


def save_neighbor_calendar(env: dict[str, Any], target_exdate: pd.Timestamp) -> None:
    joint_inputs = env["joint_inputs"]
    joint_solution = env["joint_solution"]
    joint_solution_qp = env["joint_solution_qp"]
    build_kernel_matrix = env["build_kernel_matrix"]

    exdates = sorted(pd.Timestamp(x) for x in joint_inputs)
    pos = exdates.index(target_exdate)
    neighbors = [exdates[pos - 1], target_exdate, exdates[pos + 1]]
    labels = [f"{joint_inputs[x]['dte']}D" for x in neighbors]
    k_min = max(float(joint_inputs[x]["market_k"].min()) for x in neighbors)
    k_max = min(float(joint_inputs[x]["market_k"].max()) for x in neighbors)
    grid = np.linspace(k_min, k_max, 300)

    def model_curve(sol: dict[str, Any], exdate: pd.Timestamp) -> np.ndarray:
        item = joint_inputs[exdate]
        return build_kernel_matrix(item["anchor_k"], grid, item["smooth_variance"]) @ sol["q_by_expiry"][exdate]

    lp_curves = [model_curve(joint_solution, x) for x in neighbors]
    has_qp = "q_by_expiry" in joint_solution_qp
    qp_curves = [model_curve(joint_solution_qp, x) for x in neighbors] if has_qp else None

    nrows = 2 if has_qp else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 8 if has_qp else 5), dpi=150, sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    axes[0].plot(grid, lp_curves[1] - lp_curves[0], label=f"LP {labels[1]} - {labels[0]}", linewidth=1.8)
    axes[0].plot(grid, lp_curves[2] - lp_curves[1], label=f"LP {labels[2]} - {labels[1]}", linewidth=1.8)
    if has_qp and qp_curves is not None:
        axes[1].plot(grid, qp_curves[1] - qp_curves[0], label=f"QP {labels[1]} - {labels[0]}", linewidth=1.8)
        axes[1].plot(grid, qp_curves[2] - qp_curves[1], label=f"QP {labels[2]} - {labels[1]}", linewidth=1.8)

    for ax in axes:
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("C_next - C_prev")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("normalized strike k")
    fig.suptitle("Neighbor Calendar Slack Around 74D")
    fig.tight_layout()
    fig.savefig(OUT / "10_74d_neighbor_calendar_slack.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    env = execute_notebook_state()

    joint_inputs = env["joint_inputs"]
    joint_solution = env["joint_solution"]
    joint_solution_qp = env["joint_solution_qp"]
    solve_single_expiry_lp = env["solve_single_expiry_lp"]

    target_exdate = min(
        (pd.Timestamp(x) for x in joint_inputs),
        key=lambda x: abs(int(joint_inputs[x]["dte"]) - TARGET_DTE),
    )
    item = joint_inputs[target_exdate]
    quotes = item["quotes"].copy().sort_values("k").reset_index(drop=True)

    single_solution = solve_single_expiry_lp(item)
    fit_single = item["G"] @ single_solution["q"]
    fit_joint = item["G"] @ joint_solution["q_by_expiry"][target_exdate]
    has_qp = "q_by_expiry" in joint_solution_qp
    fit_qp = item["G"] @ joint_solution_qp["q_by_expiry"][target_exdate] if has_qp else None

    # item["G"] follows the original quote order; align with the sorted quote frame.
    aligned = item["quotes"].copy().reset_index(drop=True)
    add_fit_columns(aligned, "single", fit_single)
    add_fit_columns(aligned, "joint", fit_joint)
    if fit_qp is not None:
        add_fit_columns(aligned, "qp", fit_qp)
    quotes = aligned.sort_values("k").reset_index(drop=True)

    bid = quotes["pure_bid"].to_numpy(dtype=float)
    mid = quotes["pure_mid"].to_numpy(dtype=float)
    ask = quotes["pure_ask"].to_numpy(dtype=float)
    metrics = pd.DataFrame(
        [
            {"model": "single_expiry_lp", **coverage_metrics(bid, mid, ask, quotes["fit_single"].to_numpy(float))},
            {"model": "joint_lp", **coverage_metrics(bid, mid, ask, quotes["fit_joint"].to_numpy(float))},
        ]
    )
    if "fit_qp" in quotes.columns:
        metrics.loc[len(metrics)] = {
            "model": "wing_stabilized_qp",
            **coverage_metrics(bid, mid, ask, quotes["fit_qp"].to_numpy(float)),
        }

    keep_cols = [
        "exdate",
        "dte",
        "k",
        "strike_price",
        "source_cp_flag",
        "pure_bid",
        "pure_mid",
        "pure_ask",
        "fit_single",
        "outside_single",
        "inside_single",
        "fit_joint",
        "outside_joint",
        "inside_joint",
        "fit_qp",
        "outside_qp",
        "inside_qp",
    ]
    keep_cols = [c for c in keep_cols if c in quotes.columns]
    quotes[keep_cols].to_csv(OUT / "74d_quote_level_diagnostics.csv", index=False)
    metrics.to_csv(OUT / "74d_model_metrics.csv", index=False)

    save_74d_price_fit(quotes, int(item["dte"]))
    save_74d_outside(quotes, int(item["dte"]))
    save_neighbor_calendar(env, target_exdate)

    lines = [
        "# 74D Deep Diagnostics",
        "",
        f"- Target expiry: {target_exdate.date()}",
        f"- DTE: {int(item['dte'])}",
        f"- Quotes: {len(quotes)}",
        "",
        metrics.to_markdown(index=False),
        "",
        "Worst joint-LP outside points:",
        "",
        quotes.sort_values("outside_joint", ascending=False)
        .head(10)[[c for c in ["k", "strike_price", "source_cp_flag", "pure_bid", "pure_mid", "pure_ask", "fit_joint", "outside_joint"] if c in quotes.columns]]
        .to_markdown(index=False),
    ]
    (OUT / "74d_deep_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote 74D deep diagnostics to {OUT}")


if __name__ == "__main__":
    main()
