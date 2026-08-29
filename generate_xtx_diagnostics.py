from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "option_surface_step1.ipynb"
OUT = ROOT / "xtx_diagnostics"

def execute_notebook_state(stop_cell: int = 79) -> dict[str, Any]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    env: dict[str, Any] = {"__name__": "__xtx_diagnostics__", "__file__": str(NOTEBOOK)}
    for idx, cell in enumerate(nb["cells"]):
        if idx > stop_cell:
            break
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if ".show()" in src:
            continue
        print(f"execute cell {idx}", flush=True)
        exec(compile(src, f"{NOTEBOOK.name}:cell{idx}", "exec"), env)
    return env

def _cell(nb: dict[str, Any], idx: int) -> dict[str, Any]:
    return nb["cells"][idx]


def _html_table(nb: dict[str, Any], idx: int) -> pd.DataFrame:
    for output in _cell(nb, idx).get("outputs", []):
        html = output.get("data", {}).get("text/html")
        if html is None:
            continue
        if isinstance(html, list):
            html = "".join(html)
        frames = pd.read_html(StringIO(html))
        if not frames:
            break
        df = frames[0]
        return df.drop(columns=["Unnamed: 0"], errors="ignore")
    raise ValueError(f"cell {idx} does not contain an HTML table")


def _plotly_surface(nb: dict[str, Any], idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fig = _cell(nb, idx)["outputs"][0]["data"]["application/vnd.plotly.v1+json"]
    surface = fig["data"][0]
    return (
        np.asarray(surface["x"], dtype=float),
        np.asarray(surface["y"], dtype=float),
        np.asarray(surface["z"], dtype=float),
    )


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def save_joint_fit(joint: pd.DataFrame) -> None:
    dte = joint["dte"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150, sharex=True)

    axes[0, 0].plot(dte, joint["bid_ask_coverage"], marker="o", linewidth=1.8)
    axes[0, 0].axhline(joint["bid_ask_coverage"].median(), color="gray", linestyle="--", linewidth=1)
    axes[0, 0].set_ylabel("bid/ask coverage")
    axes[0, 0].set_ylim(0, 1.05)

    axes[0, 1].plot(dte, joint["mid_rmse"], marker="o", color="#b24c00", linewidth=1.8)
    axes[0, 1].set_ylabel("mid RMSE")

    axes[1, 0].plot(dte, joint["max_outside_bid_ask"], marker="o", color="#7a3db8", linewidth=1.8)
    axes[1, 0].set_ylabel("max outside bid/ask")
    axes[1, 0].set_xlabel("DTE")

    axes[1, 1].bar(dte, joint["nonzero_q_count"], width=1.8, color="#2f7f73")
    axes[1, 1].set_ylabel("nonzero q count")
    axes[1, 1].set_xlabel("DTE")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    fig.suptitle("Joint LP Fit Diagnostics by Expiry")
    fig.tight_layout()
    fig.savefig(OUT / "01_joint_lp_fit_diagnostics.png")
    plt.close(fig)


def save_calendar(calendar: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.plot(calendar["next_dte"], calendar["min_calendar_diff"], marker="o", linewidth=1.8)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("next expiry DTE")
    ax.set_ylabel("min C_next - C_prev")
    ax.set_title("Joint LP Calendar Grid Diagnostics")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "02_calendar_grid_diagnostics.png")
    plt.close(fig)


def save_lp_qp(qcomp: pd.DataFrame) -> None:
    if qcomp.empty or "dte" not in qcomp.columns:
        return
    dte = qcomp["dte"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150, sharex=True)

    axes[0, 0].plot(dte, qcomp["rough_lp"], marker="o", label="LP", linewidth=1.8)
    axes[0, 0].plot(dte, qcomp["rough_qp"], marker="o", label="QP", linewidth=1.8)
    axes[0, 0].set_ylabel("q roughness")
    axes[0, 0].legend()

    axes[0, 1].plot(dte, qcomp["nnz_lp"], marker="o", label="LP", linewidth=1.8)
    axes[0, 1].plot(dte, qcomp["nnz_qp"], marker="o", label="QP", linewidth=1.8)
    axes[0, 1].set_ylabel("nonzero q count")
    axes[0, 1].legend()

    axes[1, 0].plot(dte, qcomp["core_cov_lp"], marker="o", label="LP", linewidth=1.8)
    axes[1, 0].plot(dte, qcomp["core_cov_qp"], marker="o", label="QP", linewidth=1.8)
    axes[1, 0].set_ylabel("core coverage")
    axes[1, 0].set_xlabel("DTE")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].legend()

    axes[1, 1].plot(dte, qcomp["wing_cov_lp"], marker="o", label="LP", linewidth=1.8)
    axes[1, 1].plot(dte, qcomp["wing_cov_qp"], marker="o", label="QP", linewidth=1.8)
    axes[1, 1].set_ylabel("wing coverage")
    axes[1, 1].set_xlabel("DTE")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    fig.suptitle("LP vs Wing-Stabilized QP Diagnostics")
    fig.tight_layout()
    fig.savefig(OUT / "03_lp_qp_stability_diagnostics.png")
    plt.close(fig)


def save_s33(joint: pd.DataFrame, s33: pd.DataFrame) -> None:
    if s33.empty or "coverage" not in s33.columns:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True)
    axes[0].plot(joint["dte"], joint["bid_ask_coverage"], marker="o", label="Joint LP", linewidth=1.8)
    axes[0].plot(s33["dte"], s33["coverage"], marker="o", label="S3.3 forward-product", linewidth=1.8)
    axes[0].set_ylabel("bid/ask coverage")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    axes[1].plot(joint["dte"], joint["mid_rmse"], marker="o", label="Joint LP", linewidth=1.8)
    axes[1].plot(s33["dte"], s33["mid_rmse"], marker="o", label="S3.3 forward-product", linewidth=1.8)
    axes[1].set_xlabel("DTE")
    axes[1].set_ylabel("mid RMSE")
    axes[1].legend()

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Joint LP vs S3.3 Forward-Product")
    fig.tight_layout()
    fig.savefig(OUT / "04_lp_vs_s33_diagnostics.png")
    plt.close(fig)


def save_surfaces(surface_k_grid: np.ndarray, surface_dte: np.ndarray, pure_call_surface: np.ndarray, otm_iv_surface: np.ndarray) -> None:
    for values, filename, title, zlabel in [
        (pure_call_surface, "05_pure_call_surface.png", "Fitted SANOS Pure Call Surface", "pure call price"),
        (np.where(np.isfinite(otm_iv_surface), otm_iv_surface, np.nan), "06_otm_iv_surface.png", "Fitted SANOS OTM IV Surface", "OTM IV"),
    ]:
        if values.size == 0:
            continue
        xx, yy = np.meshgrid(surface_k_grid, surface_dte)
        fig = plt.figure(figsize=(11, 7), dpi=150)
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(xx, yy, values, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel("normalized strike k")
        ax.set_ylabel("DTE")
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        ax.view_init(elev=25, azim=-135)
        fig.colorbar(surf, shrink=0.55, pad=0.1)
        fig.tight_layout()
        fig.savefig(OUT / filename)
        plt.close(fig)


def save_iv_slices(surface_k_grid: np.ndarray, surface_dte: np.ndarray, otm_iv_surface: np.ndarray) -> None:
    if otm_iv_surface.size == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    for dte in [1, 2, 7, 14, 31, 59, 74, 88]:
        where = np.where(surface_dte == dte)[0]
        if len(where):
            ax.plot(surface_k_grid, otm_iv_surface[where[0]], label=f"{dte}D", linewidth=1.8)
    ax.set_xlabel("normalized strike k")
    ax.set_ylabel("OTM implied volatility")
    ax.set_title("OTM IV Surface Slices")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / "07_otm_iv_slices.png")
    plt.close(fig)


def save_summary(joint: pd.DataFrame, calendar: pd.DataFrame, qcomp: pd.DataFrame, s33: pd.DataFrame) -> None:
    worst_joint = joint.sort_values("bid_ask_coverage").head(5)
    cal_worst = calendar["min_calendar_diff"].min()

    lines = [
        "# XTX/SANOS Diagnostics Summary",
        "",
        "## Headline",
        "",
        f"- Joint LP expiries: {len(joint)}",
        f"- Joint LP median bid/ask coverage: {joint['bid_ask_coverage'].median():.3f}",
        f"- Joint LP worst bid/ask coverage: {joint['bid_ask_coverage'].min():.3f}",
        f"- Joint LP max outside bid/ask: {joint['max_outside_bid_ask'].max():.6f}",
        f"- Calendar grid worst min diff: {cal_worst:+.3e}",
    ]
    if not qcomp.empty and {"rough_lp", "rough_qp"}.issubset(qcomp.columns):
        rough_drop = 1.0 - qcomp["rough_qp"].median() / qcomp["rough_lp"].median()
        lines.extend([
            f"- LP median q roughness: {qcomp['rough_lp'].median():.4f}",
            f"- QP median q roughness: {qcomp['rough_qp'].median():.4f}",
            f"- QP roughness drop: {rough_drop:.1%}",
        ])
    else:
        lines.append("- QP comparison: unavailable in this run")
    if not s33.empty and "coverage" in s33.columns:
        lines.append(f"- S3.3 median bid/ask coverage: {s33['coverage'].median():.3f}")
    else:
        lines.append("- S3.3 comparison: unavailable in this run")

    lines.extend([
        "",
        "## Worst Joint LP Expiries",
        "",
        worst_joint[
            ["exdate", "dte", "n_quotes", "bid_ask_coverage", "max_outside_bid_ask", "mid_rmse", "nonzero_q_count"]
        ].to_markdown(index=False),
    ])
    if not s33.empty and "coverage" in s33.columns:
        worst_s33 = s33.sort_values("coverage").head(5)
        lines.extend([
            "",
            "## Worst S3.3 Expiries",
            "",
            worst_s33[["exdate", "dte", "n_quotes", "coverage", "mid_rmse", "dV_med"]].to_markdown(index=False),
        ])
    lines.extend([
        "",
        "## Notes",
        "",
        "- The pure call price surface is the primary fitted object.",
        "- The OTM IV surface is a diagnostic transform and is very sensitive in short-dated low-vega wings.",
        "- The 74D expiry remains the main price-space weak point in the project case study.",
    ])
    (OUT / "diagnostic_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    env = execute_notebook_state()

    joint = env["joint_per_expiry"].copy()
    calendar = env["joint_calendar_checks"].copy()
    qcomp = env.get("wing_compare", pd.DataFrame()).copy()
    s33 = env.get("joint_s33_per_expiry", pd.DataFrame()).copy()

    joint.to_csv(OUT / "joint_lp_per_expiry.csv", index=False)
    calendar.to_csv(OUT / "calendar_grid_checks.csv", index=False)
    qcomp.to_csv(OUT / "lp_qp_stability.csv", index=False)
    s33.to_csv(OUT / "s33_per_expiry.csv", index=False)

    save_joint_fit(joint)
    save_calendar(calendar)
    save_lp_qp(qcomp)
    save_s33(joint, s33)
    save_surfaces(
        np.asarray(env["surface_k_grid"], dtype=float),
        np.asarray(env["surface_dte"], dtype=float),
        np.asarray(env["pure_call_surface"], dtype=float),
        np.asarray(env["otm_iv_surface"], dtype=float),
    )
    save_iv_slices(
        np.asarray(env["surface_k_grid"], dtype=float),
        np.asarray(env["surface_dte"], dtype=float),
        np.asarray(env["otm_iv_surface"], dtype=float),
    )
    save_summary(joint, calendar, qcomp, s33)

    print(f"Wrote diagnostics to {OUT}")


if __name__ == "__main__":
    main()
