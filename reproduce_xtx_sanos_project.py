from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_step(label: str, command: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the final XTX/SANOS surface diagnostics.")
    parser.add_argument("--trade-date", default="2025-06-02", help="Case-study trade date in YYYY-MM-DD format.")
    parser.add_argument("--skip-core-diagnostics", action="store_true", help="Skip baseline diagnostic figures.")
    parser.add_argument("--skip-auto-regime", action="store_true", help="Skip automatic regime remediation.")
    parser.add_argument(
        "--multi-day-regime",
        action="store_true",
        help="Run the full multi-day ETA regime study. This can take several minutes.",
    )
    parser.add_argument(
        "--summarize-multi-day-partial",
        action="store_true",
        help="Summarize an existing multi-day partial CSV without rerunning the sweep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable

    if not args.skip_core_diagnostics:
        run_step("baseline diagnostics", [py, "generate_xtx_diagnostics.py"])

    if not args.skip_auto_regime:
        run_step(
            "automatic regime remediation",
            [py, "generate_xtx_auto_regime_remediation.py", "--trade-date", args.trade_date],
        )

    if args.multi_day_regime:
        run_step("multi-day ETA regime sweep", [py, "generate_xtx_multi_day_eta_regime_study.py"])

    if args.summarize_multi_day_partial:
        run_step("summarize multi-day partial sweep", [py, "summarize_xtx_multi_day_eta_partial.py"])

    print("\nDone. Main outputs:")
    print("  xtx_diagnostics/diagnostic_summary.md")
    print("  xtx_diagnostics/auto_regime/auto_regime_summary.md")
    print("  xtx_diagnostics/multi_day_eta/multi_day_eta_regime_summary.md")


if __name__ == "__main__":
    main()
