from __future__ import annotations
import _bootstrap  # project paths for direct script execution

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(label: str, command: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(command), flush=True)
    command = [command[0], str(ROOT / "scripts" / command[1]), *command[2:]]
    subprocess.run(command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the 1-120D SANOS convex-order baseline.")
    parser.add_argument("--trade-date", default="2025-06-02", help="Case-study trade date in YYYY-MM-DD format.")
    parser.add_argument("--skip-core-diagnostics", action="store_true", help="Skip baseline diagnostic figures.")
    parser.add_argument("--multi-day-validation", action="store_true", help="Validate the baseline on January 2, April 1 and the selected trade date.")
    parser.add_argument("--dlv-validation",action="store_true",help="Validate DLV generation/recovery and diagnose finite DLV encoding of the saved baseline.")
    parser.add_argument("--dlv-calibration",action="store_true",help="Run the separate direct DLV market-calibration experiment for the selected date(s).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable
    dates = list(dict.fromkeys(["2025-01-02", "2025-04-01", args.trade_date])) if args.multi_day_validation else [args.trade_date]

    if not args.skip_core_diagnostics:
        run_step("convex-order baseline", [py, "run_xtx_baseline.py", "--trade-date", *dates])

    if args.dlv_validation:
        run_step("DLV parametrization validation",[py,"run_xtx_dlv.py","--dates",*dates])

    if args.dlv_calibration:
        run_step("direct DLV market calibration",[py,"run_xtx_dlv_calibration.py","--dates",*dates])

    print("\nDone. Main outputs:")
    print("  data/outputs/baseline_paper_gap001/<trade-date>/summary.json")
    print("  data/outputs/baseline_paper_gap001/multi_day_validation.csv")


if __name__ == "__main__":
    main()
