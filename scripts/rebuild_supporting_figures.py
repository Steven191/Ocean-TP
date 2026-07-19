"""Regenerate current manuscript-support figures from repository CSV files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    ROOT / "scripts" / "figures" / "plot_result1_benchmark_frontier.py",
    ROOT / "scripts" / "figures" / "plot_result2_monthly_mae_equation_comparison.py",
    ROOT / "scripts" / "figures" / "plot_result2_annual_metrics_combo.py",
    ROOT / "scripts" / "figures" / "plot_result2_daily_prediction_2025.py",
    ROOT / "scripts" / "figures" / "plot_result2_fixed_equation_daily_stability_2023_2024.py",
]


def main() -> None:
    for script in SCRIPTS:
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
