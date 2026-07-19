#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$ROOT_DIR/scripts/figures/plot_result1_benchmark_frontier.py"
python "$ROOT_DIR/scripts/figures/plot_result2_monthly_mae_equation_comparison.py"
python "$ROOT_DIR/scripts/figures/plot_result2_annual_metrics_combo.py"
python "$ROOT_DIR/scripts/figures/plot_result2_daily_prediction_2025.py"
python "$ROOT_DIR/scripts/figures/plot_result2_fixed_equation_daily_stability_2023_2024.py"
