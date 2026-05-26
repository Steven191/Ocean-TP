#!/bin/bash
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_DIR="$SCRIPT_DIR/src"

echo "Starting OOI Experiment Pipeline..."
echo "Working directory: $SRC_DIR"

cd "$SRC_DIR"

# 1. Term Increase Experiment
echo "------------------------------------------------"
echo "Running Experiment 1: Term Increase (Basis Functions)..."
python3 term_increase_experiment_new.py
echo "Done."

# 2. D-Optimal Fit
echo "------------------------------------------------"
echo "Running Experiment 4: D-Optimal Fitting..."
python3 doptimal8_fit_new.py
echo "Done."

# 3. Seasonal Validation
echo "------------------------------------------------"
echo "Running Seasonal Validation..."
python3 time_season_validation_new.py
echo "Done."

# 4. Recent Prediction
echo "------------------------------------------------"
echo "Running Recent Prediction Accuracy (2020-2025)..."
python3 recent_prediction_new.py --data-dir ../data --output ../results/recent_prediction_results.json
echo "Done."

# 5. Generate Report
echo "------------------------------------------------"
echo "Generating Final Report..."
python3 generate_final_report.py
echo "Done."

echo "------------------------------------------------"
echo "All experiments completed successfully."
echo "Final Report: $SCRIPT_DIR/results/experiment_summary.md"
