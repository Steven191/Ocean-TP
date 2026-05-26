# OOI Ocean Temperature Modeling Project

This project contains the code and results for modeling ocean temperature profiles using OOI data.

## Directory Structure

- `src/`: Python source code for experiments and modeling.
- `data/`: Input dataset (`ooi-new.csv`).
- `results/`: Generated reports, metrics (JSON/CSV), and summary markdown.
  - `figures/`: Generated plots.

## How to Run

1. Navigate to the `src` directory:
   ```bash
   cd src
   ```

2. Run the experiments individually:

   - **Basis Function Experiment**:
     ```bash
     python3 term_increase_experiment_new.py
     ```
   
   - **D-Optimal Fitting**:
     ```bash
     python3 doptimal8_fit_new.py
     ```
   
   - **Seasonal Validation**:
     ```bash
     python3 time_season_validation_new.py
     ```
   
   - **Recent Prediction Accuracy**:
     ```bash
     python3 recent_prediction_new.py --data-dir ../data --output ../results/recent_prediction_results.json
     ```

3. Generate the Final Report:
   ```bash
   python3 generate_final_report.py
   ```
   The report will be available at `results/experiment_summary.md`.

## Requirements

- Python 3.8+
- numpy, pandas, matplotlib, scikit-learn
