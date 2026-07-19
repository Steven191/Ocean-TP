# Ocean-TP Manuscript Companion Repository

This repository contains the code, tabulated support records, figures and historical reference materials associated with the Ocean-TP manuscript.

The current manuscript-facing materials are separated from earlier exploratory outputs so that reviewers can inspect the active support files without sorting through historical experiments.

## Repository Map

- `docs/supporting_information/` - compiled supporting information PDF and LaTeX source.
- `data/supporting/` - current CSV records used by the supporting file tables and manuscript-support figures.
- `results/supporting_tables/` - LaTeX table fragments used in the supporting information file.
- `results/figures/supporting/` - current support figures in PDF, SVG and PNG formats where available.
- `scripts/figures/` - scripts for regenerating the current manuscript-support figures from repository CSV files.
- `src/ocean_tp/` - Ocean-TP analysis and modelling scripts from the project code base.
- `data/processed/`, `results/legacy_metrics/`, `docs/legacy_reports/`, `models/legacy/` - earlier processed outputs, model checkpoints and reports retained for traceability.
- `archive/historical_reference/` - older experiment deliverables and manuscript-stage reference material.

## Current Supporting Records

The main index for current tabulated materials is [docs/DATA_INDEX.md](docs/DATA_INDEX.md).

Key records include:

- Result 1 benchmark catalogue: `data/supporting/result1_baseline_metrics_10k_cleaned.csv`
- Result 2 seasonal and monthly Mixed-B equation records: `data/supporting/result2_seasonal_mixedb_coefficients.csv`, `data/supporting/result2_monthly_mixedb_coefficients.csv`
- Result 2 monthly-seasonal-annual MAE comparison: `data/supporting/result2_monthly_seasonal_annual_mae_comparison.csv`
- Result 2 annual error distribution and metrics: `data/supporting/result2_annual_error_distribution_r2_stats.csv`
- Result 2 depth-resolved 1 m accuracy table: `data/supporting/result2_depth_accuracy_1m.csv`
- Result 2 daily 2025 prediction record: `data/supporting/result2_daily_prediction_2025_corrected.csv`
- Result 2 fixed-equation daily stability record: `data/supporting/result2_fixed_equation_daily_stability_2023_2024.csv`

All displayed numerical values, tabulated metrics and incremental changes in the current supporting information file are taken from the manuscript fitting and evaluation records.

## Rebuild Figures

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Regenerate the current support figures:

```bash
python scripts/rebuild_supporting_figures.py
```

Or run each plotting script individually:

```bash
python scripts/figures/plot_result1_benchmark_frontier.py
python scripts/figures/plot_result2_monthly_mae_equation_comparison.py
python scripts/figures/plot_result2_annual_metrics_combo.py
python scripts/figures/plot_result2_daily_prediction_2025.py
python scripts/figures/plot_result2_fixed_equation_daily_stability_2023_2024.py
```

The scripts read from `data/supporting/` and write to `results/figures/supporting/`.

## Legacy Analysis Code

The scripts in `src/ocean_tp/` are retained as the project analysis code used during model development. Some commands require the original OOI CSV files, which are not bundled in full form here. Use `scripts/run_legacy_pipeline.sh` as a documented entry point by setting `OOI_DATA_DIR` to a local directory containing `ooi-*.csv` files.

## License

See [LICENSE](LICENSE).
