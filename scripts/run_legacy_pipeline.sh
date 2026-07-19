#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${OOI_DATA_DIR:-}" ]]; then
  echo "Set OOI_DATA_DIR to a local directory containing ooi-*.csv files."
  echo "Example: OOI_DATA_DIR=/path/to/ooi-data bash scripts/run_legacy_pipeline.sh"
  exit 1
fi

python "$ROOT_DIR/src/ocean_tp/analyze_ooi_data.py" "$OOI_DATA_DIR"
python "$ROOT_DIR/src/ocean_tp/compute_depth_layers.py" --data-dir "$OOI_DATA_DIR" --samples 60000
python "$ROOT_DIR/src/ocean_tp/pinn_tp_fit.py" --data-dir "$OOI_DATA_DIR" --samples 50000 --epochs 3000 --save-curve
