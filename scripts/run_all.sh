#!/usr/bin/env bash
# Reproduce every result and figure: download -> tests -> E1 -> E2a -> E2b -> E3 -> figures.
# Ensembles whose checkpoints already exist in checkpoints/ are loaded instead of retrained.
# Usage: DEVICE=cuda:0 bash scripts/run_all.sh   (default cuda:1)
set -euo pipefail

cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda:1}"
PYTHON="${PYTHON:-.venv/bin/python}"

mkdir -p results/preds
bash scripts/download_data.sh
"$PYTHON" -m pytest -q
"$PYTHON" scripts/analyse_missingness.py
"$PYTHON" scripts/run_e1.py --device "$DEVICE"
"$PYTHON" scripts/run_e2a.py --device "$DEVICE"
"$PYTHON" scripts/run_e2b.py --device "$DEVICE"
"$PYTHON" scripts/run_e3.py
"$PYTHON" scripts/make_figures.py
