#!/usr/bin/env bash
set -e

# project root assumed
dirs=(
  data/raw data/processed
  notebooks
  src/data src/analysis src/models src/utils
  results/figures results/tables
  docs/assets
)

files=(
  LICENSE environment.yml requirements.txt
  notebooks/01_data_ingestion.ipynb
  notebooks/02_event_study.ipynb
  notebooks/03_model_training.ipynb
  notebooks/04_visualization.ipynb
  src/data/fetch.py src/data/preprocess.py
  src/analysis/event_study.py src/analysis/metrics.py
  src/models/train.py src/models/predict.py
  src/utils/helpers.py
  docs/report.md
)

# create directories
for d in "${dirs[@]}"; do
  mkdir -p "$d"
done

# create empty files (or open your editor to fill them)
for f in "${files[@]}"; do
  dir=$(dirname "$f")
  mkdir -p "$dir"
  [[ -f "$f" ]] || touch "$f"
done

echo "Project scaffold created."
