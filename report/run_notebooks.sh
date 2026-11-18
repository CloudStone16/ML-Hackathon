#!/bin/bash
# Execute the visualisation notebooks in place. Run from the repository root.
# Needs: results/ (training, tuning, evaluator_test outputs) and artifacts/ (features, evaluator predictions).
set -e
for nb in 01_eda 02_model_comparison 03_tuning_and_hmm 04_folium_maps; do
  uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 report/notebooks/$nb.ipynb
done
