#!/usr/bin/env bash
# Full evaluation pipeline on updated case_node_matrix.csv (run from repo root).
# Requires: pgmpy, pandas, numpy, scikit-learn, matplotlib
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== 1) Train-test split, refit BN, test-set figures ==="
python code/train_test_split_evaluation.py

echo "=== 2–3) Full corpus + scenario rows -> scenario_results_full.csv ==="
python code/run_inference_scenarios.py \
  --bif "$ROOT/outputs/homicide_bn_train.bif" \
  --complete-scenarios-full

echo "=== 4) Consistency (Jaccard 0.60 + 0.50) ==="
python code/evaluate_bn_consistency.py

echo "=== 5) Calibration figures ==="
python code/evaluate_bn_calibration.py

echo "=== 6) Test-set figures (step 1 already wrote fig_test_*) ==="

echo "=== 7) Regenerate paper figures ==="
python code/generate_results_figures.py

echo "=== 8) Before/after comparison ==="
python code/pipeline_rerun_comparison.py

echo "Done."
