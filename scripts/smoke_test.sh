#!/bin/bash
# Smoke test: verifies all CLI commands are registered and dataset is valid.
set -e

echo "── Smoke Test ──"

echo "[1/4] CLI help..."
python -m ms_ai.cli --help

echo "[2/4] check-data..."
python -m ms_ai.cli check-data

echo "[3/4] train-seg --help..."
python -m ms_ai.cli train-seg --help

echo "[4/4] infer-seg --help..."
python -m ms_ai.cli infer-seg --help

echo ""
echo "Smoke test passed ✅"
