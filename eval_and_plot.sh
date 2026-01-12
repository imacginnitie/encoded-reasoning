#!/bin/bash
# Evaluate and plot results

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <results_dir> [config_file]"
    echo "Example: $0 results/20260112_081513"
    exit 1
fi

RESULTS_DIR="$1"
EVAL_CONFIG_FILE="${2:-}"

echo "Evaluating results in: $RESULTS_DIR"

if [ -n "$EVAL_CONFIG_FILE" ]; then
    uv run python -m encoded_reasoning.eval --results-dir "$RESULTS_DIR" --config "$EVAL_CONFIG_FILE"
else
    uv run python -m encoded_reasoning.eval --results-dir "$RESULTS_DIR"
fi

echo ""
echo "Plotting results..."

if [ -n "$EVAL_CONFIG_FILE" ]; then
    uv run python -m encoded_reasoning.plotting.plot_results --results-dir "$RESULTS_DIR" --config "$EVAL_CONFIG_FILE"
else
    uv run python -m encoded_reasoning.plotting.plot_results --results-dir "$RESULTS_DIR"
fi

echo "Evaluation and plotting complete!"

