#!/bin/bash
# Evaluate results and generate plots

set -e

EVAL_CONFIG_FILE="${1:-config/eval_config.yaml}"

echo "Evaluating and plotting with config: $EVAL_CONFIG_FILE"

# Run evaluation
uv run python -m encoded_reasoning.eval --config "$EVAL_CONFIG_FILE"

# Generate plots
uv run python -m encoded_reasoning.plotting.plot_results --config "$EVAL_CONFIG_FILE"

echo "Evaluation and plotting complete!"

