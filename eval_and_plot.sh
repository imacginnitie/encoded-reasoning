#!/bin/bash
# Evaluate results

set -e

EVAL_CONFIG_FILE="${1:-config/eval_config.yaml}"

echo "Evaluating with config: $EVAL_CONFIG_FILE"

uv run python -m encoded_reasoning.eval --config "$EVAL_CONFIG_FILE"

echo "Evaluation complete!"

