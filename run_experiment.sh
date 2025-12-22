#!/bin/bash
# Run an experiment with the specified config

set -e

CONFIG_FILE="${1:-config.yaml}"

echo "Running experiment with config: $CONFIG_FILE"

uv run python -m encoded_reasoning.run_experiment --config "$CONFIG_FILE"

echo "Experiment complete!"

