#!/bin/bash
# Run an experiment with the specified config

set -e

CONFIG_FILE="${1:-config.yaml}"

echo "Running experiment with config: $CONFIG_FILE"

# Process dataset (create encoded examples)
uv run python -m encoded_reasoning.process_dataset --config "$CONFIG_FILE"

# Run experiment
uv run python -m encoded_reasoning.run_experiment --config "$CONFIG_FILE"

echo "Experiment complete!"

