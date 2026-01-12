#!/bin/bash
# Evaluate and plot results

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <results_dir> [config_file]"
    echo "Example: $0 results/20260112_081513"
    echo "         $0 results/20260112_092547  (processes all experiments in timestamp directory)"
    exit 1
fi

RESULTS_DIR="$1"
EVAL_CONFIG_FILE="${2:-}"

# Check if this is a timestamp directory (contains provider subdirectories)
# Structure: results/<timestamp>/<provider>/<model>/<cipher>
if [ -d "$RESULTS_DIR" ] && [ -n "$(find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d)" ]; then
    # Check if first subdirectory looks like a provider (not a direct results.json)
    FIRST_SUBDIR=$(find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d | head -1)
    if [ -n "$FIRST_SUBDIR" ] && [ ! -f "$RESULTS_DIR/results.json" ]; then
        # This is a timestamp directory - process all experiments
        echo "Detected timestamp directory. Processing all experiments..."
        echo ""
        
        # Find all experiment directories (those containing results.json)
        ALL_EXPERIMENT_DIRS=$(find "$RESULTS_DIR" -name "results.json" -type f | sed 's|/results.json||' | sort)
        
        if [ -z "$ALL_EXPERIMENT_DIRS" ]; then
            echo "No experiment results found in $RESULTS_DIR"
            exit 1
        fi
        
        # Separate identity experiments to process first (for ratio plots)
        IDENTITY_DIRS=$(echo "$ALL_EXPERIMENT_DIRS" | grep '/identity$' || true)
        OTHER_DIRS=$(echo "$ALL_EXPERIMENT_DIRS" | grep -v '/identity$' || true)
        
        # Combine: identity first, then others
        if [ -n "$IDENTITY_DIRS" ] && [ -n "$OTHER_DIRS" ]; then
            EXPERIMENT_DIRS="$IDENTITY_DIRS"$'\n'"$OTHER_DIRS"
        elif [ -n "$IDENTITY_DIRS" ]; then
            EXPERIMENT_DIRS="$IDENTITY_DIRS"
        else
            EXPERIMENT_DIRS="$OTHER_DIRS"
        fi
        
        COUNT=$(echo "$EXPERIMENT_DIRS" | grep -v '^$' | wc -l | tr -d ' ')
        echo "Found $COUNT experiment(s) to process"
        if [ -n "$IDENTITY_DIRS" ]; then
            IDENTITY_COUNT=$(echo "$IDENTITY_DIRS" | grep -v '^$' | wc -l | tr -d ' ')
            echo "  - $IDENTITY_COUNT identity baseline(s) (will be processed first)"
        fi
        echo ""
        
        for EXP_DIR in $EXPERIMENT_DIRS; do
            [ -z "$EXP_DIR" ] && continue
            echo "=========================================="
            echo "Processing: $EXP_DIR"
            echo "=========================================="
            
            if [ -n "$EVAL_CONFIG_FILE" ]; then
                uv run python -m encoded_reasoning.eval --results-dir "$EXP_DIR" --config "$EVAL_CONFIG_FILE"
            else
                uv run python -m encoded_reasoning.eval --results-dir "$EXP_DIR"
            fi
            
            echo ""
            echo "Plotting results..."
            
            if [ -n "$EVAL_CONFIG_FILE" ]; then
                uv run python -m encoded_reasoning.plotting.plot_results --results-dir "$EXP_DIR" --config "$EVAL_CONFIG_FILE"
            else
                uv run python -m encoded_reasoning.plotting.plot_results --results-dir "$EXP_DIR"
            fi
            
            echo ""
        done
        
        echo "=========================================="
        echo "All experiments processed!"
        echo "=========================================="
        exit 0
    fi
fi

# Single experiment directory
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

