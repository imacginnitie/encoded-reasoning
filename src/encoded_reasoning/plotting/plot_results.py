"""Plot experiment results."""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def plot_results(config: dict[str, Any]) -> None:
    """Generate plots from experiment results."""
    results_dir = Path(config["results_dir"])

    # Load results
    results_path = results_dir / "results.json"
    with open(results_path) as f:
        results = json.load(f)

    # TODO: Implement plotting
    # - Load matplotlib/seaborn
    # - Create plots with error bars
    # - Save to results_dir

    print(f"Plotting results from {results_path}")
    print("TODO: Implement actual plotting code")
    print("Plots should include error bars!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument(
        "--config",
        type=str,
        default="config/eval_config.yaml",
        help="Path to eval config YAML file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    plot_results(config)


if __name__ == "__main__":
    main()
