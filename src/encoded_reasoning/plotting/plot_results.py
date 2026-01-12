"""Plot experiment results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _plot_bar(ax, value, error, label, color, n):
    """Helper to plot a single bar with error bars."""
    ax.bar(
        [label],
        [value],
        yerr=[error],
        capsize=10,
        color=color,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_ylim([0, 1.1])
    ax.set_ylabel(label, fontsize=12, fontweight="bold")
    ax.set_title(f"{label} (n={n})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)
    ax.text(
        0,
        value + error + 0.05,
        f"{value:.3f} ± {error:.3f}",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )


def plot_results(results_dir: str | Path, config: dict | None = None):
    """Generate plots from experiment results."""
    results_dir = Path(results_dir)
    results = json.loads((results_dir / "results.json").read_text())

    metrics = results.get("metrics", {})
    if not metrics:
        print(f"Warning: No metrics found in {results_dir / 'results.json'}")
        return

    accuracy = metrics.get("accuracy", 0.0)
    accuracy_err = metrics.get("accuracy_std_error", 0.0)
    adherence = metrics.get("adherence_rate", 0.0)
    adherence_err = metrics.get("adherence_std_error", 0.0)
    n = metrics.get("n", 0)

    # Side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _plot_bar(ax1, accuracy, accuracy_err, "Accuracy", "steelblue", n)
    _plot_bar(ax2, adherence, adherence_err, "Adherence Rate", "coral", n)

    cipher_type = results.get("config", {}).get("cipher", {}).get("type", "unknown")
    model_name = results.get("config", {}).get("model", {}).get("name", "unknown")
    fig.suptitle(
        f"Experiment Results: {cipher_type} | Model: {model_name}",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    (results_dir / "results_plot.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "results_plot.png", dpi=300, bbox_inches="tight")
    print(f"✓ Plot saved to {results_dir / 'results_plot.png'}")

    # Combined plot
    fig2, ax = plt.subplots(figsize=(8, 6))
    x_pos = np.arange(2)
    ax.bar(
        x_pos,
        [accuracy, adherence],
        yerr=[accuracy_err, adherence_err],
        capsize=10,
        color=["steelblue", "coral"],
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Accuracy", "Adherence"], fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Accuracy vs Adherence (n={n})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)
    for i, (val, err) in enumerate(zip([accuracy, adherence], [accuracy_err, adherence_err])):
        ax.text(
            i,
            val + err + 0.05,
            f"{val:.3f} ± {err:.3f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(results_dir / "combined_plot.png", dpi=300, bbox_inches="tight")
    print(f"✓ Combined plot saved to {results_dir / 'combined_plot.png'}")
    plt.close("all")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to results directory (e.g., results/20260112_081513)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to evaluation config file (for additional settings)",
    )
    args = parser.parse_args()

    config = {}
    if args.config:
        config = load_config(args.config)

    plot_results(args.results_dir, config)


if __name__ == "__main__":
    main()
