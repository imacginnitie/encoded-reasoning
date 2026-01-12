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


def _find_identity_results(results_dir: Path):
    """Find identity baseline results for the same model/provider.

    Directory structure: results/<timestamp>/<provider>/<model>/<cipher>
    """
    parts = results_dir.parts
    if len(parts) < 5:
        return None

    try:
        # Find the 'results' directory index
        results_idx = None
        for i, part in enumerate(parts):
            if part == "results":
                results_idx = i
                break

        if results_idx is None or results_idx + 4 >= len(parts):
            return None

        # Structure: results/<timestamp>/<provider>/<model>/<cipher>
        timestamp = parts[results_idx + 1]
        provider = parts[results_idx + 2]
        model = parts[results_idx + 3]

        # Look for identity in same timestamp directory
        identity_base = Path(*parts[: results_idx + 1]) / timestamp / provider / model / "identity"

        if not identity_base.exists():
            return None

        identity_results_file = identity_base / "results.json"
        if identity_results_file.exists():
            try:
                identity_results = json.loads(identity_results_file.read_text())
                identity_metrics = identity_results.get("metrics", {})
                if identity_metrics:
                    return identity_metrics
            except Exception:
                pass

        return None
    except Exception:
        return None


def _calculate_ratio_error(numerator, num_err, denominator, denom_err):
    """Calculate error propagation for ratio: num/denom."""
    if denominator == 0:
        return 0.0
    # Using standard error propagation for division
    ratio = numerator / denominator
    relative_err = ((num_err / numerator) ** 2 + (denom_err / denominator) ** 2) ** 0.5
    return ratio * relative_err


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
    adherent_and_correct = metrics.get("adherent_and_correct_rate", 0.0)
    adherent_and_correct_err = metrics.get("adherent_and_correct_std_error", 0.0)
    n = metrics.get("n", 0)

    # Side-by-side plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    _plot_bar(ax1, accuracy, accuracy_err, "Accuracy", "steelblue", n)
    _plot_bar(ax2, adherence, adherence_err, "Adherence Rate", "coral", n)
    _plot_bar(
        ax3,
        adherent_and_correct,
        adherent_and_correct_err,
        "Adherent & Correct",
        "mediumseagreen",
        n,
    )

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
    fig2, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(3)
    ax.bar(
        x_pos,
        [accuracy, adherence, adherent_and_correct],
        yerr=[accuracy_err, adherence_err, adherent_and_correct_err],
        capsize=10,
        color=["steelblue", "coral", "mediumseagreen"],
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        ["Accuracy", "Adherence", "Adherent & Correct"],
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_title(
        f"Accuracy vs Adherence vs Adherent & Correct (n={n})",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)
    for i, (val, err) in enumerate(
        zip(
            [accuracy, adherence, adherent_and_correct],
            [accuracy_err, adherence_err, adherent_and_correct_err],
        )
    ):
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

    # Ratio plot: ciphered / identity (only if identity results exist and current is not identity)
    cipher_type = results.get("config", {}).get("cipher", {}).get("type", "unknown")
    if cipher_type.lower() != "identity":
        identity_metrics = _find_identity_results(results_dir)
        if identity_metrics:
            identity_accuracy = identity_metrics.get("accuracy", 0.0)
            identity_accuracy_err = identity_metrics.get("accuracy_std_error", 0.0)
            identity_adherent_and_correct = identity_metrics.get("adherent_and_correct_rate", 0.0)
            identity_adherent_and_correct_err = identity_metrics.get(
                "adherent_and_correct_std_error", 0.0
            )

            # Calculate ratios
            accuracy_ratio = accuracy / identity_accuracy if identity_accuracy > 0 else 0.0
            accuracy_ratio_err = _calculate_ratio_error(
                accuracy, accuracy_err, identity_accuracy, identity_accuracy_err
            )

            adherent_and_correct_ratio = (
                adherent_and_correct / identity_adherent_and_correct
                if identity_adherent_and_correct > 0
                else 0.0
            )
            adherent_and_correct_ratio_err = _calculate_ratio_error(
                adherent_and_correct,
                adherent_and_correct_err,
                identity_adherent_and_correct,
                identity_adherent_and_correct_err,
            )

            # Create ratio plot
            fig3, ax = plt.subplots(figsize=(10, 6))
            x_pos = np.arange(2)
            ratios = [accuracy_ratio, adherent_and_correct_ratio]
            ratio_errs = [accuracy_ratio_err, adherent_and_correct_ratio_err]

            colors = ["steelblue", "mediumseagreen"]
            ax.bar(
                x_pos,
                ratios,
                yerr=ratio_errs,
                capsize=10,
                color=colors,
                alpha=0.7,
                edgecolor="black",
                linewidth=1.5,
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                ["Accuracy Ratio", "Adherent & Correct Ratio"],
                fontsize=12,
                fontweight="bold",
            )
            ax.set_ylabel("Ratio (Ciphered / Identity)", fontsize=12, fontweight="bold")
            ax.set_title(
                f"Performance Ratio vs Identity Baseline (n={n})",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3, axis="y")
            ax.axhline(
                y=1.0,
                color="red",
                linestyle="--",
                alpha=0.7,
                linewidth=2,
                label="Baseline (1.0)",
            )
            ax.legend()

            for i, (val, err) in enumerate(zip(ratios, ratio_errs)):
                ax.text(
                    i,
                    val + err + 0.05,
                    f"{val:.3f} ± {err:.3f}",
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                )

            model_name = results.get("config", {}).get("model", {}).get("name", "unknown")
            fig3.suptitle(
                f"Ratio Plot: {cipher_type} / Identity | Model: {model_name}",
                fontsize=14,
                fontweight="bold",
                y=0.98,
            )

            plt.tight_layout()
            plt.savefig(results_dir / "ratio_plot.png", dpi=300, bbox_inches="tight")
            print(f"✓ Ratio plot saved to {results_dir / 'ratio_plot.png'}")
            plt.close("all")
        else:
            print("ℹ No identity baseline found - skipping ratio plot")


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
