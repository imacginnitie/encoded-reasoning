"""Plot experiment results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from matplotlib.patches import Patch

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _find_sibling_results(results_dir: Path, cipher_name: str):
    """Find results for a sibling cipher (same model/provider, different cipher).

    Works by navigating up from the cipher directory to the model directory,
    then looking for the sibling cipher. Supports both old and new directory structures:
      - results/<timestamp>/<provider>/<model>/<cipher>
      - results/<timestamp>/<dataset>/<provider>/<model>/<cipher>
    """
    try:
        # results_dir is the cipher directory; go up one level to the model directory
        model_dir = results_dir.parent
        sibling_dir = model_dir / cipher_name

        if not sibling_dir.exists():
            return None

        sibling_results_file = sibling_dir / "results.json"
        if sibling_results_file.exists():
            try:
                sibling_results = json.loads(sibling_results_file.read_text())
                sibling_metrics = sibling_results.get("metrics", {})
                if sibling_metrics:
                    return sibling_metrics
            except Exception:
                pass

        return None
    except Exception:
        return None


def _find_identity_results(results_dir: Path):
    """Find identity baseline results for the same model/provider."""
    return _find_sibling_results(results_dir, "identity")


def _find_direct_results(results_dir: Path):
    """Find direct cipher results for the same model/provider."""
    return _find_sibling_results(results_dir, "direct")


def _calculate_ratio_error(numerator, num_err, denominator, denom_err):
    """Calculate error propagation for ratio: num/denom.

    Uses the standard error propagation formula for division:
    σ_R = R * sqrt((σ_A/A)² + (σ_B/B)²)

    where R = A/B, σ_A is error in numerator, σ_B is error in denominator.

    Assumes numerator and denominator are independent (which is true in our case
    since they come from different experiments: cipher vs identity baseline).

    Args:
        numerator: The numerator value (e.g., adherent_and_correct_rate)
        num_err: Standard error of numerator
        denominator: The denominator value (e.g., identity_accuracy)
        denom_err: Standard error of denominator

    Returns:
        Standard error of the ratio
    """
    if denominator == 0:
        return 0.0
    if numerator == 0:
        return 0.0
    # Using standard error propagation for division
    # Formula: σ_R = R * sqrt((σ_A/A)² + (σ_B/B)²)
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


def _make_experiment_label(exp_dir: Path, cipher_type: str, filler_type: str | None) -> str:
    """Create a display label for an experiment from its directory structure."""
    # The experiment directory name is the cipher name (last component of the path)
    label = exp_dir.name

    if not label or label == ".":
        if cipher_type == "filler" and filler_type:
            return f"filler_{filler_type}"
        return f"{cipher_type}"

    if cipher_type == "filler" and filler_type and f"filler_{filler_type}" not in label:
        label = label.replace("filler", f"filler_{filler_type}")
    return label


def _load_matrix_experiments(
    base_dir: Path, skip_identity: bool = False, include_full_metrics: bool = False
) -> list[dict]:
    """Load all experiment results from a matrix results directory.

    Args:
        base_dir: Base directory containing experiment results.
        skip_identity: If True, skip identity cipher experiments.
        include_full_metrics: If True, include accuracy/adherence fields (for plot_matrix_results).

    Returns:
        List of experiment dicts with label, cipher, model, provider, ratio, ratio_err, n,
        and optionally accuracy/adherence fields.
    """
    results_files = list(base_dir.rglob("results.json"))
    if not results_files:
        return []

    experiments = []
    for results_file in sorted(results_files):
        try:
            results = json.loads(results_file.read_text())
            metrics = results.get("metrics", {})
            if not metrics:
                continue

            exp_dir = results_file.parent
            cipher_type = results.get("config", {}).get("cipher", {}).get("type", "unknown")
            model_name = results.get("config", {}).get("model", {}).get("name", "unknown")
            provider = results.get("config", {}).get("model", {}).get("provider", "unknown")

            if skip_identity and cipher_type.lower() == "identity":
                continue

            filler_config = results.get("config", {}).get("cipher", {}).get("filler", {})
            filler_type = filler_config.get("type") if filler_config else None

            label = _make_experiment_label(exp_dir, cipher_type, filler_type)

            identity_metrics = _find_identity_results(exp_dir)
            identity_accuracy = identity_metrics.get("accuracy", 0.0) if identity_metrics else 0.0
            identity_accuracy_err = (
                identity_metrics.get("accuracy_std_error", 0.0) if identity_metrics else 0.0
            )

            # Check if this experiment has strict/loose adherence data (emojispeak)
            has_strict_loose = "strict_and_correct_rate" in metrics

            if has_strict_loose:
                # Create two entries: one for strict, one for loose threshold
                for threshold_key, threshold_label in [
                    ("strict", "strict"),
                    ("loose", "loose"),
                ]:
                    adh_corr = metrics.get(f"{threshold_key}_and_correct_rate", 0.0)
                    adh_corr_err = metrics.get(f"{threshold_key}_and_correct_std_error", 0.0)
                    adh_rate = metrics.get(f"{threshold_key}_adherence_rate", 0.0)
                    adh_rate_err = metrics.get(f"{threshold_key}_adherence_std_error", 0.0)

                    ratio = adh_corr / identity_accuracy if identity_accuracy > 0 else 0.0
                    ratio_err = (
                        _calculate_ratio_error(
                            adh_corr, adh_corr_err, identity_accuracy, identity_accuracy_err
                        )
                        if identity_accuracy > 0
                        else 0.0
                    )

                    threshold_suffix = f" ({threshold_label})"
                    exp_entry = {
                        "label": label + threshold_suffix,
                        "cipher": cipher_type,
                        "model": model_name,
                        "provider": provider,
                        "ratio": ratio,
                        "ratio_err": ratio_err,
                        "n": metrics.get("n", 0),
                    }

                    if include_full_metrics:
                        exp_entry.update(
                            {
                                "accuracy": metrics.get("accuracy", 0.0),
                                "accuracy_err": metrics.get("accuracy_std_error", 0.0),
                                "adherence": adh_rate,
                                "adherence_err": adh_rate_err,
                                "adherent_and_correct": adh_corr,
                                "adherent_and_correct_err": adh_corr_err,
                            }
                        )

                    experiments.append(exp_entry)
            else:
                adherent_and_correct = metrics.get("adherent_and_correct_rate", 0.0)
                adherent_and_correct_err = metrics.get("adherent_and_correct_std_error", 0.0)

                ratio = (
                    adherent_and_correct / identity_accuracy if identity_accuracy > 0 else 0.0
                )
                ratio_err = (
                    _calculate_ratio_error(
                        adherent_and_correct,
                        adherent_and_correct_err,
                        identity_accuracy,
                        identity_accuracy_err,
                    )
                    if identity_accuracy > 0
                    else 0.0
                )

                exp = {
                    "label": label,
                    "cipher": cipher_type,
                    "model": model_name,
                    "provider": provider,
                    "ratio": ratio,
                    "ratio_err": ratio_err,
                    "n": metrics.get("n", 0),
                }

                if include_full_metrics:
                    exp.update(
                        {
                            "accuracy": metrics.get("accuracy", 0.0),
                            "accuracy_err": metrics.get("accuracy_std_error", 0.0),
                            "adherence": metrics.get("adherence_rate", 0.0),
                            "adherence_err": metrics.get("adherence_std_error", 0.0),
                            "adherent_and_correct": adherent_and_correct,
                            "adherent_and_correct_err": adherent_and_correct_err,
                        }
                    )

                experiments.append(exp)
        except Exception as e:
            print(f"Warning: Failed to load {results_file}: {e}")
            continue

    return experiments


def _find_direct_ratios(base_dir: Path, models: list[str]) -> dict[str, float]:
    """Find direct accuracy / identity accuracy ratio for each model.

    Returns:
        Dict mapping model name to direct/identity accuracy ratio.
    """
    model_direct_ratios: dict[str, float] = {}
    for model in models:
        for results_file in base_dir.rglob("results.json"):
            try:
                results = json.loads(results_file.read_text())
                exp_model = results.get("config", {}).get("model", {}).get("name", "")
                exp_cipher = results.get("config", {}).get("cipher", {}).get("type", "")
                if exp_model == model and exp_cipher.lower() != "identity":
                    exp_dir = results_file.parent
                    direct_metrics = _find_direct_results(exp_dir)
                    identity_metrics = _find_identity_results(exp_dir)
                    if direct_metrics and identity_metrics:
                        direct_accuracy = direct_metrics.get("accuracy", 0.0)
                        identity_accuracy = identity_metrics.get("accuracy", 0.0)
                        if identity_accuracy > 0:
                            model_direct_ratios[model] = direct_accuracy / identity_accuracy
                            break
            except Exception:
                continue
    return model_direct_ratios


def plot_matrix_results(base_dir: str | Path, config: dict | None = None):
    """Plot accuracy and adherence for a matrix of experiments on one graph.

    Args:
        base_dir: Base directory containing experiment results. Can be:
            - A timestamp directory (e.g., results/20260112_081513) containing
              subdirectories like <provider>/<model>/<cipher>/results.json
            - A directory containing multiple experiment directories with results.json
        config: Optional configuration dict (currently unused but kept for API consistency)
    """
    base_dir = Path(base_dir)

    experiments = _load_matrix_experiments(base_dir, include_full_metrics=True)

    if not experiments:
        print(f"Warning: No valid experiment results found in {base_dir}")
        return

    # Group experiments by label (which includes filler type) and model
    unique_models = sorted(set(exp["model"] for exp in experiments))
    # Use label instead of cipher for grouping, so filler types are separated
    unique_labels = sorted(set(exp["label"] for exp in experiments))

    # Create a mapping: label -> model -> experiment data
    # This ensures filler types are grouped separately
    label_model_map = {}
    for exp in experiments:
        label = exp["label"]
        model = exp["model"]
        if label not in label_model_map:
            label_model_map[label] = {}
        label_model_map[label][model] = exp

    # Determine if we have multiple models
    has_multiple_models = len(unique_models) > 1

    # Create the plot
    if has_multiple_models:
        # Grouped bar chart: groups are labels (including filler types),
        # bars within groups are models
        n_labels = len(unique_labels)
        n_models = len(unique_models)

        # Width calculations for grouped bars
        # Each label group has n_models * 3 bars (accuracy + adherence + ratio)
        group_width = 0.8  # Total width for each label group
        bar_width = group_width / (n_models * 3)  # Width of each individual bar

        fig, ax = plt.subplots(figsize=(max(12, n_labels * 2.5), 8))

        x_pos = np.arange(n_labels)

        # Colors for models (cycling through a palette)
        cmap = plt.cm.get_cmap("Set3")
        model_colors = [cmap(i / max(1, n_models - 1)) for i in range(n_models)]

        # Plot bars for each model
        for model_idx, model in enumerate(unique_models):
            model_accuracy = []
            model_accuracy_err = []
            model_adherence = []
            model_adherence_err = []
            model_ratio = []
            model_ratio_err = []

            for label in unique_labels:
                if label in label_model_map and model in label_model_map[label]:
                    exp = label_model_map[label][model]
                    model_accuracy.append(exp["accuracy"])
                    model_accuracy_err.append(exp["accuracy_err"])
                    model_adherence.append(exp["adherence"])
                    model_adherence_err.append(exp["adherence_err"])
                    model_ratio.append(exp["ratio"])
                    model_ratio_err.append(exp["ratio_err"])
                else:
                    # Missing data
                    model_accuracy.append(0)
                    model_accuracy_err.append(0)
                    model_adherence.append(0)
                    model_adherence_err.append(0)
                    model_ratio.append(0)
                    model_ratio_err.append(0)

            # Calculate positions for this model's bars
            # Within each cipher group: [model1_acc, model1_adh, model1_ratio, model2_acc, ...]
            offset = model_idx * 3 * bar_width

            # Plot ratio bars (Adherent & Correct / Identity Accuracy) - plain/solid
            ratio_positions = x_pos - group_width / 2 + offset + bar_width / 2
            ax.bar(
                ratio_positions,
                model_ratio,
                bar_width,
                yerr=model_ratio_err,
                label=f"{model} - Adh&Corr/IdAcc" if model_idx == 0 else "",
                color=model_colors[model_idx],
                alpha=0.8,
                capsize=3,
                edgecolor="black",
                linewidth=1,
            )

            # Plot accuracy bars - crosshatched
            acc_positions = x_pos - group_width / 2 + offset + bar_width * 1.5
            ax.bar(
                acc_positions,
                model_accuracy,
                bar_width,
                yerr=model_accuracy_err,
                label=f"{model} - Accuracy" if model_idx == 0 else "",
                color=model_colors[model_idx],
                alpha=0.7,
                capsize=3,
                edgecolor="black",
                linewidth=1,
                hatch="xx",
            )

            # Plot adherence bars - dots texture
            adh_positions = x_pos - group_width / 2 + offset + bar_width * 2.5
            ax.bar(
                adh_positions,
                model_adherence,
                bar_width,
                yerr=model_adherence_err,
                label=f"{model} - Adherence" if model_idx == 0 else "",
                color=model_colors[model_idx],
                alpha=0.7,
                capsize=3,
                edgecolor="black",
                linewidth=1,
                hatch="..",
            )

            # Add value labels
            for i, (acc, acc_err, adh, adh_err, ratio, ratio_err) in enumerate(
                zip(
                    model_accuracy,
                    model_accuracy_err,
                    model_adherence,
                    model_adherence_err,
                    model_ratio,
                    model_ratio_err,
                )
            ):
                if ratio > 0:  # Only label if there's data
                    ax.text(
                        ratio_positions[i],
                        ratio + ratio_err + 0.02,
                        f"{ratio:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                    )
                if acc > 0:
                    ax.text(
                        acc_positions[i],
                        acc + acc_err + 0.02,
                        f"{acc:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                    )
                if adh > 0:
                    ax.text(
                        adh_positions[i],
                        adh + adh_err + 0.02,
                        f"{adh:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                    )

        # Get n values for title
        n_values = [exp["n"] for exp in experiments]
        n_str = (
            f"n={n_values[0]}" if len(set(n_values)) == 1 else f"n={min(n_values)}-{max(n_values)}"
        )

        # Customize plot
        ax.set_xlabel("Experiment Type", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        title = f"Accuracy vs Adherence vs Ratio Across Experiments (Multiple Models, {n_str})"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        # Clean up labels for display (remove common prefixes, make readable)
        display_labels = []
        for label in unique_labels:
            # Clean up label for display
            if label.startswith("filler_"):
                display_label = label.replace("filler_", "").replace("_", " ").title()
            else:
                display_label = label.replace("_", " ").title()
            display_labels.append(display_label)
        ax.set_xticklabels(display_labels, rotation=45, ha="right")
        ax.set_ylim(0, 1.1)

        # Create custom legend
        legend_elements = []
        for model_idx, model in enumerate(unique_models):
            legend_elements.append(
                Patch(
                    facecolor=model_colors[model_idx],
                    alpha=0.8,
                    edgecolor="black",
                    label=f"{model} - Adh&Corr/IdAcc",
                )
            )
            legend_elements.append(
                Patch(
                    facecolor=model_colors[model_idx],
                    alpha=0.7,
                    edgecolor="black",
                    hatch="xx",
                    label=f"{model} - Accuracy",
                )
            )
            legend_elements.append(
                Patch(
                    facecolor=model_colors[model_idx],
                    alpha=0.7,
                    edgecolor="black",
                    hatch="..",
                    label=f"{model} - Adherence",
                )
            )
        ax.legend(
            handles=legend_elements, fontsize=9, loc="upper left",
            bbox_to_anchor=(1.02, 1), borderaxespad=0,
        )
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)

    else:
        # Single model: grouped bars (accuracy + adherence + ratio)
        fig, ax = plt.subplots(figsize=(max(12, len(experiments) * 0.9), 8))

        x_pos = np.arange(len(experiments))
        width = 0.25  # Width of bars (3 bars per experiment)

        # Plot bars - ratio first (plain), then accuracy (crosshatched), then adherence (dots)
        ax.bar(
            x_pos - width,
            [exp["ratio"] for exp in experiments],
            width,
            yerr=[exp["ratio_err"] for exp in experiments],
            label="Adh&Corr/IdAcc",
            color="mediumseagreen",
            alpha=0.8,
            capsize=5,
            edgecolor="black",
            linewidth=1.5,
        )

        ax.bar(
            x_pos,
            [exp["accuracy"] for exp in experiments],
            width,
            yerr=[exp["accuracy_err"] for exp in experiments],
            label="Accuracy (% Correct)",
            color="steelblue",
            alpha=0.7,
            capsize=5,
            edgecolor="black",
            linewidth=1.5,
            hatch="xx",
        )

        ax.bar(
            x_pos + width,
            [exp["adherence"] for exp in experiments],
            width,
            yerr=[exp["adherence_err"] for exp in experiments],
            label="Adherence Rate",
            color="coral",
            alpha=0.7,
            capsize=5,
            edgecolor="black",
            linewidth=1.5,
            hatch="..",
        )

        # Get n values for title
        n_values = [exp["n"] for exp in experiments]
        n_str = (
            f"n={n_values[0]}" if len(set(n_values)) == 1 else f"n={min(n_values)}-{max(n_values)}"
        )

        # Customize plot
        ax.set_xlabel("Experiment", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        title = f"Accuracy vs Adherence vs Ratio Across Experiments ({n_str})"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([exp["label"] for exp in experiments], rotation=45, ha="right")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)

        # Add value labels on bars
        for i, exp in enumerate(experiments):
            # Ratio label (first bar)
            if exp["ratio"] > 0:
                ax.text(
                    i - width,
                    exp["ratio"] + exp["ratio_err"] + 0.02,
                    f"{exp['ratio']:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
            # Accuracy label (second bar)
            ax.text(
                i,
                exp["accuracy"] + exp["accuracy_err"] + 0.02,
                f"{exp['accuracy']:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
            # Adherence label (third bar)
            ax.text(
                i + width,
                exp["adherence"] + exp["adherence_err"] + 0.02,
                f"{exp['adherence']:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()

    # Save plot
    output_path = base_dir / "matrix_plot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Matrix plot saved to {output_path}")
    plt.close()


def plot_matrix_results_per_model(base_dir: str | Path, config: dict | None = None):
    """Generate a separate matrix plot (accuracy + adherence + ratio) for each model.

    Each plot shows all cipher experiments for one model, with three bars per cipher:
    ratio, accuracy, and adherence.
    """
    base_dir = Path(base_dir)

    experiments = _load_matrix_experiments(base_dir, include_full_metrics=True)
    if not experiments:
        print(f"Warning: No valid experiment results found in {base_dir}")
        return

    unique_models = sorted(set(exp["model"] for exp in experiments))

    for model in unique_models:
        model_exps = [exp for exp in experiments if exp["model"] == model]
        if not model_exps:
            continue

        n_exps = len(model_exps)
        fig, ax = plt.subplots(figsize=(max(10, n_exps * 1.2), 7))

        x_pos = np.arange(n_exps)
        width = 0.25

        # Ratio bars (plain)
        ax.bar(
            x_pos - width,
            [exp["ratio"] for exp in model_exps],
            width,
            yerr=[exp["ratio_err"] for exp in model_exps],
            label="Adh&Corr / Identity Acc",
            color="mediumseagreen",
            alpha=0.8,
            capsize=5,
            edgecolor="black",
            linewidth=1.2,
        )

        # Accuracy bars (crosshatched)
        ax.bar(
            x_pos,
            [exp["accuracy"] for exp in model_exps],
            width,
            yerr=[exp["accuracy_err"] for exp in model_exps],
            label="Accuracy",
            color="steelblue",
            alpha=0.7,
            capsize=5,
            edgecolor="black",
            linewidth=1.2,
            hatch="xx",
        )

        # Adherence bars (dots)
        ax.bar(
            x_pos + width,
            [exp["adherence"] for exp in model_exps],
            width,
            yerr=[exp["adherence_err"] for exp in model_exps],
            label="Adherence",
            color="coral",
            alpha=0.7,
            capsize=5,
            edgecolor="black",
            linewidth=1.2,
            hatch="..",
        )

        # Value labels
        for i, exp in enumerate(model_exps):
            if exp["ratio"] > 0:
                ax.text(
                    i - width, exp["ratio"] + exp["ratio_err"] + 0.02,
                    f"{exp['ratio']:.2f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold",
                )
            ax.text(
                i, exp["accuracy"] + exp["accuracy_err"] + 0.02,
                f"{exp['accuracy']:.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold",
            )
            ax.text(
                i + width, exp["adherence"] + exp["adherence_err"] + 0.02,
                f"{exp['adherence']:.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold",
            )

        n_values = [exp["n"] for exp in model_exps]
        n_str = (
            f"n={n_values[0]}" if len(set(n_values)) == 1
            else f"n={min(n_values)}-{max(n_values)}"
        )

        ax.set_xlabel("Cipher", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title(f"{model} — {base_dir.name} ({n_str})", fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        display_labels = []
        for exp in model_exps:
            lbl = exp["label"]
            if lbl.startswith("filler_"):
                lbl = lbl.replace("filler_", "").replace("_", " ").title()
            else:
                lbl = lbl.replace("_", " ").title()
            display_labels.append(lbl)
        ax.set_xticklabels(display_labels, rotation=45, ha="right")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)

        plt.tight_layout()

        # Sanitize model name for filename
        safe_model = model.replace("/", "--").replace(" ", "_")
        output_path = base_dir / f"matrix_plot_{safe_model}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Per-model matrix plot saved to {output_path}")
        plt.close()


def plot_adherent_correct_by_model(base_dir: str | Path, config: dict | None = None):
    """Plot absolute adherent & correct rate for all ciphers, grouped by model, on one plot.

    X-axis: models. Bars within each group: identity, emojispeak (strict/loose), direct, filler.
    """
    base_dir = Path(base_dir)

    experiments = _load_matrix_experiments(base_dir, include_full_metrics=True)
    if not experiments:
        print(f"Warning: No valid experiment results found in {base_dir}")
        return

    unique_models = sorted(set(exp["model"] for exp in experiments))
    unique_labels = sorted(set(exp["label"] for exp in experiments))

    # model -> label -> exp
    model_label_map: dict[str, dict[str, dict]] = {}
    for exp in experiments:
        model_label_map.setdefault(exp["model"], {})[exp["label"]] = exp

    n_models = len(unique_models)
    n_labels = len(unique_labels)

    group_width = 0.8
    bar_width = group_width / n_labels

    fig, ax = plt.subplots(figsize=(max(14, n_models * 2.5), 7))
    x_pos = np.arange(n_models)

    # Use a qualitative palette
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3"]

    for label_idx, label in enumerate(unique_labels):
        values = []
        errors = []
        for model in unique_models:
            if model in model_label_map and label in model_label_map[model]:
                exp = model_label_map[model][label]
                values.append(exp["adherent_and_correct"])
                errors.append(exp["adherent_and_correct_err"])
            else:
                values.append(0)
                errors.append(0)

        positions = x_pos - group_width / 2 + label_idx * bar_width + bar_width / 2

        # Clean display label
        display_label = label
        if display_label.startswith("filler_"):
            display_label = display_label.replace("filler_", "filler: ").replace("_", " ")
        display_label = display_label.replace("_", " ")

        ax.bar(
            positions, values, bar_width,
            yerr=errors, label=display_label,
            color=colors[label_idx % len(colors)],
            alpha=0.85, capsize=4,
            edgecolor="black", linewidth=0.8,
        )

        # Value labels
        for i, (val, err) in enumerate(zip(values, errors)):
            if val > 0:
                ax.text(
                    positions[i], val + err + 0.01,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold",
                )

    n_values = [exp["n"] for exp in experiments]
    n_str = (
        f"n={n_values[0]}" if len(set(n_values)) == 1
        else f"n={min(n_values)}-{max(n_values)}"
    )

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Adherent & Correct Rate", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Adherent & Correct Rate by Model and Cipher — {base_dir.name} ({n_str})",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(unique_models, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(
        fontsize=9, title="Cipher", loc="upper left",
        bbox_to_anchor=(1.02, 1), borderaxespad=0,
    )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = base_dir / "adherent_correct_by_model.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Adherent & correct by model plot saved to {output_path}")
    plt.close()


def plot_matrix_results_ratio_only(base_dir: str | Path, config: dict | None = None):
    """Plot only the ratio (Adherent & Correct / Identity Accuracy) for all experiments.

    Args:
        base_dir: Base directory containing experiment results. Can be:
            - A timestamp directory (e.g., results/20260112_081513) containing
              subdirectories like <provider>/<model>/<cipher>/results.json
            - A directory containing multiple experiment directories with results.json
        config: Optional configuration dict (currently unused but kept for API consistency)
    """
    base_dir = Path(base_dir)

    experiments = _load_matrix_experiments(base_dir, skip_identity=True)

    if not experiments:
        print(f"Warning: No valid experiment results found in {base_dir}")
        return

    # Group experiments by label (which includes filler type) and model
    unique_models = sorted(set(exp["model"] for exp in experiments))
    # Use label instead of cipher for grouping, so filler types are separated
    unique_labels = sorted(set(exp["label"] for exp in experiments))

    # Create a mapping: label -> model -> experiment data
    # This ensures filler types are grouped separately
    label_model_map = {}
    for exp in experiments:
        label = exp["label"]
        model = exp["model"]
        if label not in label_model_map:
            label_model_map[label] = {}
        label_model_map[label][model] = exp

    # Determine if we have multiple models
    has_multiple_models = len(unique_models) > 1

    # Create the plot
    if has_multiple_models:
        # Grouped bar chart: groups are labels (including filler types),
        # bars within groups are models
        n_labels = len(unique_labels)
        n_models = len(unique_models)

        # Width calculations for grouped bars
        group_width = 0.8  # Total width for each label group
        bar_width = group_width / n_models  # Width of each individual bar

        fig, ax = plt.subplots(figsize=(max(12, n_labels * 1.5), 8))

        x_pos = np.arange(n_labels)

        # Colors for models (cycling through a palette)
        cmap = plt.cm.get_cmap("Set3")
        model_colors = [cmap(i / max(1, n_models - 1)) for i in range(n_models)]

        # Plot bars for each model
        for model_idx, model in enumerate(unique_models):
            model_ratio = []
            model_ratio_err = []

            for label in unique_labels:
                if label in label_model_map and model in label_model_map[label]:
                    exp = label_model_map[label][model]
                    model_ratio.append(exp["ratio"])
                    model_ratio_err.append(exp["ratio_err"])
                else:
                    # Missing data
                    model_ratio.append(0)
                    model_ratio_err.append(0)

            # Calculate positions for this model's bars
            offset = model_idx * bar_width

            # Plot ratio bars (Adherent & Correct / Identity Accuracy) - plain/solid
            ratio_positions = x_pos - group_width / 2 + offset + bar_width / 2
            ax.bar(
                ratio_positions,
                model_ratio,
                bar_width,
                yerr=model_ratio_err,
                label=model,
                color=model_colors[model_idx],
                alpha=0.8,
                capsize=5,
                edgecolor="black",
                linewidth=1.5,
            )

            # Add value labels
            for i, (ratio, ratio_err) in enumerate(zip(model_ratio, model_ratio_err)):
                if ratio > 0:  # Only label if there's data
                    ax.text(
                        ratio_positions[i],
                        ratio + ratio_err + 0.02,
                        f"{ratio:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )

        # Get n values for title
        n_values = [exp["n"] for exp in experiments]
        n_str = (
            f"n={n_values[0]}" if len(set(n_values)) == 1 else f"n={min(n_values)}-{max(n_values)}"
        )

        # Customize plot
        ax.set_xlabel("Experiment Type", fontsize=12, fontweight="bold")
        ax.set_ylabel("Adh&Corr/IdAcc Ratio", fontsize=12, fontweight="bold")
        title = f"Adherent & Correct / Identity Accuracy Ratio Across Experiments ({n_str})"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        # Clean up labels for display (remove common prefixes, make readable)
        display_labels = []
        for label in unique_labels:
            # Clean up label for display
            if label.startswith("filler_"):
                display_label = label.replace("filler_", "").replace("_", " ").title()
            else:
                display_label = label.replace("_", " ").title()
            display_labels.append(display_label)
        ax.set_xticklabels(display_labels, rotation=45, ha="right")
        max_ratio = max((exp["ratio"] + exp["ratio_err"] for exp in experiments), default=1.1)
        ax.set_ylim(0, max(1.1, max_ratio))

        model_direct_ratios = _find_direct_ratios(base_dir, unique_models)

        # Plot baseline line first
        ax.axhline(
            y=1.0,
            color="green",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="Baseline (1.0)",
        )

        # Plot direct accuracy ratio lines for each model
        direct_line_colors = ["orange", "purple", "brown", "pink", "cyan", "magenta"]
        for model_idx, model in enumerate(unique_models):
            if model in model_direct_ratios:
                direct_ratio = model_direct_ratios[model]
                line_color = direct_line_colors[model_idx % len(direct_line_colors)]
                ax.axhline(
                    y=direct_ratio,
                    color=line_color,
                    linestyle=":",
                    alpha=0.9,
                    linewidth=2.5,
                    label=f"{model} - Direct Acc",
                )

        ax.legend(
            fontsize=9, title="Model", loc="upper left",
            bbox_to_anchor=(1.02, 1), borderaxespad=0,
        )
        ax.grid(True, alpha=0.3, axis="y")

    else:
        # Single model: simple bars
        fig, ax = plt.subplots(figsize=(max(12, len(experiments) * 0.6), 8))

        x_pos = np.arange(len(experiments))
        width = 0.6  # Width of bars

        # Plot bars
        ax.bar(
            x_pos,
            [exp["ratio"] for exp in experiments],
            width,
            yerr=[exp["ratio_err"] for exp in experiments],
            label="Adh&Corr/IdAcc",
            color="mediumseagreen",
            alpha=0.8,
            capsize=5,
            edgecolor="black",
            linewidth=1.5,
        )

        # Get n values for title
        n_values = [exp["n"] for exp in experiments]
        n_str = (
            f"n={n_values[0]}" if len(set(n_values)) == 1 else f"n={min(n_values)}-{max(n_values)}"
        )

        # Customize plot
        ax.set_xlabel("Experiment", fontsize=12, fontweight="bold")
        ax.set_ylabel("Adh&Corr/IdAcc Ratio", fontsize=12, fontweight="bold")
        title = f"Adherent & Correct / Identity Accuracy Ratio Across Experiments ({n_str})"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([exp["label"] for exp in experiments], rotation=45, ha="right")
        max_ratio = max((exp["ratio"] + exp["ratio_err"] for exp in experiments), default=1.1)
        ax.set_ylim(0, max(1.1, max_ratio))

        # Plot baseline line first
        ax.axhline(
            y=1.0,
            color="green",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="Baseline (1.0)",
        )

        # Find and plot direct accuracy ratio for the single model
        single_model_direct = _find_direct_ratios(base_dir, [experiments[0]["model"]])
        direct_ratio = single_model_direct.get(experiments[0]["model"])

        if direct_ratio is not None:
            ax.axhline(
                y=direct_ratio,
                color="orange",
                linestyle=":",
                alpha=0.9,
                linewidth=2.5,
                label="Direct Acc",
            )

        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for i, exp in enumerate(experiments):
            if exp["ratio"] > 0:
                ax.text(
                    i,
                    exp["ratio"] + exp["ratio_err"] + 0.02,
                    f"{exp['ratio']:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.tight_layout()

    # Save plot
    output_path = base_dir / "matrix_plot_ratio_only.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Ratio-only matrix plot saved to {output_path}")
    plt.close()


def plot_matrix_results_ratio_only_by_model(base_dir: str | Path, config: dict | None = None):
    """Plot only the ratio (Adherent & Correct / Identity Accuracy) grouped by model.

    Bars are grouped by model on the x-axis, with different ciphers as bars within each group.

    Args:
        base_dir: Base directory containing experiment results. Can be:
            - A timestamp directory (e.g., results/20260112_081513) containing
              subdirectories like <provider>/<model>/<cipher>/results.json
            - A directory containing multiple experiment directories with results.json
        config: Optional configuration dict (currently unused but kept for API consistency)
    """
    base_dir = Path(base_dir)

    experiments = _load_matrix_experiments(base_dir, skip_identity=True)

    if not experiments:
        print(f"Warning: No valid experiment results found in {base_dir}")
        return

    # Group experiments by model and label (cipher)
    unique_models = sorted(set(exp["model"] for exp in experiments))
    unique_labels = sorted(set(exp["label"] for exp in experiments))

    # Create a mapping: model -> label -> experiment data
    model_label_map = {}
    for exp in experiments:
        model = exp["model"]
        label = exp["label"]
        if model not in model_label_map:
            model_label_map[model] = {}
        model_label_map[model][label] = exp

    # Determine if we have multiple models
    has_multiple_models = len(unique_models) > 1
    has_multiple_labels = len(unique_labels) > 1

    # Create the plot
    if has_multiple_models or has_multiple_labels:
        # Grouped bar chart: groups are models, bars within groups are labels (ciphers)
        n_models = len(unique_models)
        n_labels = len(unique_labels)

        # Width calculations for grouped bars
        group_width = 0.8  # Total width for each model group
        bar_width = group_width / n_labels  # Width of each individual bar

        fig, ax = plt.subplots(figsize=(max(12, n_models * 1.5), 8))

        x_pos = np.arange(n_models)

        # Colors for labels/ciphers (cycling through a palette)
        cmap = plt.cm.get_cmap("Set3")
        label_colors = [cmap(i / max(1, n_labels - 1)) for i in range(n_labels)]

        # Plot bars for each label (cipher)
        for label_idx, label in enumerate(unique_labels):
            label_ratio = []
            label_ratio_err = []

            for model in unique_models:
                if model in model_label_map and label in model_label_map[model]:
                    exp = model_label_map[model][label]
                    label_ratio.append(exp["ratio"])
                    label_ratio_err.append(exp["ratio_err"])
                else:
                    # Missing data
                    label_ratio.append(0)
                    label_ratio_err.append(0)

            # Calculate positions for this label's bars
            offset = label_idx * bar_width

            # Plot ratio bars (Adherent & Correct / Identity Accuracy)
            ratio_positions = x_pos - group_width / 2 + offset + bar_width / 2
            # Format label for display
            if label.startswith("filler_"):
                display_label = label.replace("filler_", "").replace("_", " ").title()
            else:
                display_label = label.replace("_", " ").title()
            ax.bar(
                ratio_positions,
                label_ratio,
                bar_width,
                yerr=label_ratio_err,
                label=display_label,
                color=label_colors[label_idx],
                alpha=0.8,
                capsize=5,
                edgecolor="black",
                linewidth=1.5,
            )

            # Add value labels
            for i, (ratio, ratio_err) in enumerate(zip(label_ratio, label_ratio_err)):
                if ratio > 0:  # Only label if there's data
                    ax.text(
                        ratio_positions[i],
                        ratio + ratio_err + 0.02,
                        f"{ratio:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

        # Get n values for title
        n_values = [exp["n"] for exp in experiments]
        n_str = (
            f"n={n_values[0]}" if len(set(n_values)) == 1 else f"n={min(n_values)}-{max(n_values)}"
        )

        # Customize plot
        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Adh&Corr/IdAcc Ratio", fontsize=12, fontweight="bold")
        title = f"Adherent & Correct / Identity Accuracy Ratio by Model ({n_str})"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(unique_models, rotation=45, ha="right")
        max_ratio = max((exp["ratio"] + exp["ratio_err"] for exp in experiments), default=1.1)
        ax.set_ylim(0, max(1.1, max_ratio))

        # Plot baseline line
        ax.axhline(
            y=1.0,
            color="green",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="Baseline (1.0)",
        )

        ax.legend(
            fontsize=9, title="Cipher Type", loc="upper left",
            bbox_to_anchor=(1.02, 1), borderaxespad=0,
        )
        ax.grid(True, alpha=0.3, axis="y")

    else:
        # Single model and single label: simple bar
        fig, ax = plt.subplots(figsize=(8, 6))

        x_pos = np.arange(1)
        width = 0.6

        exp = experiments[0]
        ax.bar(
            x_pos,
            [exp["ratio"]],
            width,
            yerr=[exp["ratio_err"]],
            label="Adh&Corr/IdAcc",
            color="mediumseagreen",
            alpha=0.8,
            capsize=5,
            edgecolor="black",
            linewidth=1.5,
        )

        n_str = f"n={exp['n']}"
        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Adh&Corr/IdAcc Ratio", fontsize=12, fontweight="bold")
        title = f"Adherent & Correct / Identity Accuracy Ratio by Model ({n_str})"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([exp["model"]], rotation=45, ha="right")
        max_ratio = exp["ratio"] + exp["ratio_err"]
        ax.set_ylim(0, max(1.1, max_ratio))

        ax.axhline(
            y=1.0,
            color="green",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="Baseline (1.0)",
        )

        single_model_direct = _find_direct_ratios(base_dir, [exp["model"]])
        direct_ratio = single_model_direct.get(exp["model"])

        if direct_ratio is not None:
            ax.axhline(
                y=direct_ratio,
                color="orange",
                linestyle=":",
                alpha=0.9,
                linewidth=2.5,
                label="Direct Acc",
            )

        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        if exp["ratio"] > 0:
            ax.text(
                0,
                exp["ratio"] + exp["ratio_err"] + 0.02,
                f"{exp['ratio']:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    plt.tight_layout()

    # Save plot
    output_path = base_dir / "matrix_plot_ratio_only_by_model.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Ratio-only matrix plot (grouped by model) saved to {output_path}")
    plt.close()


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
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Plot matrix of experiments (accuracy and adherence for all experiments in directory)",
    )
    parser.add_argument(
        "--matrix-ratio-only",
        action="store_true",
        help="Plot matrix of experiments showing only the ratio (Adh&Corr/IdAcc)",
    )
    parser.add_argument(
        "--matrix-ratio-only-by-model",
        action="store_true",
        help="Plot matrix of experiments showing only the ratio (Adh&Corr/IdAcc), grouped by model",
    )
    parser.add_argument(
        "--matrix-per-model",
        action="store_true",
        help="Generate a separate full matrix plot (accuracy + adherence + ratio) for each model",
    )
    parser.add_argument(
        "--adherent-correct-by-model",
        action="store_true",
        help="Plot absolute adherent & correct rate for all ciphers grouped by model",
    )
    args = parser.parse_args()

    config = {}
    if args.config:
        config = load_config(args.config)

    if args.adherent_correct_by_model:
        plot_adherent_correct_by_model(args.results_dir, config)
    elif args.matrix_per_model:
        plot_matrix_results_per_model(args.results_dir, config)
    elif args.matrix_ratio_only_by_model:
        plot_matrix_results_ratio_only_by_model(args.results_dir, config)
    elif args.matrix_ratio_only:
        plot_matrix_results_ratio_only(args.results_dir, config)
    elif args.matrix:
        plot_matrix_results(args.results_dir, config)
    else:
        plot_results(args.results_dir, config)


if __name__ == "__main__":
    main()
