"""Evaluate experiment results."""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from encoded_reasoning.adherence import evaluate_adherence
from encoded_reasoning.dataset import extract_answer_from_boxed
from encoded_reasoning.encoding import get_encoding


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def calculate_accuracy(predictions: list[str], ground_truth: list[str]) -> float:
    """Calculate accuracy of predictions."""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
    return correct / len(predictions)


def calculate_accuracy_with_error_bars(
    predictions: list[str],
    ground_truth: list[str],
    num_bootstrap: int = 1000,
) -> dict[str, float]:
    """Calculate accuracy with error bars using bootstrap."""
    # Simple implementation - can be enhanced with proper bootstrap
    accuracy = calculate_accuracy(predictions, ground_truth)

    # For now, return simple stats
    # TODO: Implement proper bootstrap confidence intervals
    n = len(predictions)
    std_error = (accuracy * (1 - accuracy) / n) ** 0.5

    return {
        "accuracy": accuracy,
        "std_error": std_error,
        "n": n,
    }


def evaluate_results(config: dict[str, Any]) -> None:
    """Evaluate experiment results - both correctness and cipher adherence."""
    results_dir = Path(config["results_dir"])

    # Load results
    results_path = results_dir / "results.json"
    with open(results_path) as f:
        results = json.load(f)

    # Load encoding to check adherence
    encoding_type = results["config"]["cipher"]["type"]
    encoding = get_encoding(encoding_type, **results["config"]["cipher"].get("params", {}))

    # Extract predictions and ground truth from results
    predictions = []
    ground_truth = []
    adherence_scores = []

    for item in results.get("responses", []):
        response = item.get("response", "")
        expected_answer = item.get("expected_answer", "")

        # Extract predicted answer from response
        predicted_answer = extract_answer_from_boxed(response)
        predictions.append(predicted_answer)
        ground_truth.append(expected_answer)

        # Evaluate adherence
        adherence = evaluate_adherence(response, encoding, encoding_type)
        adherence_scores.append(adherence)
        item["adherence"] = adherence

    # Calculate correctness metrics
    correctness_metrics = calculate_accuracy_with_error_bars(predictions, ground_truth)

    # Calculate adherence metrics
    adherence_rate = (
        sum(1 for a in adherence_scores if a["is_adherent"]) / len(adherence_scores)
        if adherence_scores
        else 0.0
    )
    adherence_std_error = (
        (adherence_rate * (1 - adherence_rate) / len(adherence_scores)) ** 0.5
        if adherence_scores
        else 0.0
    )

    # Combine metrics
    metrics = {
        "correctness": correctness_metrics,
        "adherence": {
            "rate": adherence_rate,
            "std_error": adherence_std_error,
            "n": len(adherence_scores),
        },
    }

    # Add metrics to results
    results["metrics"] = metrics

    # Save updated results
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation complete! Updated results in {results_path}")
    print(
        f"Correctness: {correctness_metrics['accuracy']:.3f} ± "
        f"{correctness_metrics['std_error']:.3f}"
    )
    print(f"Adherence: {adherence_rate:.3f} ± {adherence_std_error:.3f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate experiment results")
    parser.add_argument(
        "--config",
        type=str,
        default="config/eval_config.yaml",
        help="Path to eval config YAML file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate_results(config)


if __name__ == "__main__":
    main()
