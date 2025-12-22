"""Evaluate experiment results."""

import argparse
import json
import re
from pathlib import Path

import yaml

from encoded_reasoning.adherence import check_adherence
from encoded_reasoning.ciphers import get_encoding_scheme


def extract_answer_from_boxed(text: str) -> str:
    """Extract answer from \\boxed{} format."""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else ""


def evaluate_results(config):
    """Evaluate experiment results."""
    results_dir = Path(config["results_dir"])

    # Load results
    with open(results_dir / "results.json") as f:
        results = json.load(f)

    # Get encoding scheme for adherence checking
    encoding_type = results["config"]["cipher"]["type"]
    scheme = get_encoding_scheme(encoding_type, **results["config"]["cipher"].get("params", {}))

    # Calculate accuracy and adherence
    predictions = []
    ground_truth = []
    adherence_scores = []

    for item in results.get("responses", []):
        response = item.get("response", "")
        expected = item.get("expected_answer", "")

        predicted = extract_answer_from_boxed(response)
        predictions.append(predicted)
        ground_truth.append(expected)

        # Check adherence
        adherence = check_adherence(response, encoding_type, scheme)
        adherence_scores.append(adherence)
        item["adherence"] = adherence

    # Calculate accuracy
    correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
    accuracy = correct / len(predictions) if predictions else 0.0

    # Calculate adherence rate
    adherence_rate = (
        sum(1 for a in adherence_scores if a["is_adherent"]) / len(adherence_scores)
        if adherence_scores
        else 0.0
    )

    # Calculate standard errors
    n = len(predictions)
    accuracy_std_error = (accuracy * (1 - accuracy) / n) ** 0.5 if n > 0 else 0.0
    adherence_std_error = (adherence_rate * (1 - adherence_rate) / n) ** 0.5 if n > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "accuracy_std_error": accuracy_std_error,
        "adherence_rate": adherence_rate,
        "adherence_std_error": adherence_std_error,
        "n": n,
        "correct": correct,
    }

    # Save metrics
    results["metrics"] = metrics
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Accuracy: {accuracy:.3f} ± {accuracy_std_error:.3f} (n={n})")
    print(f"Adherence: {adherence_rate:.3f} ± {adherence_std_error:.3f} (n={n})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/eval_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    evaluate_results(config)


if __name__ == "__main__":
    main()
