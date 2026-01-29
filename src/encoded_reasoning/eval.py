"""Evaluate experiment results."""

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from encoded_reasoning.adherence import check_adherence
from encoded_reasoning.ciphers import get_encoding_scheme
from encoded_reasoning.dataset import extract_answer_from_boxed, normalize_latex_answer


def _calculate_std_error(rate: float, n: int) -> float:
    """Calculate standard error for a binomial proportion.

    Uses the formula: SE = sqrt(p * (1-p) / n)
    This is the standard error of a proportion assuming binomial distribution.

    Args:
        rate: The proportion (between 0 and 1)
        n: Sample size

    Returns:
        Standard error of the proportion
    """
    return (rate * (1 - rate) / n) ** 0.5 if n > 0 else 0.0


def evaluate_results(results_dir: str | Path, config: dict | None = None):
    """Evaluate experiment results."""
    results_dir = Path(results_dir)
    results = json.loads((results_dir / "results.json").read_text())

    encoding_type = results["config"]["cipher"]["type"]
    scheme = get_encoding_scheme(encoding_type, **results["config"]["cipher"].get("params", {}))

    predictions, ground_truth, adherence_scores = [], [], []
    responses = results.get("responses", [])
    print(f"Evaluating {len(responses)} responses...")
    for item in tqdm(responses, desc="Evaluating responses", unit="response"):
        predicted = extract_answer_from_boxed(item.get("response") or "")
        predictions.append(predicted)
        expected = item.get("expected_answer", "")
        ground_truth.append(expected)
        is_correct = normalize_latex_answer(predicted) == normalize_latex_answer(expected)
        item["is_correct"] = is_correct
        adherence = check_adherence(item.get("response") or "", encoding_type, scheme)
        adherence_scores.append(adherence)
        item["adherence"] = adherence

    n = len(predictions)
    correct = sum(
        1
        for p, gt in zip(predictions, ground_truth)
        if normalize_latex_answer(p) == normalize_latex_answer(gt)
    )
    accuracy = correct / n if n > 0 else 0.0
    adherence_rate = sum(1 for a in adherence_scores if a["is_adherent"]) / n if n > 0 else 0.0
    adherent_and_correct = sum(
        1
        for p, gt, a in zip(predictions, ground_truth, adherence_scores)
        if normalize_latex_answer(p) == normalize_latex_answer(gt) and a["is_adherent"]
    )
    adherent_and_correct_rate = adherent_and_correct / n if n > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "accuracy_std_error": _calculate_std_error(accuracy, n),
        "adherence_rate": adherence_rate,
        "adherence_std_error": _calculate_std_error(adherence_rate, n),
        "adherent_and_correct_rate": adherent_and_correct_rate,
        "adherent_and_correct_std_error": _calculate_std_error(adherent_and_correct_rate, n),
        "n": n,
        "correct": correct,
        "adherent_and_correct": adherent_and_correct,
    }

    results["metrics"] = metrics
    (results_dir / "results.json").write_text(json.dumps(results, indent=2))
    acc_str = f"{metrics['accuracy']:.3f} ± {metrics['accuracy_std_error']:.3f}"
    adh_str = f"{metrics['adherence_rate']:.3f} ± {metrics['adherence_std_error']:.3f}"
    adh_corr_str = (
        f"{metrics['adherent_and_correct_rate']:.3f} ± "
        f"{metrics['adherent_and_correct_std_error']:.3f}"
    )
    print(f"Accuracy: {acc_str} ({correct}/{n})")
    print(f"Adherence: {adh_str} ({sum(1 for a in adherence_scores if a['is_adherent'])}/{n})")
    print(f"Adherent & Correct: {adh_corr_str} ({metrics['adherent_and_correct']}/{n})")


def main():
    parser = argparse.ArgumentParser()
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
        with open(args.config) as f:
            config = yaml.safe_load(f)

    evaluate_results(args.results_dir, config)


if __name__ == "__main__":
    main()
