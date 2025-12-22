"""Dataset loading utilities for MATH 500."""

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def load_math500_dataset(
    cache_dir: str | Path | None = None,
    split: str | None = None,
) -> list[dict[str, Any]]:
    """
    Load MATH 500 dataset from HuggingFace.

    Args:
        cache_dir: Directory to cache the dataset
        split: Dataset split to load (train, test, etc.)

    Returns:
        List of examples, each containing:
        - problem: The problem statement
        - solution: The solution steps
        - answer: The final answer (usually in \\boxed{})
    """
    try:
        # Convert Path to str if needed
        cache_dir_str: str | None = None
        if cache_dir is not None:
            cache_dir_str = str(cache_dir) if isinstance(cache_dir, Path) else cache_dir

        dataset = load_dataset("ricdomolm/MATH-500", cache_dir=cache_dir_str)

        # If split specified, use it; otherwise use test split
        if split:
            data = dataset[split]
        else:
            # Default to test split if available, otherwise use first available split
            splits = list(dataset.keys())
            if "test" in splits:
                data = dataset["test"]
            else:
                data = dataset[splits[0]]

        # Convert to list of dicts
        examples = []
        for item in data:
            # Convert item to dict to ensure proper type handling
            item_dict = dict(item) if not isinstance(item, dict) else item
            examples.append(
                {
                    "problem": item_dict.get("problem", ""),
                    "solution": item_dict.get("solution", ""),
                    "answer": item_dict.get("answer", ""),
                }
            )

        return examples
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MATH 500 dataset. Make sure datasets library is installed. Error: {e}"
        )


def extract_answer_from_boxed(text: str) -> str:
    """
    Extract answer from \\boxed{} format.

    Args:
        text: Text that may contain \\boxed{answer}

    Returns:
        The answer inside \\boxed{}, or empty string if not found
    """
    import re

    # Look for \boxed{...} pattern
    pattern = r"\\boxed\{([^}]+)\}"
    matches = re.findall(pattern, text)

    if matches:
        # Return the last match (most likely the final answer)
        return matches[-1].strip()

    return ""


def save_processed_dataset(
    examples: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save processed dataset as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


def load_processed_dataset(input_path: Path) -> list[dict[str, Any]]:
    """Load processed dataset from JSONL."""
    examples = []
    with open(input_path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples
