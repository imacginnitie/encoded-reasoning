"""Dataset loading utilities."""

import json
import re
from pathlib import Path

from datasets import load_dataset


def load_math500_dataset(cache_dir=None, split=None):
    """Load MATH 500 dataset."""
    cache_dir_str = str(cache_dir) if isinstance(cache_dir, Path) else cache_dir
    dataset = load_dataset("ricdomolm/MATH-500", cache_dir=cache_dir_str)

    if split:
        data = dataset[split]
    else:
        splits = list(dataset.keys())
        data = dataset["test"] if "test" in splits else dataset[splits[0]]

    examples = []
    for item in data:
        # Convert to dict if needed
        item_dict = dict(item) if not isinstance(item, dict) else item
        examples.append(
            {
                "problem": item_dict.get("problem", ""),
                "solution": item_dict.get("solution", ""),
                "answer": item_dict.get("answer", ""),
            }
        )
    return examples


def extract_answer_from_boxed(text: str) -> str:
    """Extract answer from \\boxed{} format."""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else ""


def save_processed_dataset(examples, output_path):
    """Save processed dataset as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def load_processed_dataset(input_path):
    """Load processed dataset from JSONL."""
    with open(input_path) as f:
        return [json.loads(line) for line in f]
