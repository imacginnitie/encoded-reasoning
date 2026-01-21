"""Dataset loading utilities."""

import json
import re
from pathlib import Path

from datasets import load_dataset


def load_math500_dataset(cache_dir=None, split=None):
    """Load MATH 500 dataset."""
    dataset = load_dataset("ricdomolm/MATH-500", cache_dir=str(cache_dir) if cache_dir else None)
    data = dataset[split] if split else dataset.get("test", list(dataset.values())[0])
    return [
        {
            "problem": dict(item).get("problem", ""),
            "solution": dict(item).get("solution", ""),
            "answer": dict(item).get("answer", ""),
        }
        for item in data
    ]


def extract_answer_from_boxed(text: str) -> str:
    """Extract answer from \\boxed{} format or "Answer: <number>" format, handling nested braces."""
    # First, try to extract from "Answer: <number>" format (for direct/filler schemes)
    # Handle duplicate prefixes like "Answer: Answer: ..." by stripping all of them
    answer_match = re.search(r"Answer:\s*(.+)", text, re.IGNORECASE)
    if answer_match:
        # Extract everything after the first "Answer:", then strip any remaining "Answer:" prefixes
        answer = answer_match.group(1).strip()
        # Strip any duplicate "Answer:" prefixes
        while answer.lower().startswith("answer:"):
            answer = answer[7:].strip()
        # Normalize whitespace: remove spaces after commas and around parentheses
        answer = re.sub(r",\s+", ",", answer)  # Remove space after comma
        answer = re.sub(r"\(\s+", "(", answer)  # Remove space after opening paren
        answer = re.sub(r"\s+\)", ")", answer)  # Remove space before closing paren
        return answer

    # Fall back to \\boxed{} format
    pattern = r"\\boxed\{"
    matches = []
    start = 0
    while True:
        match = re.search(pattern, text[start:])
        if not match:
            break
        boxed_start = start + match.end() - 1  # Position of opening brace
        # Find matching closing brace by counting depth
        depth = 0
        i = boxed_start + 1
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                if depth == 0:
                    # Found matching closing brace
                    matches.append(text[boxed_start + 1 : i])
                    break
                depth -= 1
            i += 1
        start = boxed_start + 1

    if matches:
        # Return the last match and normalize whitespace
        answer = matches[-1].strip()
        # Normalize whitespace: remove spaces after commas and around parentheses
        answer = re.sub(r",\s+", ",", answer)  # Remove space after comma
        answer = re.sub(r"\(\s+", "(", answer)  # Remove space after opening paren
        answer = re.sub(r"\s+\)", ")", answer)  # Remove space before closing paren
        return answer

    # If no format found, return the text as-is (might be a plain answer)
    # This handles cases where we've already stripped "Answer:" prefix
    if text.strip():
        answer = text.strip()
        # Normalize whitespace: remove spaces after commas and around parentheses
        answer = re.sub(r",\s+", ",", answer)  # Remove space after comma
        answer = re.sub(r"\(\s+", "(", answer)  # Remove space after opening paren
        answer = re.sub(r"\s+\)", ")", answer)  # Remove space before closing paren
        return answer

    return ""


def save_processed_dataset(examples, output_path):
    """Save processed dataset as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def load_processed_dataset(input_path):
    """Load processed dataset from JSONL."""
    with open(input_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_encoded_examples(encoding_type: str, num_examples: int):
    """Load pre-made encoded examples if available."""
    import random

    encoded_examples_path = Path(f"data/encoded_examples/{encoding_type}.jsonl")
    if not encoded_examples_path.exists():
        return None

    examples = load_processed_dataset(encoded_examples_path)
    if len(examples) > num_examples:
        random.seed(42)
        return random.sample(examples, num_examples)
    return examples
