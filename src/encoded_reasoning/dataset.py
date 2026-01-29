"""Dataset loading utilities."""

import json
import random
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


def load_aime2025_dataset(cache_dir=None):
    """Load AIME 2025 dataset."""
    dataset = load_dataset(
        "yentinglin/aime_2025", cache_dir=str(cache_dir) if cache_dir else None
    )
    data = dataset["train"]
    return [
        {
            "problem": dict(item).get("problem", ""),
            "solution": dict(item).get("solution", ""),
            "answer": str(dict(item).get("answer", "")),
        }
        for item in data
    ]


def load_gpqa_diamond_dataset(cache_dir=None):
    """Load GPQA Diamond dataset with shuffled multiple-choice options."""
    dataset = load_dataset(
        "Idavidrein/gpqa", "gpqa_diamond", cache_dir=str(cache_dir) if cache_dir else None
    )
    data = dataset["train"]
    examples = []
    for idx, item in enumerate(data):
        item = dict(item)
        choices = [
            item["Correct Answer"],
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]
        rng = random.Random(idx)
        rng.shuffle(choices)
        correct_letter = chr(65 + choices.index(item["Correct Answer"]))

        choice_text = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
        problem = f"{item['Question']}\n\n{choice_text}"

        examples.append({
            "problem": problem,
            "solution": item.get("Explanation", ""),
            "answer": correct_letter,
        })
    return examples


def load_dataset_by_name(name: str, cache_dir=None):
    """Load dataset by name."""
    loaders = {
        "math500": load_math500_dataset,
        "aime2025": load_aime2025_dataset,
        "gpqa_diamond": load_gpqa_diamond_dataset,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Must be one of: {', '.join(loaders)}")
    return loaders[name](cache_dir=cache_dir)


def normalize_latex_answer(s: str) -> str:
    """Normalize LaTeX answer string for comparison.

    Handles common equivalent representations:
    - \\dfrac/\\tfrac -> \\frac
    - \\frac XY -> \\frac{X}{Y} (single-char shorthand)
    - \\text{...} -> bare content
    - Whitespace normalization
    """
    s = s.strip()
    # \dfrac and \tfrac are display variants of \frac
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")
    # \text{...} -> just the content inside
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\s*\{([^}]*)\}", r"\1", s)
    # \frac XY or \fracXY -> \frac{X}{Y} for single-char arguments (LaTeX shorthand)
    # e.g. \frac 59 -> \frac{5}{9}, \frac43 -> \frac{4}{3}
    s = re.sub(r"\\frac\s*(\w)(\w)(?!\w)", r"\\frac{\1}{\2}", s)
    # \frac{X}Y -> \frac{X}{Y} for single-char second argument
    s = re.sub(r"\\frac(\{[^}]*\})\s*(\w)(?![{\\\w])", r"\\frac\1{\2}", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


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


def load_encoded_examples(encoding_type: str, num_examples: int, dataset_name: str | None = None):
    """Load pre-made encoded examples if available.

    Checks dataset-specific directory first
    (e.g., data/encoded_examples/gpqa_diamond/identity.jsonl),
    then falls back to the generic directory.
    """
    if dataset_name:
        dataset_path = Path(f"data/encoded_examples/{dataset_name}/{encoding_type}.jsonl")
        if dataset_path.exists():
            examples = load_processed_dataset(dataset_path)
            if len(examples) > num_examples:
                random.seed(42)
                return random.sample(examples, num_examples)
            return examples

    encoded_examples_path = Path(f"data/encoded_examples/{encoding_type}.jsonl")
    if not encoded_examples_path.exists():
        return None

    examples = load_processed_dataset(encoded_examples_path)
    if len(examples) > num_examples:
        random.seed(42)
        return random.sample(examples, num_examples)
    return examples
