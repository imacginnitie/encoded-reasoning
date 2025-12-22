"""Process dataset: encode examples and format prompts."""

import argparse
import random
from pathlib import Path
from typing import Any, cast

import yaml

from encoded_reasoning.ciphers import Cipher
from encoded_reasoning.dataset import (
    extract_answer_from_boxed,
    load_math500_dataset,
    save_processed_dataset,
)
from encoded_reasoning.encoding import get_encoding, is_programmatic


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def format_few_shot_prompt(
    examples: list[dict[str, str]],
    encoding,
    include_instructions: bool = True,
) -> str:
    """Format few-shot examples into a prompt."""
    prompt_parts = []

    if include_instructions:
        # Use encoding's instruction method if available
        if hasattr(encoding, "get_instruction"):
            instruction = encoding.get_instruction()
            prompt_parts.append(f"Instructions: {instruction}")
        else:
            encoding_type = (
                type(encoding).__name__.replace("Cipher", "").replace("Encoding", "").lower()
            )
            prompt_parts.append(f"Instructions: Use the {encoding_type} encoding.")
        prompt_parts.append("")

    # For programmatic ciphers, show encoded examples
    # For prompt encodings, examples are shown as-is (model will encode them)
    if is_programmatic(encoding):
        prompt_parts.append("Examples:")
        for example in examples:
            encoded_input = encoding.encode(example["input"])
            encoded_output = encoding.encode(example["output"])
            prompt_parts.append(f"Input: {encoded_input}")
            prompt_parts.append(f"Output: {encoded_output}")
            prompt_parts.append("")
    else:
        # For prompt encodings, show unencoded examples
        # The model will follow the instruction to encode its thinking
        prompt_parts.append("Examples:")
        for example in examples:
            prompt_parts.append(f"Input: {example['input']}")
            prompt_parts.append(f"Output: {example['output']}")
            prompt_parts.append("")

    return "\n".join(prompt_parts)


def process_dataset(config: dict[str, Any]) -> None:
    """Process MATH 500 dataset: create encoded examples and format prompts."""
    encoding_type = config["cipher"]["type"]
    encoding = get_encoding(encoding_type, **config["cipher"].get("params", {}))

    num_examples = config["prompt"]["num_examples"]
    num_test = config["dataset"]["num_test_examples"]
    dataset_name = config["dataset"].get("name", "math500")

    # Load MATH 500 dataset
    print(f"Loading {dataset_name} dataset...")
    all_examples = load_math500_dataset(cache_dir="data/cache")

    # Shuffle and split
    random.seed(42)  # For reproducibility
    random.shuffle(all_examples)

    # Split into train/val/test
    train_split = config["dataset"].get("train_split", 0.7)
    val_split = config["dataset"].get("val_split", 0.15)

    total = len(all_examples)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)

    train_examples = all_examples[:train_end]
    test_examples = all_examples[val_end : val_end + num_test]  # Limit test examples

    # Create output directory structure
    output_dir = Path("data/processed") / encoding_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample few-shot examples from train set
    few_shot_examples = random.sample(train_examples, min(num_examples, len(train_examples)))

    # Process few-shot examples
    few_shot_processed = []
    for ex in few_shot_examples:
        problem = ex["problem"]
        solution = ex.get("solution", "")
        answer = extract_answer_from_boxed(ex.get("answer", ""))

        if is_programmatic(encoding):
            # Encode for programmatic ciphers
            cipher = cast(Cipher, encoding)
            encoded_problem = cipher.encode(problem)
            encoded_solution = cipher.encode(solution) if solution else ""
            encoded_answer = cipher.encode(answer) if answer else ""

            few_shot_processed.append(
                {
                    "input": encoded_problem,
                    "output": (
                        f"{encoded_solution}\n\\boxed{{{encoded_answer}}}"
                        if encoded_answer
                        else encoded_solution
                    ),
                    "original_problem": problem,
                    "original_answer": answer,
                }
            )
        else:
            # Keep unencoded for prompt encodings
            few_shot_processed.append(
                {
                    "input": problem,
                    "output": f"{solution}\n\\boxed{{{answer}}}" if answer else solution,
                    "original_problem": problem,
                    "original_answer": answer,
                }
            )

    # Process test examples
    test_processed = []
    for ex in test_examples:
        problem = ex["problem"]
        answer = extract_answer_from_boxed(ex.get("answer", ""))

        if is_programmatic(encoding):
            cipher = cast(Cipher, encoding)
            encoded_problem = cipher.encode(problem)
            test_processed.append(
                {
                    "input": encoded_problem,
                    "expected_answer": answer,
                    "original_problem": problem,
                }
            )
        else:
            test_processed.append(
                {
                    "input": problem,
                    "expected_answer": answer,
                    "original_problem": problem,
                }
            )

    # Save processed datasets
    few_shot_path = output_dir / "few_shot.jsonl"
    test_path = output_dir / "test.jsonl"

    save_processed_dataset(few_shot_processed, few_shot_path)
    save_processed_dataset(test_processed, test_path)

    # Create formatted prompt
    prompt = format_few_shot_prompt(
        few_shot_processed,
        encoding,
        include_instructions=config["prompt"].get("include_instructions", True),
    )

    prompt_path = output_dir / "few_shot_prompt.txt"
    with open(prompt_path, "w") as f:
        f.write(prompt)

    encoding_type_str = "programmatic cipher" if is_programmatic(encoding) else "prompt encoding"
    print(f"Processing complete with {encoding_type} ({encoding_type_str})")
    print(f"Created {len(few_shot_processed)} few-shot examples")
    print(f"Created {len(test_processed)} test examples")
    print(f"Saved to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Process dataset for experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    process_dataset(config)


if __name__ == "__main__":
    main()
