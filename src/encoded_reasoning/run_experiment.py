"""Run experiment: send prompts to LLM API and collect responses."""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
import openai
import yaml

from encoded_reasoning.dataset import load_processed_dataset


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_api_key(provider: str) -> str:
    """Load API key from environment variables."""
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return key
    elif provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        return key
    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_llm_api(
    prompt: str,
    provider: str,
    model_name: str,
    api_key: str,
    max_retries: int = 5,
) -> str:
    """Call LLM API with exponential backoff for rate limiting."""
    for attempt in range(max_retries):
        try:
            if provider == "openai":
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,  # Deterministic for reproducibility
                )
                return response.choices[0].message.content or ""
            elif provider == "anthropic":
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model=model_name,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,  # Deterministic for reproducibility
                )
                # Anthropic returns a list of text blocks
                return "".join(block.text for block in response.content if hasattr(block, "text"))
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:
            # Check if it's a rate limit error
            is_rate_limit = (
                (isinstance(e, openai.RateLimitError) if provider == "openai" else False)
                or (isinstance(e, anthropic.RateLimitError) if provider == "anthropic" else False)
                or "rate limit" in str(e).lower()
                or "429" in str(e)
            )

            if attempt < max_retries - 1:
                if is_rate_limit:
                    wait_time = 2**attempt
                    print(f"Rate limit hit, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    # For other errors, wait a bit but don't retry too many times
                    wait_time = 1
                    print(f"Error occurred: {e}. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
            else:
                error_msg = f"Failed to get response after {max_retries} retries: {e}"
                raise RuntimeError(error_msg) from e

    raise RuntimeError("Failed to get response after retries")


def run_experiment(config: dict[str, Any]) -> None:
    """Run experiment: process dataset and call LLM API."""
    provider = config["model"]["provider"]
    model_name = config["model"]["name"]

    api_key = load_api_key(provider)

    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config["experiment"]["output_dir"]) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config used for this experiment
    config_path = results_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Load processed dataset
    encoding_type = config["cipher"]["type"]
    processed_dir = Path("data/processed") / encoding_type

    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_dir}. "
            "Please run process_dataset.py first to process the dataset."
        )

    test_path = processed_dir / "test.jsonl"
    few_shot_prompt_path = processed_dir / "few_shot_prompt.txt"

    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_path}")
    if not few_shot_prompt_path.exists():
        raise FileNotFoundError(f"Few-shot prompt not found at {few_shot_prompt_path}")

    # Load test examples and few-shot prompt
    test_examples = load_processed_dataset(test_path)
    with open(few_shot_prompt_path) as f:
        few_shot_prompt = f.read()

    results = {
        "timestamp": timestamp,
        "config": config,
        "responses": [],
    }

    # Iterate through test examples
    print(f"Running experiment on {len(test_examples)} test examples...")
    for i, example in enumerate(test_examples, 1):
        print(f"Processing example {i}/{len(test_examples)}...")

        # Format full prompt: few-shot prompt + test example input
        full_prompt = f"{few_shot_prompt}\n\nInput: {example['input']}\nOutput:"

        # Call LLM API
        try:
            response = call_llm_api(
                prompt=full_prompt,
                provider=provider,
                model_name=model_name,
                api_key=api_key,
            )

            # Store response with metadata
            results["responses"].append(
                {
                    "example_index": i - 1,
                    "input": example["input"],
                    "expected_answer": example.get("expected_answer", ""),
                    "original_problem": example.get("original_problem", ""),
                    "response": response,
                    "prompt": full_prompt,
                }
            )

            # Save intermediate results periodically (every 10 examples)
            if i % 10 == 0:
                intermediate_path = results_dir / "results_intermediate.json"
                with open(intermediate_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  Saved intermediate results ({i} examples processed)")
        except Exception as e:
            print(f"  Error processing example {i}: {e}")
            results["responses"].append(
                {
                    "example_index": i - 1,
                    "input": example["input"],
                    "expected_answer": example.get("expected_answer", ""),
                    "original_problem": example.get("original_problem", ""),
                    "response": None,
                    "error": str(e),
                }
            )

    # Save final results
    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Experiment complete! Results saved to {results_dir}")
    print("To evaluate and plot, update config/eval_config.yaml with:")
    print(f"  results_dir: {results_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
