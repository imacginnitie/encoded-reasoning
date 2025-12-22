"""Run experiment: send prompts to LLM API and collect responses."""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import anthropic
import openai
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from encoded_reasoning.ciphers import get_encoding_scheme
from encoded_reasoning.dataset import (
    extract_answer_from_boxed,
    load_math500_dataset,
    save_processed_dataset,
)


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def call_llm(prompt: str, provider: str, model: str, api_key: str, max_retries: int = 5):
    """Call LLM API with retries."""
    for attempt in range(max_retries):
        try:
            if provider == "openai":
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return response.choices[0].message.content or ""
            elif provider == "anthropic":
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                # Extract text from Anthropic response blocks
                text_parts = []
                for block in response.content:
                    if hasattr(block, "text"):
                        text_parts.append(getattr(block, "text", ""))
                return "".join(text_parts)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt if "rate limit" in str(e).lower() or "429" in str(e) else 1
                print(f"Error (attempt {attempt + 1}/{max_retries}): {e}. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError("Failed after retries")


def format_prompt(examples, encoding_scheme, include_instructions=True):
    """Format few-shot prompt."""
    parts = []

    if include_instructions and not encoding_scheme.get("is_programmatic", True):
        parts.append(f"Instructions: {encoding_scheme['instruction']}\n")

    parts.append("Examples:")
    for ex in examples:
        parts.append(f"Input: {ex['input']}")
        parts.append(f"Output: {ex['output']}\n")

    return "\n".join(parts)


def process_dataset(config):
    """Process dataset and create prompts."""
    scheme_name = config["cipher"]["type"]
    scheme = get_encoding_scheme(scheme_name, **config["cipher"].get("params", {}))

    # Load dataset
    all_examples = load_math500_dataset(cache_dir="data/cache")

    # Split
    random.seed(42)
    random.shuffle(all_examples)

    n_train = int(len(all_examples) * config["dataset"].get("train_split", 0.7))
    train_examples = all_examples[:n_train]
    test_examples = all_examples[n_train : n_train + config["dataset"]["num_test_examples"]]

    # Sample few-shot examples
    k = config["prompt"]["num_examples"]
    few_shot_examples = random.sample(train_examples, min(k, len(train_examples)))

    # Process examples
    few_shot_processed = []
    for ex in few_shot_examples:
        problem = ex["problem"]
        solution = ex.get("solution", "")
        answer = extract_answer_from_boxed(ex.get("answer", ""))

        if scheme.get("is_programmatic", True):
            encoded_problem = scheme["encode"](problem)
            encoded_solution = scheme["encode"](solution) if solution else ""
            encoded_answer = scheme["encode"](answer) if answer else ""
            few_shot_processed.append(
                {
                    "input": encoded_problem,
                    "output": (
                        f"{encoded_solution}\n\\boxed{{{encoded_answer}}}"
                        if answer
                        else encoded_solution
                    ),
                }
            )
        else:
            few_shot_processed.append(
                {
                    "input": problem,
                    "output": (f"{solution}\n\\boxed{{{answer}}}" if answer else solution),
                }
            )

    # Process test examples
    test_processed = []
    for ex in test_examples:
        problem = ex["problem"]
        answer = extract_answer_from_boxed(ex.get("answer", ""))

        if scheme.get("is_programmatic", True):
            encoded_problem = scheme["encode"](problem)
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

    # Save processed data
    output_dir = Path("data/processed") / scheme_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_processed_dataset(few_shot_processed, output_dir / "few_shot.jsonl")
    save_processed_dataset(test_processed, output_dir / "test.jsonl")

    # Create prompt
    prompt = format_prompt(
        few_shot_processed, scheme, config["prompt"].get("include_instructions", True)
    )
    with open(output_dir / "few_shot_prompt.txt", "w") as f:
        f.write(prompt)

    return test_processed, prompt


def run_experiment(config):
    """Run experiment."""
    provider = config["model"]["provider"]
    model = config["model"]["name"]

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if not api_key:
        raise ValueError(
            f"API key not found for {provider}. "
            f"Set {provider.upper()}_API_KEY environment variable."
        )

    # Process dataset
    test_examples, few_shot_prompt = process_dataset(config)

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config["experiment"]["output_dir"]) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Run experiment
    results = {"timestamp": timestamp, "config": config, "responses": []}

    print(f"Running experiment on {len(test_examples)} examples...")
    for i, example in enumerate(test_examples, 1):
        print(f"Processing {i}/{len(test_examples)}...")

        prompt = f"{few_shot_prompt}\n\nInput: {example['input']}\nOutput:"

        try:
            response = call_llm(prompt, provider, model, api_key)
            results["responses"].append(
                {
                    "example_index": i - 1,
                    "input": example["input"],
                    "expected_answer": example.get("expected_answer", ""),
                    "original_problem": example.get("original_problem", ""),
                    "response": response,
                    "prompt": prompt,
                }
            )

            if i % 10 == 0:
                with open(results_dir / "results_intermediate.json", "w") as f:
                    json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error: {e}")
            results["responses"].append(
                {
                    "example_index": i - 1,
                    "input": example["input"],
                    "response": None,
                    "error": str(e),
                }
            )

    # Save results
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Experiment complete! Results saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
