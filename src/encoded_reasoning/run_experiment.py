"""Run experiment: send prompts to LLM API and collect responses."""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
import openai
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from encoded_reasoning.ciphers import generate_filler_tokens, get_encoding_scheme
from encoded_reasoning.dataset import (
    extract_answer_from_boxed,
    load_encoded_examples,
    load_math500_dataset,
    save_processed_dataset,
)

# Load environment variables from .env file
load_dotenv()


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def call_llm(
    prompt: str,
    provider: str,
    model: str,
    api_key: str,
    max_retries: int = 5,
    assistant_prefill: Optional[str] = None,
):
    """Call LLM API with retries.

    Args:
        prompt: The user prompt
        provider: The LLM provider ("openai" or "anthropic")
        model: The model name
        api_key: The API key
        max_retries: Maximum number of retry attempts
        assistant_prefill: Optional prefill text for assistant response (Anthropic only)
    """
    for attempt in range(max_retries):
        try:
            if provider == "openai":
                client = openai.OpenAI(api_key=api_key)
                # For OpenAI, include prefill in the prompt text if provided
                full_prompt = prompt
                if assistant_prefill:
                    full_prompt = f"{prompt} {assistant_prefill}"
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.0,
                )
                response_text = response.choices[0].message.content or ""
                # If we prefilled, prepend the prefill to the response
                if assistant_prefill:
                    return assistant_prefill + " " + response_text
                return response_text
            elif provider == "anthropic":
                client = anthropic.Anthropic(api_key=api_key)
                messages: list = [{"role": "user", "content": prompt}]
                # For Anthropic, use assistant message prefill if provided
                if assistant_prefill:
                    messages.append({"role": "assistant", "content": assistant_prefill})
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=messages,  # type: ignore
                    temperature=0.0,
                )
                # Extract text from Anthropic response blocks
                text_parts = []
                for block in response.content:
                    if hasattr(block, "text"):
                        text_parts.append(getattr(block, "text", ""))
                # If we prefilled, prepend the prefill to the response
                if assistant_prefill:
                    return assistant_prefill + " " + "".join(text_parts)
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


def format_prompt(examples, encoding_scheme):
    """Format few-shot prompt."""
    parts = []

    if "instruction" in encoding_scheme:
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

    # Check for pre-made encoded examples first
    k = config["prompt"]["num_examples"]
    # For filler ciphers, use direct examples (they'll get filler tokens appended dynamically)
    if scheme_name == "filler":
        encoded_type = "direct"
    else:
        encoded_type = scheme_name
    few_shot_processed = load_encoded_examples(encoded_type, k)

    if few_shot_processed is None:
        # Fall back to generating examples from MATH-500 dataset
        # Load dataset
        all_examples = load_math500_dataset(cache_dir="data/cache")

        # Split
        random.seed(42)
        random.shuffle(all_examples)

        # Only reserve what's needed for few-shot examples (with a small buffer)
        # Since we're not actually training, we don't need 70% of the data
        num_test_requested = config["dataset"]["num_test_examples"]
        # Reserve enough for few-shot examples: use k * 3 as buffer, or use train_split if specified
        if "train_split" in config["dataset"]:
            n_train = int(len(all_examples) * config["dataset"]["train_split"])
        else:
            # Reserve only what's needed: k examples + small buffer (k * 2)
            n_train = min(k * 3, int(len(all_examples) * 0.1))  # Cap at 10% max

        train_examples = all_examples[:n_train]
        num_test_available = len(all_examples) - n_train
        if num_test_requested > num_test_available:
            print(
                f"Warning: Requested {num_test_requested} test examples, "
                f"but only {num_test_available} available after reserving "
                f"{n_train} for few-shot examples ({len(all_examples)} total). "
                f"Using {num_test_available} test examples."
            )
        test_examples = all_examples[
            n_train : n_train + min(num_test_requested, num_test_available)
        ]

        # Sample few-shot examples
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
    else:
        # If using pre-made examples, still need to load test examples
        all_examples = load_math500_dataset(cache_dir="data/cache")
        random.seed(42)
        random.shuffle(all_examples)
        # Only reserve what's needed for few-shot examples (with a small buffer)
        num_test_requested = config["dataset"]["num_test_examples"]
        if "train_split" in config["dataset"]:
            n_train = int(len(all_examples) * config["dataset"]["train_split"])
        else:
            # Reserve only what's needed: k examples + small buffer (k * 2)
            n_train = min(k * 3, int(len(all_examples) * 0.1))  # Cap at 10% max
        num_test_available = len(all_examples) - n_train
        if num_test_requested > num_test_available:
            print(
                f"Warning: Requested {num_test_requested} test examples, "
                f"but only {num_test_available} available after reserving "
                f"{n_train} for few-shot examples ({len(all_examples)} total). "
                f"Using {num_test_available} test examples."
            )
        test_examples = all_examples[
            n_train : n_train + min(num_test_requested, num_test_available)
        ]

    # For filler ciphers, append filler tokens to inputs dynamically
    if scheme_name == "filler" and "filler" in config["cipher"]:
        filler_config = config["cipher"]["filler"]
        filler_type = filler_config.get("type", "counting")
        filler_count = filler_config.get("count", 300)

        for ex in few_shot_processed:
            problem = ex["input"]
            filler_tokens = generate_filler_tokens(
                filler_type,
                filler_count,
                problem=problem,
                repeat_string=filler_config.get("string", None),
            )
            ex["input"] = f"{problem}\n\n{filler_tokens}"

    # Process test examples
    test_processed = []
    for ex in test_examples:
        problem = ex["problem"]
        answer = extract_answer_from_boxed(ex.get("answer", ""))

        if scheme.get("is_programmatic", True):
            encoded_problem = scheme["encode"](problem)
            test_input = encoded_problem
        else:
            test_input = problem

        # For filler ciphers, append filler tokens to test inputs
        if scheme_name == "filler" and "filler" in config["cipher"]:
            filler_config = config["cipher"]["filler"]
            filler_type = filler_config.get("type", "counting")
            filler_count = filler_config.get("count", 300)
            filler_tokens = generate_filler_tokens(
                filler_type,
                filler_count,
                problem=problem,
                repeat_string=filler_config.get("string", None),
            )
            test_input = f"{test_input}\n\n{filler_tokens}"

        test_processed.append(
            {
                "input": test_input,
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
    prompt = format_prompt(few_shot_processed, scheme)
    with open(output_dir / "few_shot_prompt.txt", "w") as f:
        f.write(prompt)

    return test_processed, prompt


def get_api_key(provider: str) -> Optional[str]:
    """Get API key for provider, return None if not found."""
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    elif provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def run_single_experiment(config, results_base_dir=None):
    """Run a single experiment with one model and cipher combination."""
    provider = config["model"]["provider"]
    model = config["model"]["name"]

    api_key = get_api_key(provider)

    if not api_key:
        raise ValueError(
            f"API key not found for {provider}. "
            f"Set {provider.upper()}_API_KEY environment variable."
        )

    # Process dataset
    test_examples, few_shot_prompt = process_dataset(config)

    # Get encoding scheme to check if we should prefill "Answer: " for direct/filler ciphers
    scheme_name = config["cipher"]["type"]
    scheme = get_encoding_scheme(scheme_name, **config["cipher"].get("params", {}))
    should_prefill_answer = scheme.get("is_direct", False)

    # Create results directory
    if results_base_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(config["experiment"]["output_dir"]) / timestamp
    else:
        # For matrix experiments, use the provided base directory
        results_dir = results_base_dir

    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Check for existing intermediate results to resume from
    intermediate_file = results_dir / "results_intermediate.json"
    if intermediate_file.exists():
        print(f"Found existing intermediate results at {intermediate_file}")
        print("Resuming from previous run...")
        with open(intermediate_file) as f:
            results = json.load(f)
        # Get the number of already processed examples
        num_processed = len(results.get("responses", []))
        print(f"Found {num_processed} already processed examples")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {"timestamp": timestamp, "config": config, "responses": []}
        num_processed = 0

    # Run experiment with progress bar
    remaining_examples = test_examples[num_processed:]
    if num_processed > 0:
        print(f"Skipping first {num_processed} examples (already processed)")

    for example in tqdm(
        remaining_examples,
        desc="Processing examples",
        unit="example",
        initial=num_processed,
        total=len(test_examples),
    ):
        i = len(results["responses"]) + 1
        # For direct/filler ciphers, prefill "Answer:" to guide the model
        # For Anthropic, use assistant message prefill; for OpenAI, include in prompt
        assistant_prefill: Optional[str] = "Answer:" if should_prefill_answer else None
        prompt = f"{few_shot_prompt}\n\nInput: {example['input']}\nOutput:"

        try:
            response = call_llm(
                prompt, provider, model, api_key, assistant_prefill=assistant_prefill
            )
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
        except Exception as e:
            print(f"\nError: {e}")
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

        # Save after every example for better caching/resume capability
        with open(results_dir / "results_intermediate.json", "w") as f:
            json.dump(results, f, indent=2)

    # Save final results
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Experiment complete! Results saved to {results_dir}")


def run_experiment(config):
    """Run experiment(s) - handles both single and matrix configurations."""
    # Check if this is a matrix configuration
    if "matrix" in config:
        # Matrix configuration: run all combinations
        matrix = config["matrix"]
        models = matrix.get("models", [])
        ciphers = matrix.get("ciphers", [])

        if not models or not ciphers:
            raise ValueError("Matrix configuration must have at least one model and one cipher")

        # Check API keys upfront and filter out models without keys
        available_models = []
        for model_config in models:
            provider = model_config["provider"]
            api_key = get_api_key(provider)
            if api_key:
                available_models.append(model_config)
            else:
                print(
                    f"Warning: Skipping {provider}/{model_config['name']} - "
                    f"{provider.upper()}_API_KEY not found"
                )

        if not available_models:
            raise ValueError(
                "No API keys found for any provider. "
                "Please set OPENAI_API_KEY and/or ANTHROPIC_API_KEY environment variables."
            )

        # Create base timestamp directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_results_dir = Path(config["experiment"]["output_dir"]) / timestamp
        base_results_dir.mkdir(parents=True, exist_ok=True)

        total_experiments = len(available_models) * len(ciphers)
        print(
            f"Running matrix experiment: {len(available_models)} model(s) × "
            f"{len(ciphers)} cipher(s) = {total_experiments} experiment(s)"
        )
        print(f"Results will be saved to: {base_results_dir}")
        print()

        experiment_num = 0
        for model_config in available_models:
            provider = model_config["provider"]
            model_name = model_config["name"]

            for cipher_config in ciphers:
                experiment_num += 1
                cipher_type = cipher_config["type"]

                # For filler ciphers, include all filler sub-settings in the directory name
                if cipher_type == "filler" and "filler" in cipher_config:
                    filler_config = cipher_config["filler"]
                    filler_parts = [cipher_type]

                    # Add type (required)
                    if "type" in filler_config:
                        filler_parts.append(filler_config["type"])

                    # Add all other parameters, with count last
                    other_params = {k: v for k, v in filler_config.items() if k != "type"}
                    # Separate count from other params
                    count_value = other_params.pop("count", None)
                    other_params_list = [(k, v) for k, v in sorted(other_params.items())]

                    # Add non-count params first (sorted)
                    for key, value in other_params_list:
                        # Skip None/empty values, but include 0
                        if value is not None and value != "":
                            # Sanitize value for filesystem (replace spaces/special chars)
                            value_str = str(value).replace(" ", "_").replace("/", "_")
                            filler_parts.append(value_str)

                    # Add count last
                    if count_value is not None:
                        filler_parts.append(str(count_value))

                    cipher_dir_name = "_".join(filler_parts)
                else:
                    cipher_dir_name = cipher_type

                print("=" * 60)
                exp_info = (
                    f"Experiment {experiment_num}/{total_experiments}: "
                    f"{provider}/{model_name}/{cipher_dir_name}"
                )
                print(exp_info)
                print("=" * 60)

                # Create a single-experiment config
                single_config = {
                    "model": model_config,
                    "cipher": cipher_config,
                    "prompt": config.get("prompt", {}),
                    "dataset": config.get("dataset", {}),
                    "experiment": config.get("experiment", {}),
                }

                # Create results directory for this combination
                results_dir = base_results_dir / provider / model_name / cipher_dir_name

                # Run the experiment
                run_single_experiment(single_config, results_base_dir=results_dir)
                print()

        print("=" * 60)
        print(f"All {total_experiments} experiments complete! Results saved to {base_results_dir}")
        print("=" * 60)
    else:
        # Single experiment configuration
        if "model" not in config or "cipher" not in config:
            raise ValueError(
                "Configuration must have either 'matrix' (for multiple experiments) "
                "or both 'model' and 'cipher' (for single experiment)"
            )
        run_single_experiment(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
