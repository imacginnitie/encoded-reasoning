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


def estimate_input_tokens(messages_or_prompt, provider: str) -> int:
    """Estimate the number of input tokens for a prompt.

    Uses a rough approximation: ~4 characters per token for Anthropic,
    ~3.5 characters per token for OpenAI.

    Args:
        messages_or_prompt: For Anthropic, a list of message dicts.
            For OpenAI, a string prompt.
        provider: "openai" or "anthropic"

    Returns:
        Estimated number of input tokens
    """
    if provider == "anthropic":
        if isinstance(messages_or_prompt, str):
            text = messages_or_prompt
        else:
            # Sum up all message content
            text = ""
            for msg in messages_or_prompt:
                if isinstance(msg.get("content"), str):
                    text += msg["content"]
                elif isinstance(msg.get("content"), list):
                    # Handle content blocks
                    for block in msg["content"]:
                        if isinstance(block, dict) and "text" in block:
                            text += block["text"]
        # Rough estimate: ~4 characters per token for Anthropic
        return int(len(text) / 4)
    else:
        # OpenAI
        if isinstance(messages_or_prompt, str):
            text = messages_or_prompt
        else:
            text = ""
            for msg in messages_or_prompt:
                if isinstance(msg.get("content"), str):
                    text += msg["content"]
        # Rough estimate: ~3.5 characters per token for OpenAI
        return int(len(text) / 3.5)


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def call_llm(messages_or_prompt, provider: str, model: str, api_key: str, max_retries: int = 5):
    """Call LLM API with retries.

    Args:
        messages_or_prompt: For Anthropic, a list of message dicts. For OpenAI, a string prompt.
        provider: "openai" or "anthropic"
        model: Model name
        api_key: API key
        max_retries: Maximum number of retries
    """
    for attempt in range(max_retries):
        try:
            if provider == "openai":
                client = openai.OpenAI(api_key=api_key)
                # OpenAI expects messages format
                if isinstance(messages_or_prompt, str):
                    messages = [{"role": "user", "content": messages_or_prompt}]
                else:
                    # Type cast for OpenAI messages
                    messages = messages_or_prompt  # type: ignore
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore
                    temperature=0.0,
                )
                return response.choices[0].message.content or ""
            elif provider == "anthropic":
                client = anthropic.Anthropic(api_key=api_key)
                # Anthropic expects messages format
                if isinstance(messages_or_prompt, str):
                    messages = [{"role": "user", "content": messages_or_prompt}]
                else:
                    # Type cast for Anthropic messages
                    messages = messages_or_prompt  # type: ignore
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
                return "".join(text_parts)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except anthropic.RateLimitError as e:
            # Anthropic rate limit: wait longer (at least 60s to reset minute window)
            if attempt < max_retries - 1:
                # Wait 60 seconds + exponential backoff for rate limits
                wait_time = 60 + (2**attempt) * 10
                print(f"Rate limit error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Waiting {wait_time}s for rate limit to reset...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            # Check for rate limit errors in string representation (for OpenAI or other cases)
            is_rate_limit = "rate limit" in str(e).lower() or "429" in str(e)
            if attempt < max_retries - 1:
                if is_rate_limit:
                    # For rate limits, wait longer
                    wait_time = 60 + (2**attempt) * 10
                else:
                    # For other errors, use shorter exponential backoff
                    wait_time = 2**attempt
                print(f"Error (attempt {attempt + 1}/{max_retries}): {e}. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError("Failed after retries")


def format_prompt(examples, encoding_scheme, provider="anthropic"):
    """Format few-shot prompt as messages (for Anthropic) or text (for OpenAI)."""
    messages = []

    # Add system message with instructions if present
    if "instruction" in encoding_scheme:
        messages.append({"role": "user", "content": encoding_scheme["instruction"]})
        messages.append(
            {
                "role": "assistant",
                "content": "I understand. I will answer in the format 'Answer: [ANSWER]'.",
            }
        )

    # Add few-shot examples as user→assistant pairs
    for ex in examples:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})

    if provider == "openai":
        # Convert messages to text format for OpenAI
        parts = []
        if "instruction" in encoding_scheme:
            parts.append(f"Instructions: {encoding_scheme['instruction']}\n")
        parts.append("Examples:")
        for i in range(0, len(messages), 2):
            if messages[i]["role"] == "user":
                parts.append(f"Input: {messages[i]['content']}")
                if i + 1 < len(messages):
                    parts.append(f"Output: {messages[i + 1]['content']}\n")
        return "\n".join(parts)
    else:
        # Return messages list for Anthropic
        return messages


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

        n_train = int(len(all_examples) * config["dataset"].get("train_split", 0.7))
        train_examples = all_examples[:n_train]
        test_examples = all_examples[n_train : n_train + config["dataset"]["num_test_examples"]]

        # Sample few-shot examples
        few_shot_examples = random.sample(train_examples, min(k, len(train_examples)))

        # Process examples
        few_shot_processed = []
        for ex in few_shot_examples:
            problem = ex["problem"]
            solution = ex.get("solution", "")
            answer = extract_answer_from_boxed(ex.get("answer", ""))

            # For direct/filler schemes, use "Answer: <number>" format
            if scheme.get("is_direct", False):
                if scheme.get("is_programmatic", True):
                    encoded_problem = scheme["encode"](problem)
                    few_shot_processed.append(
                        {
                            "input": encoded_problem,
                            "output": f"Answer: {answer}" if answer else "Answer:",
                            # Store original for filler token generation
                            "original_problem": problem,
                        }
                    )
                else:
                    few_shot_processed.append(
                        {
                            "input": problem,
                            "output": f"Answer: {answer}" if answer else "Answer:",
                            "original_problem": problem,
                        }
                    )
            else:
                # For other schemes, use the original format
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
                            # Store original for filler token generation
                            "original_problem": problem,
                        }
                    )
                else:
                    few_shot_processed.append(
                        {
                            "input": problem,
                            "output": (f"{solution}\n\\boxed{{{answer}}}" if answer else solution),
                            "original_problem": problem,
                        }
                    )
    else:
        # If using pre-made examples, still need to load test examples
        all_examples = load_math500_dataset(cache_dir="data/cache")
        random.seed(42)
        random.shuffle(all_examples)
        n_train = int(len(all_examples) * config["dataset"].get("train_split", 0.7))
        test_examples = all_examples[n_train : n_train + config["dataset"]["num_test_examples"]]

    # For filler ciphers, append filler tokens to inputs dynamically
    if scheme_name == "filler" and "filler" in config["cipher"]:
        filler_config = config["cipher"]["filler"]
        filler_type = filler_config.get("type", "counting")
        filler_count = filler_config.get("count", 300)

        for ex in few_shot_processed:
            # Use original problem for filler token generation (not encoded input)
            original_problem = ex.get("original_problem", ex["input"])
            filler_tokens = generate_filler_tokens(
                filler_type,
                filler_count,
                problem=original_problem if filler_type == "repeat" else None,
                repeat_string=filler_config.get("string", None),
            )
            ex["input"] = f"{ex['input']}\n\n{filler_tokens}"

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

    # Create prompt (text format for saving to file)
    prompt_text = format_prompt(few_shot_processed, scheme, provider="openai")
    if isinstance(prompt_text, str):
        with open(output_dir / "few_shot_prompt.txt", "w") as f:
            f.write(prompt_text)

    return test_processed, few_shot_processed, scheme


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
    test_examples, few_shot_processed, scheme = process_dataset(config)

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

        # Build prompt/messages based on provider
        if provider == "anthropic":
            # Build messages with few-shot examples and prefill
            messages_list = format_prompt(few_shot_processed, scheme, provider="anthropic")
            # Type assertion: format_prompt returns list for anthropic
            if not isinstance(messages_list, list):
                raise ValueError("Expected list of messages for Anthropic provider")
            messages = messages_list
            # Add test problem as user message
            messages.append({"role": "user", "content": example["input"]})
            # Add prefill assistant message to force "Answer:" start
            messages.append({"role": "assistant", "content": "Answer:"})
            prompt_for_logging = f"Messages with {len(messages)} total messages"

            # Estimate tokens and warn if very large
            estimated_tokens = estimate_input_tokens(messages, provider)
            if estimated_tokens > 20000:
                print(
                    f"\nWarning: Large prompt detected (~{estimated_tokens} estimated tokens). "
                    f"This may hit rate limits. Consider reducing num_examples or filler count."
                )

            response = call_llm(messages, provider, model, api_key)
        else:
            # OpenAI: build text prompt and append "\n\nAnswer:"
            prompt_text = format_prompt(few_shot_processed, scheme, provider="openai")
            prompt = f"{prompt_text}\n\nInput: {example['input']}\n\nAnswer:"
            prompt_for_logging = prompt

            # Estimate tokens and warn if very large
            estimated_tokens = estimate_input_tokens(prompt, provider)
            if estimated_tokens > 20000:
                print(
                    f"\nWarning: Large prompt detected (~{estimated_tokens} estimated tokens). "
                    f"This may hit rate limits. Consider reducing num_examples or filler count."
                )

            response = call_llm(prompt, provider, model, api_key)

        try:
            # Handle "Answer:" prefix - strip it if present
            if response.startswith("Answer:"):
                response = response[7:].strip()  # Remove "Answer:" and leading whitespace

            results["responses"].append(
                {
                    "example_index": i - 1,
                    "input": example["input"],
                    "expected_answer": example.get("expected_answer", ""),
                    "original_problem": example.get("original_problem", ""),
                    "response": response,
                    "prompt": prompt_for_logging,
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

        # Add adaptive delay between requests to avoid hitting rate limits
        # Anthropic has a 30,000 input tokens/minute limit
        if provider == "anthropic":
            # Calculate delay based on estimated tokens to stay under 30k/min
            # If we use ~estimated_tokens per request, we need at least:
            # delay = (estimated_tokens / 30000) * 60 seconds
            # Add a safety margin of 20%
            if estimated_tokens > 0:
                min_delay = (estimated_tokens / 30000) * 60 * 1.2
                delay = max(2.0, min_delay)  # At least 2 seconds, but more if needed
            else:
                delay = 2.0
            time.sleep(delay)
        else:
            time.sleep(0.5)  # Smaller delay for other providers

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

                # For filler ciphers, include the filler subtype and count in the directory name
                if cipher_type == "filler" and "filler" in cipher_config:
                    filler_subtype = cipher_config["filler"].get("type", "unknown")
                    filler_count = cipher_config["filler"].get("count", "unknown")
                    cipher_dir_name = f"{cipher_type}_{filler_subtype}_{filler_count}"
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
