"""Run experiment: few-shot prompting with pre-trained LLMs (no training)."""

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
from tqdm import tqdm

load_dotenv()

from encoded_reasoning.ciphers import get_encoding_scheme
from encoded_reasoning.dataset import (
    extract_answer_from_boxed,
    load_encoded_examples,
    load_math500_dataset,
    save_processed_dataset,
)


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def call_llm(prompt: str, provider: str, model: str, api_key: str, max_retries: int = 5):
    """Call LLM API with retries."""
    if not api_key:
        raise ValueError(f"API key is empty for {provider}. Check your .env file.")

    for attempt in range(max_retries):
        try:
            if provider == "openai":
                client = openai.OpenAI(api_key=api_key, timeout=180.0)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return response.choices[0].message.content or ""
            elif provider == "anthropic":
                client = anthropic.Anthropic(api_key=api_key, timeout=180.0)
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                text_parts = []
                for block in response.content:
                    text = getattr(block, "text", None)
                    if text is not None:
                        text_parts.append(text)
                return "".join(text_parts)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt if "rate limit" in str(e).lower() or "429" in str(e) else 1
                time.sleep(wait_time)
            else:
                raise


def format_prompt(examples, encoding_scheme, include_instructions=True, has_encoded_examples=False):
    """Format few-shot prompt."""
    parts = []

    if include_instructions:
        if encoding_scheme.get("is_programmatic", True):
            parts.append("Follow the examples below. Use the same encoding for your reasoning.\n")
        else:
            instruction = encoding_scheme.get("instruction", "")
            if has_encoded_examples:
                parts.append(f"Instructions: {instruction}\n")
            else:
                parts.extend(
                    [
                        "=" * 60,
                        "CRITICAL INSTRUCTIONS:",
                        instruction,
                        "=" * 60,
                        "",
                        "The examples below show the problem structure, but YOU MUST follow "
                        "the encoding instruction above for YOUR reasoning. "
                        "Use the encoding for all your thinking/reasoning steps, "
                        "but keep the final answer in \\boxed{} unencoded.",
                        "",
                    ]
                )

    parts.append("Examples:")
    for ex in examples:
        parts.append(f"Input: {ex['input']}")
        parts.append(f"Output: {ex['output']}\n")

    return "\n".join(parts)


def _check_overlap(set1, set2, name1, name2, allow_warning=False):
    """Check for overlap between two sets of problems."""
    overlap = set1 & set2
    if overlap:
        msg = (
            f"Overlap detected between {name1} and {name2}! "
            f"Found {len(overlap)} overlapping problems."
        )
        if allow_warning:
            print(f"⚠️  Warning: {msg}")
            print("   This could lead to data leakage. Consider removing overlapping examples.")
        else:
            raise ValueError(msg)
    return len(overlap) == 0


def _process_example(ex, scheme, for_few_shot=True):
    """Process a single example for few-shot or test set."""
    problem = ex.get("problem", ex.get("input", ""))
    solution = ex.get("solution", "")
    raw_answer = ex.get("answer", "")
    answer = extract_answer_from_boxed(raw_answer) or (raw_answer.strip() if raw_answer else "")

    is_programmatic = scheme.get("is_programmatic", True)

    if for_few_shot:
        # Few-shot examples need input and output
        if is_programmatic:
            input_text = scheme["encode"](problem)
            output_text = scheme["encode"](solution) if solution else ""
            if answer:
                output_text += f"\n\\boxed{{{scheme['encode'](answer)}}}"
            return {"input": input_text, "output": output_text}
        else:
            output_text = solution if solution else ""
            if answer:
                output_text += f"\n\\boxed{{{answer}}}" if output_text else f"\\boxed{{{answer}}}"
            return {"input": problem, "output": output_text}
    else:
        # Test examples need input, expected_answer, and original_problem
        input_text = scheme["encode"](problem) if is_programmatic else problem
        return {
            "input": input_text,
            "expected_answer": answer,
            "original_problem": problem,
        }


def process_dataset(config):
    """Process dataset and create prompts."""
    scheme_name = config["cipher"]["type"]
    scheme = get_encoding_scheme(scheme_name, **config["cipher"].get("params", {}))

    random.seed(42)
    train_examples = load_math500_dataset(cache_dir="data/cache", split="train")
    test_examples_full = load_math500_dataset(cache_dir="data/cache", split="test")

    num_test = config["dataset"]["num_test_examples"]
    random.shuffle(test_examples_full)
    test_examples = test_examples_full[:num_test]

    test_problems = {ex["problem"] for ex in test_examples}
    _check_overlap({ex["problem"] for ex in train_examples}, test_problems, "train", "test")

    k = config["prompt"]["num_examples"]
    encoded_examples = load_encoded_examples(scheme_name, k)
    has_encoded_examples = encoded_examples is not None

    if encoded_examples:
        encoded_problems = {ex.get("input", "") for ex in encoded_examples if ex.get("input")}
        overlap = encoded_problems & test_problems
        if overlap:
            print(f"⚠️  Warning: Overlap detected between pre-made examples and test! Found {len(overlap)} overlapping problems.")
            print("   Removing overlapping problems from test set to prevent data leakage.")
            # Filter out overlapping problems from test set
            test_examples = [ex for ex in test_examples if ex["problem"] not in overlap]
            test_problems = {ex["problem"] for ex in test_examples}
            if len(test_examples) < num_test:
                print(f"   Note: Test set now has {len(test_examples)} examples (requested {num_test}).")
        few_shot_processed = encoded_examples
    else:
        few_shot_examples = random.sample(train_examples, min(k, len(train_examples)))
        few_shot_problems = {ex["problem"] for ex in few_shot_examples}
        _check_overlap(few_shot_problems, test_problems, "few-shot examples", "test")
        few_shot_processed = [
            _process_example(ex, scheme, for_few_shot=True)
            for ex in tqdm(few_shot_examples, desc="Processing few-shot examples", leave=False)
        ]

    test_processed = [
        _process_example(ex, scheme, for_few_shot=False)
        for ex in tqdm(test_examples, desc="Processing test examples", leave=False)
    ]

    output_dir = Path("data/processed") / scheme_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_processed_dataset(few_shot_processed, output_dir / "few_shot.jsonl")
    save_processed_dataset(test_processed, output_dir / "test.jsonl")

    prompt = format_prompt(
        few_shot_processed,
        scheme,
        config["prompt"].get("include_instructions", True),
        has_encoded_examples=has_encoded_examples,
    )
    (output_dir / "few_shot_prompt.txt").write_text(prompt)

    return test_processed, prompt


def run_experiment(config):
    """Run few-shot prompting experiment (no model training)."""
    provider = config["model"]["provider"]
    model = config["model"]["name"]

    api_keys = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    if provider not in api_keys:
        raise ValueError(f"Unknown provider: {provider}")

    api_key = os.getenv(api_keys[provider])
    if not api_key:
        raise ValueError(
            f"API key not found for {provider}. Set {api_keys[provider]} in your .env file."
        )

    test_examples, few_shot_prompt = process_dataset(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config["experiment"]["output_dir"]) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.yaml").write_text(yaml.dump(config))

    results = {"timestamp": timestamp, "config": config, "responses": []}

    print(f"Running experiment on {len(test_examples)} examples...")
    for i, example in enumerate(tqdm(test_examples, desc="Processing examples", unit="example"), 1):
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
                (results_dir / "results_intermediate.json").write_text(
                    json.dumps(results, indent=2)
                )
        except Exception as e:
            results["responses"].append(
                {
                    "example_index": i - 1,
                    "input": example["input"],
                    "response": None,
                    "error": str(e),
                }
            )

    (results_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Experiment complete! Results saved to {results_dir}")
    print("\nTo evaluate and plot results, run:")
    print(f"  bash eval_and_plot.sh {results_dir}")
    print("\nOr separately:")
    print(f"  uv run python -m encoded_reasoning.eval --results-dir {results_dir}")
    print(f"  uv run python -m encoded_reasoning.plotting.plot_results --results-dir {results_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
