# Encoded Reasoning Experiments

Few-shot prompting experiments studying how different encoding/cipher schemes affect chain-of-thought reasoning in LLMs. Tests whether encoding reasoning steps (emoji, Chinese, Caesar cipher, base64, filler tokens, etc.) impacts accuracy on math and science benchmarks.

## Datasets

- **MATH-500** — 500 competition math problems
- **AIME 2025** — 30 American Invitational Mathematics Examination problems
- **GPQA Diamond** — 198 graduate-level science multiple-choice questions

## Setup

1. Install dependencies: `uv sync`
2. Set up API keys: Copy `.env.example` to `.env` and add your keys (set `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`)
3. Configure experiment: Copy `config.yaml.example` to `config.yaml` and edit

## Project Structure

- `config.yaml` - Experiment configuration
- `config/examples/` - Example configs for different cipher types
- `data/encoded_examples/` - Pre-made few-shot examples (generic and dataset-specific)
- `src/encoded_reasoning/` - Main code
  - `run_experiment.py` - Run experiments
  - `eval.py` - Evaluate results
  - `ciphers.py` - Encoding schemes
  - `dataset.py` - Dataset loading (MATH-500, AIME 2025, GPQA Diamond)
  - `adherence.py` - Adherence checking
  - `plotting/plot_results.py` - Plotting and visualization

## Usage

**Note**: Uses few-shot prompting with pre-trained LLMs (no training/fine-tuning).

### Running Experiments

1. **Single experiment**: `bash run_experiment.sh` (or `bash run_experiment.sh custom_config.yaml`)
2. **Matrix experiments**: Configure multiple models and ciphers in `config.yaml` using the `matrix` section to run all combinations automatically

### Evaluating Results

1. **Single experiment**: `bash eval_and_plot.sh results/<timestamp>/<dataset>/<provider>/<model>/<cipher>`
2. **All experiments in timestamp**: `bash eval_and_plot.sh results/<timestamp>` (automatically processes all experiments and generates matrix plot)

Results are saved in `results/<timestamp>/<dataset>/<provider>/<model>/<cipher>/`.

## Configuration

The experiment supports two configuration modes:

### Option 1: Single Model and Cipher
```yaml
model:
  provider: "anthropic"  # or "openai"
  name: "claude-sonnet-4-5"

cipher:
  type: "emojispeak"
  params: {}
```

### Option 2: Matrix Configuration (Recommended)
Run all combinations of models × ciphers automatically:
```yaml
matrix:
  models:
    - provider: "anthropic"
      name: "claude-sonnet-4-5"
    - provider: "openai"
      name: "gpt-4o"
  ciphers:
    - type: "identity"
      params: {}
    - type: "emojispeak"
      params: {}
```

### Dataset Selection
```yaml
dataset:
  name: "math500"  # or "aime2025" or "gpqa_diamond"
  num_test_examples: 50
```

## Encoding Types

### Baseline
- **`identity`** - No encoding (baseline for comparison, important for normalization)

### Programmatic Ciphers
Auto-encoded examples (encoding/decoding handled programmatically):
- **`caesar`** - Caesar cipher (supports `shift` parameter, e.g., `params: {shift: 3}`)
- **`base64`** - Base64 encoding

### Prompt-based Encodings
Model follows instructions to encode reasoning:
- **`emojispeak`** - Think using only emojis
- **`chinese`** - Think in Chinese
- **`pinyin`** - Think in Pinyin

For prompt encodings, pre-made encoded examples in `data/encoded_examples/` improve results. Dataset-specific examples (e.g., `data/encoded_examples/gpqa_diamond/`) are used when available, falling back to generic examples.

### Direct Answering Modes
No chain-of-thought reasoning, only final answer:
- **`direct`** - Direct answering without CoT (baseline for comparison with filler tokens)

### Filler Token Experiments
Direct answering with filler tokens inserted before the answer:
- **`filler`** - Filler token mode with configurable filler types:
  - `counting` - Counting sequence (1 2 3 ... N)
  - `lorem` - Lorem ipsum text
  - `baseline_neutral` - Neutral baseline instructional text
  - `positive_personal` - Positive personal reinforcement statements
  - `positive_general` - Positive general statements about problem difficulty
  - `negative_personal` - Negative personal statements
  - `negative_general` - Negative general statements about problem difficulty
  - `dots` - Repeating dots/strings
  - `repeat` - Repeating the problem text

Example filler configuration:
```yaml
cipher:
  type: "filler"
  params: {}
  filler:
    type: "counting"  # or "lorem", "baseline_neutral", etc.
    count: 300  # Number of filler tokens
```

## Evaluation and Plotting

The evaluation system provides:
- **Accuracy metrics**: Correct answers, adherence to encoding scheme
- **Normalized ratios**: Performance relative to identity baseline
- **Individual plots**: Per-experiment visualizations
- **Matrix plots**: Comparative visualizations across all experiments in a timestamp directory

When evaluating a timestamp directory, the system:
1. Processes all experiments (identity baselines first for ratio calculations)
2. Generates individual plots for each experiment
3. Creates a matrix plot comparing all model/cipher combinations

**Note**: The `identity` cipher is important for baseline comparisons and normalization (e.g., "# correct & adherent" / "# identity correct").
