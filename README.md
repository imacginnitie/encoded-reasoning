# Encoded Reasoning Experiments

Few-shot prompting experiments with different encoding schemes on reasoning tasks.

## Setup

1. Install dependencies: `uv sync`
2. Set up API keys: Copy `.env.example` to `.env` and add your keys
3. Configure experiment: Copy `config.yaml.example` to `config.yaml` and edit

## Project Structure

- `config.yaml` - Experiment configuration
- `config/examples/` - Example configs (caesar, base64, emojispeak, chinese)
- `src/encoded_reasoning/` - Main code
  - `run_experiment.py` - Run experiments
  - `eval.py` - Evaluate results
  - `ciphers.py` - Encoding schemes
  - `dataset.py` - Dataset loading
  - `adherence.py` - Adherence checking
  - `plotting/plot_results.py` - Plotting

## Usage

**Note**: Uses few-shot prompting with pre-trained LLMs (no training/fine-tuning).

1. **Run experiment**: `bash run_experiment.sh` (or `bash run_experiment.sh custom_config.yaml`)
2. **Evaluate**: Copy `config/eval_config.yaml.example` to `config/eval_config.yaml`, set `results_dir`, then `bash eval_and_plot.sh`

Results are saved in `results/` with timestamps.

## Encoding Types

**Programmatic** (auto-encoded examples):
- `caesar` - Caesar cipher (supports `shift` parameter)
- `base64` - Base64 encoding

**Prompt-based** (model follows instructions):
- `emojispeak` - Think using only emojis
- `chinese` - Think in Chinese
- `pinyin` - Think in Pinyin

For prompt encodings, pre-made encoded examples in `data/encoded_examples/` improve results.
