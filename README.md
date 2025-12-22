# Encoded Reasoning Experiments

Simple experiments on using few-shot prompting to teach models different ciphers and evaluating their performance on reasoning tasks.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for package and environment management.

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Set up API keys:
   ```bash
   # Copy the example .env file
   cp .env.example .env
   
   # Edit .env and add your API keys
   # OPENAI_API_KEY=your_key_here
   # ANTHROPIC_API_KEY=your_key_here
   ```

3. Configure your experiment:
   ```bash
   # Copy the example config
   cp config.yaml.example config.yaml
   
   # Edit config.yaml with your desired settings
   ```

## Project Structure

```
encoded-reasoning/
├── .env                    # API keys (gitignored, copy from .env.example)
├── .env.example            # Example .env file
├── README.md
├── config.yaml             # Experiment configuration (copy from config.yaml.example)
├── config.yaml.example     # Example experiment config
├── config/
│   ├── eval_config.yaml    # Evaluation configuration (copy from eval_config.yaml.example)
│   ├── eval_config.yaml.example
│   └── examples/           # Example experiment configs
│       ├── caesar.yaml
│       ├── base64.yaml
│       ├── emojispeak.yaml
│       └── chinese.yaml
├── run_experiment.sh       # Run an experiment
├── eval_and_plot.sh        # Evaluate results
├── data/                   # Datasets (gitignored)
├── results/                # Experiment results (gitignored)
└── src/
    └── encoded_reasoning/
        ├── __init__.py
        ├── ciphers.py          # Encoding scheme implementations
        ├── dataset.py          # Dataset loading utilities
        ├── adherence.py        # Adherence checking utilities
        ├── run_experiment.py   # Main experiment runner
        ├── eval.py             # Evaluation script (accuracy + adherence)
        └── plotting/
            └── plot_results.py # Plotting utilities
```

## Running Experiments

1. **Configure experiment**: Edit `config.yaml` (or use one of the example configs in `config/examples/`):
   - Model provider and name
   - Encoding type (see below)
   - Number of few-shot examples
   - Number of test examples

   Example configs are available:
   - `config/examples/caesar.yaml` - Caesar cipher
   - `config/examples/base64.yaml` - Base64 encoding
   - `config/examples/emojispeak.yaml` - EmojiSpeak
   - `config/examples/chinese.yaml` - Chinese encoding

2. **Run experiment**:
   ```bash
   bash run_experiment.sh
   ```
   Or with a custom config:
   ```bash
   bash run_experiment.sh custom_config.yaml
   ```

3. **Evaluate results**:
   - Copy the example eval config:
     ```bash
     cp config/eval_config.yaml.example config/eval_config.yaml
     ```
   - Update `config/eval_config.yaml` with the results directory path (from the experiment output)
   - Run:
     ```bash
     bash eval_and_plot.sh
     ```
   - This will calculate both accuracy (correctness) and adherence (whether the model followed the encoding scheme)

## Encoding Types

### Programmatic Ciphers
These encode/decode text programmatically:
- `caesar` - Caesar cipher (supports `shift` parameter in `params`)
- `base64` - Base64 encoding

### Prompt Encodings
These provide instructions for the model to follow:
- `emojispeak` - Think using only emojis
- `chinese` - Think in Chinese (LaTeX kept as is)
- `pinyin` - Think in Pinyin (LaTeX kept as is)

For programmatic ciphers, examples are encoded before being shown to the model. For prompt encodings, examples are shown unencoded and the model follows the instruction to encode its thinking.

## Notes

- Results are saved with timestamps in `results/`
- Each experiment saves its config alongside results for reproducibility
