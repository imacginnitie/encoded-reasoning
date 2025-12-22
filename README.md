# Encoded Reasoning Experiments

Experiments on using few-shot prompting to teach models different ciphers and evaluating their performance on reasoning tasks.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for package and environment management.

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Set up environment variables:
   Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   ```

## Project Structure

```
encoded-cot/
├── .env                    # API keys (gitignored)
├── config.yaml             # Experiment configuration
├── config/
│   └── eval_config.yaml    # Evaluation/plotting configuration
├── data/                   # Datasets (gitignored)
├── results/                # Experiment results (gitignored)
├── run_experiment.sh       # Run an experiment
├── eval_and_plot.sh        # Evaluate and plot results
└── src/
    └── encoded_reasoning/
        ├── ciphers.py          # Programmatic cipher implementations
        ├── prompt_encodings.py # Prompt-based encoding instructions
        ├── encoding.py         # Unified interface for encodings
        ├── process_dataset.py  # Process and encode dataset
        ├── run_experiment.py   # Run LLM experiments
        ├── eval.py             # Evaluate results
        └── plotting/
            └── plot_results.py # Generate plots
```

## Running Experiments

1. **Configure experiment**: Edit `config.yaml` to set:
   - Model provider and name
   - Encoding type (see below)
   - Number of few-shot examples
   - Number of test examples

## Encoding Types

The system supports two types of encodings:

### Programmatic Ciphers
These encode/decode text programmatically:
- `caesar` - Caesar cipher (supports `shift` parameter in `params`)

### Prompt Encodings
These provide instructions for the model to follow:
- `emojispeak` - Think using only emojis
- `chinese` - Think in Chinese (LaTeX kept as is)
- `pinyin` - Think in Pinyin (LaTeX kept as is)

For programmatic ciphers, examples are encoded before being shown to the model. For prompt encodings, examples are shown unencoded and the model follows the instruction to encode its thinking.

2. **Run experiment**:
   ```bash
   bash run_experiment.sh
   ```
   Or with a custom config:
   ```bash
   bash run_experiment.sh custom_config.yaml
   ```

3. **Evaluate and plot**:
   - Update `config/eval_config.yaml` with the results directory path
   - Run:
     ```bash
     bash eval_and_plot.sh
     ```

## Development

- Format code: `uv run ruff format .`
- Lint code: `uv run ruff check .`
- Type check: `uv run pyright`

## Notes

- All plots should include error bars
- Results are saved with timestamps in `results/`
- Each experiment saves its config alongside results for reproducibility

