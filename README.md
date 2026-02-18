# metaexamples

Fresh OLMO-first synthetic grammar experiment focused on **metaexamples**.

This project intentionally does not depend on any legacy experiment directory.

## Core loop

1. Select 5 tokenizer tokens from OLMO as a synthetic alphabet.
2. Generate `g1`, `g2`, `g3` examples, metaexamples, and evaluation splits.
3. Train OLMO-1B on a canonical stream with synthetic mix.
4. Evaluate perplexity and sampled generation validity.

## Grammars

- `g1`: non-empty sequence over 5 symbols up to 12 symbols long.
- `g2`: `g1` plus each symbol appears an even number of times.
- `g3`: sequence is a palindrome and each symbol appears an even number of times.

Each grammar is wrapped with explicit tags in training data:
- `g1`: `<g1> ... </g1>`
- `g2`: `<g2> ... </g2>`
- `g3`: `<g3> ... </g3>`

## Token selection

`scripts/select_tokens.py` scans OLMO's tokenizer vocab and keeps only
single-token symbols that round-trip cleanly through encode/decode.
The default scoring prioritizes rare token IDs (as an approximation to low-frequency tokens) and avoids obvious word-like tokens.

## Install

- Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

## Quick start on Lambda (GH200)

These are recommended commands for the GH200 instance:

```bash
git clone https://github.com/tractatus1889/metaexamples.git
cd metaexamples
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("dtype:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "none")
PY
```

Generate full-size data (recommended before real runs):

```bash
python scripts/generate_data.py \
  --token-file data/tokens/selected_alphabet.json \
  --grammars g1,g2,g3 \
  --n-train 10000 \
  --n-valid 1000 \
  --n-test 1000 \
  --n-invalid-eval 1000
```

Smoke test one condition:

```bash
python scripts/run_experiment.py \
  --model-id allenai/OLMo-1B-hf \
  --run-only g1 \
  --conditions examples \
  --max-steps 200 \
  --eval-split test
```

Run full matrix:

```bash
python scripts/run_experiment.py \
  --model-id allenai/OLMo-1B-hf \
  --grammars g1,g2,g3 \
  --conditions examples,meta_1pct,meta_5pct,meta_10pct \
  --max-steps 2000 \
  --eval-split test
```

`run_experiment.py` now performs both:
- in-training evaluation (`--eval-data` with `--eval-split`), logging `eval_loss`.
- final `evaluate_perplexity.py` (PPL + discrimination metrics).
- final `evaluate_generation.py` (generation validity rate).

## Quick start (local)

1. Select symbol inventory:

```bash
python scripts/select_tokens.py --model-id allenai/OLMo-1B-hf
python scripts/select_tokens.py --model-id allenai/OLMo-1B-hf --prefer-non-ascii
python scripts/select_tokens.py --model-id allenai/OLMo-1B-hf --ascii-fallback
```

2. Generate data:

```bash
python scripts/generate_data.py
```

3. Train one configuration:

```bash
python scripts/train.py --model-id allenai/OLMo-1B-hf --corpus data/corpora/g1_examples.jsonl --max-steps 2000
```

To run in-training perplexity validation, pass an eval split file:

```bash
python scripts/train.py --model-id allenai/OLMo-1B-hf \
  --corpus data/corpora/g1_examples.jsonl \
  --eval-data data/eval/g1_test_valid.txt \
  --max-steps 2000 \
  --eval-steps 250
```

`Trainer` logs `eval_loss` at each eval step, which you can exponentiate to get perplexity for that checkpoint.

4. Evaluate:

```bash
python scripts/evaluate_perplexity.py --model checkpoints/olmo-1b_g1_examples/final --grammar g1
python scripts/evaluate_generation.py --model checkpoints/olmo-1b_g1_examples/final --grammar g1
```

## Full run helper

```bash
python scripts/run_experiment.py --model-id allenai/OLMo-1B-hf
```

This will generate data, train all selected conditions, and run generation validity by default.

Useful flags:

- `--synthetic-mix-ratio 0.1`: synthetic/canonical proportion in training stream.
- `--conditions examples,meta_1pct,meta_5pct,meta_10pct`: experiment matrix.
- `--run-only g1`: limit to one grammar.
- `--train-only` / `--eval-only`: split phases.
