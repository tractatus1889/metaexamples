# metaexamples

Metaexamples study for learning grammar rules from synthetic supervision:
train OLMO on synthetic examples and metaexamples for small token alphabets (`g1`, `g2`, `g3`) and evaluate validity discrimination.

## Grammars

- `g1`: non-empty sequence over 5 symbols with max length 12.
- `g2`: `g1` plus every symbol appears an even number of times.
- `g3`: palindrome plus every symbol appears an even number of times.

All training samples are wrapped with tags:
`<g1>...</g1>`, `<g2>...</g2>`, `<g3>...</g3>`.

## Core loop

1. Select 5 tokenizer tokens as symbolic alphabet.
2. Generate data and eval splits.
3. Train on canonical text + synthetic mix.
4. Evaluate with perplexity and sampled validity.

## Fast, clean setup (recommended)

From a fresh clone:

```bash
git clone https://github.com/tractatus1889/metaexamples.git
cd metaexamples
```

Create a fresh venv and install everything in one go:

```bash
bash scripts/bootstrap_clean_env.sh --fresh .venv_metaexamples
source .venv_metaexamples/bin/activate
```

What this script does:
- creates `.venv_metaexamples`
- installs CUDA PyTorch + project dependencies
- restores a stable NumPy/ML stack
- verifies core imports (`transformers`, `torch`, `Trainer`)
- optionally runs token selection once

If you prefer to keep your current venv, run without `--fresh`:

```bash
bash scripts/bootstrap_clean_env.sh
source .venv_metaexamples/bin/activate
```

## Minimal verification

Run this once after setup:

```bash
TRANSFORMERS_NO_TF=1 python - <<'PY'
import os, sys, numpy, torch, transformers
from transformers import Trainer, TrainingArguments

print("python:", sys.executable)
print("no_tf:", os.getenv("TRANSFORMERS_NO_TF"))
print("numpy:", numpy.__version__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("trainer import:", "ok")
PY
```

## Required workflow commands

1) Select symbolic alphabet

```bash
python3 scripts/select_tokens.py \
  --model-id allenai/OLMo-1B-hf \
  --prefer-non-ascii \
  --ascii-fallback
```

2) Generate corpora/eval splits

```bash
python3 scripts/generate_data.py \
  --token-file data/tokens/selected_alphabet.json \
  --grammars g1,g2,g3 \
  --n-train 10000 \
  --n-valid 1000 \
  --n-test 1000 \
  --n-invalid-eval 1000
```

3) Smoke train (single condition)

```bash
python3 scripts/train.py \
  --model-id allenai/OLMo-1B-hf \
  --corpus data/corpora/g1_examples.jsonl \
  --run-name olmo-1b_g1_smoke \
  --output-dir checkpoints \
  --mix-ratio 0.1 \
  --max-steps 200 \
  --eval-data data/eval/g1_test_valid.txt \
  --eval-steps 50 \
  --save-steps 50
```

4) Validate smoke model

```bash
python3 scripts/evaluate_perplexity.py --model checkpoints/olmo-1b_g1_smoke/final --grammar g1 --split test
python3 scripts/evaluate_generation.py --model checkpoints/olmo-1b_g1_smoke/final --grammar g1 --n-samples 500
```

5) Full matrix (all grammars + all conditions)

```bash
python3 scripts/run_experiment.py \
  --model-id allenai/OLMo-1B-hf \
  --grammars g1,g2,g3 \
  --conditions examples,meta_1pct,meta_5pct,meta_10pct \
  --max-steps 2000 \
  --eval-split test
```

`run_experiment.py` will train each `(grammar, condition)` pair and run perplexity + generation checks after each run.

## Useful flags

- `--synthetic-mix-ratio 0.1`: synthetic fraction in training stream.
- `--conditions examples,meta_1pct,meta_5pct,meta_10pct`: run matrix.
- `--run-only g1`: limit to one grammar.
- `--train-only` / `--eval-only`: split training and evaluation.
- `--eval-steps`: validation interval in training.
- `--save-steps`: checkpoint interval in training.

## Troubleshooting (short)

- Use `python3` (not `/usr/bin/python3`) for all commands.
- `torchvision` is not required for this project. If you see torch/vision version conflicts, uninstall vision with `pip uninstall -y torchvision`.
- If your environment is still broken, rerun bootstrap with `--fresh` and delete/recreate the venv.
