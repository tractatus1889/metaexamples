# metaexamples

Metaexamples study for learning grammar rules from synthetic supervision.
Train OLMO with wrapped grammar examples (`<g1>...</g1>`, `<g2>...</g2>`, `<g3>...</g3>`) and evaluate the ability to discriminate valid vs invalid strings.

## Grammars

- `g1`: non-empty sequence over 5 symbols with max length 12.
- `g2`: `g1` plus every symbol appears an even number of times.
- `g3`: palindrome plus every symbol appears an even number of times.

## Fast, clean setup (recommended)

```bash
git clone https://github.com/tractatus1889/metaexamples.git
cd metaexamples

bash scripts/bootstrap_clean_env.sh --fresh .venv_metaexamples
source .venv_metaexamples/bin/activate
```

## Minimal verification

```bash
TRANSFORMERS_NO_TF=1 TRANSFORMERS_NO_TORCHVISION=1 python3 - <<'PY'
import os
import sys
import numpy
import torch
import transformers
from transformers import Trainer, TrainingArguments

print("python:", sys.executable)
print("no_tf:", os.getenv("TRANSFORMERS_NO_TF"))
print("no_torchvision:", os.getenv("TRANSFORMERS_NO_TORCHVISION"))
print("numpy:", numpy.__version__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("trainer import:", "ok")
print("args import:", "ok")
PY
```

## Required commands

1) Select the 5-symbol alphabet

```bash
python3 scripts/select_tokens.py \
  --model-id allenai/OLMo-1B-hf \
  --prefer-non-ascii \
  --ascii-fallback
```

2) Generate synthetic corpora and wrapped eval files

```bash
python3 scripts/generate_data.py \
  --token-file data/tokens/selected_alphabet.json \
  --grammars g1,g2,g3 \
  --n-train 10000 \
  --n-valid 1000 \
  --n-test 1000 \
  --n-invalid-eval 1000
```

`generate_data.py` writes:
- `data/corpora/{g}_examples.jsonl`
- `data/corpora/{g}_metaexamples_1pct.jsonl`
- `data/corpora/{g}_metaexamples_5pct.jsonl`
- `data/corpora/{g}_metaexamples_10pct.jsonl`
- wrapped eval files in `data/eval/` (use these for all eval runs)

3) Smoke train (single condition)

```bash
python3 scripts/train.py \
  --model-id allenai/OLMo-1B-hf \
  --corpus data/corpora/g1_examples.jsonl \
  --run-name olmo-1b_g1_smoke \
  --output-dir checkpoints \
  --mix-ratio 0.1 \
  --max-steps 200 \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --warmup-steps 100 \
  --canonical-dataset allenai/c4 \
  --eval-data data/eval/g1_test_valid_wrapped.txt \
  --eval-steps 50 \
  --train-logging-steps 25 \
  --save-steps 50
```

This uses the default HF canonical stream (`allenai/c4`) for non-synthetic examples.

If you want to force a non-streaming canonical load:

```bash
python3 scripts/train.py ... --no-canonical-streaming
```

4) Full matrix training + eval

```bash
python3 scripts/run_experiment.py \
  --model-id allenai/OLMo-1B-hf \
  --grammars g1,g2,g3 \
  --conditions examples,meta_1pct,meta_5pct,meta_10pct \
  --canonical-dataset allenai/c4 \
  --max-steps 2000 \
  --eval-split test \
  --results-dir results
```

5) Evaluate a trained run

```bash
python3 scripts/evaluate_perplexity.py \
  --model checkpoints/olmo-1b_g1_smoke/final \
  --grammar g1 \
  --split test

python3 scripts/evaluate_generation.py \
  --model checkpoints/olmo-1b_g1_smoke/final \
  --grammar g1 \
  --token-file data/tokens/selected_alphabet.json \
  --n-samples 5000
```

## What changed from the old pipeline

The code no longer materializes canonical corpora or mixed synthetic+canonical data files.
- Do not generate `data/canonical/*.txt` for this workflow.
- Do not pass materialization flags to `generate_data.py`, `run_experiment.py`, or `train.py`.
- Canonical data is mixed from HF at train time, while synthetic corpora come from `data/corpora/*.jsonl`.

## Useful flags

- `--mix-ratio 0.1` (`train.py`): synthetic fraction in mixed training stream.
- `--canonical-dataset`: HF dataset id (or local file path if desired) for canonical text.
- `--canonical-config`: HF dataset config (`en` for `allenai/c4`).
- `--canonical-text-key`: field used from canonical HF dataset.
- `--canonical-streaming` / `--no-canonical-streaming`: control canonical loading mode.
- `--conditions examples,meta_1pct,meta_5pct,meta_10pct`: experiment condition matrix in `run_experiment.py`.
- `--run-only g1`: limit experiment to one grammar.
- `--train-only` / `--eval-only`: split phases.
- `--eval-steps`: validation interval in training.
- `--save-steps`: checkpoint interval.
- `--train-logging-steps`: train metrics logging interval.
- `--logging-steps`: alias for `--train-logging-steps`.
- `--eval-at-step-0`: run one eval pass before first update.
- `--learning-rate`: training learning rate.
- `--gradient-accumulation-steps`: gradient accumulation.
- `--warmup-steps`: LR warmup.
- `--max-seq-length`: tokenization sequence cap.
- `--results-dir`: output directory for eval artifacts.
- `--generation-n-samples`: number of generated samples for validity check.
- `--max-generation-length`: max tokens generated per example.
- `--perplexity-max-length`: tokenization max length for perplexity eval.
- `--metrics-log`: path to save train/eval logs in JSONL.
- `--save-samples` (`evaluate_generation.py`): include raw generated rows in the output JSON.
- `--seed-with-symbol` (`evaluate_generation.py`): use prompts like `<gX> <valid_symbol>`.
- `--trust-remote-code` / `--no-trust-remote-code`: control HF loading.

## Notes

- `Writing model shards` is normal checkpointing output from Hugging Face; each checkpoint is serialized into multiple shard files under `checkpoints/<run-name>/`.

## Troubleshooting (short)

- Use `python3` (not `/usr/bin/python3`) for all commands.
- `torchvision` is not required. If you get vision-related import failures, uninstall `torchvision`.
- Keep env vars set for this repo:
  `TRANSFORMERS_NO_TF=1` and `TRANSFORMERS_NO_TORCHVISION=1`.
- If HF is slow/unthrottled, set `HF_TOKEN` in the environment.
