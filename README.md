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

## Lambda (GH200) full run

From a new GH200 SSH session:

### GPU sanity check (do this first)

Run these immediately after SSH and before install:

```bash
nvidia-smi

python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
```

If torch shows `+cpu` / `cuda in torch: None` / `device count: 0`, reinstall CUDA torch:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --no-deps
```

Then re-run the check above before continuing.


1) Start workspace and clone repo

```bash
git clone https://github.com/tractatus1889/metaexamples.git
cd metaexamples
```

2) Create and activate a venv, install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3) Verify GPU

```bash
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "none")
PY
```

4) Select 5 tokens for the synthetic alphabet

```bash
python scripts/select_tokens.py \
  --model-id allenai/OLMo-1B-hf \
  --prefer-non-ascii \
  --ascii-fallback
```

This writes `data/tokens/selected_alphabet.json`.

5) Generate train/eval corpora for all grammars

```bash
python scripts/generate_data.py \
  --token-file data/tokens/selected_alphabet.json \
  --grammars g1,g2,g3 \
  --n-train 10000 \
  --n-valid 1000 \
  --n-test 1000 \
  --n-invalid-eval 1000
```

6) Smoke train + eval (single grammar condition)

```bash
python scripts/train.py \
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

7) Validate smoke model

```bash
python scripts/evaluate_perplexity.py --model checkpoints/olmo-1b_g1_smoke/final --grammar g1 --split test
python scripts/evaluate_generation.py --model checkpoints/olmo-1b_g1_smoke/final --grammar g1 --n-samples 500
```

8) Full experiment matrix (all grammars + all conditions)

```bash
python scripts/run_experiment.py \
  --model-id allenai/OLMo-1B-hf \
  --grammars g1,g2,g3 \
  --conditions examples,meta_1pct,meta_5pct,meta_10pct \
  --max-steps 2000 \
  --eval-split test
```

`run_experiment.py` will:
- regenerate corpora if needed,
- train each `(grammar, condition)` run with `--eval-data` on `data/eval/<grammar>_<split>`,
- run `scripts/evaluate_perplexity.py`,
- run `scripts/evaluate_generation.py` with `--n-samples 500`.

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
