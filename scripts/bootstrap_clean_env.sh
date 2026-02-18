#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "--fresh" ]]; then
  FRESH=1
  shift
else
  FRESH=0
fi

VENV_DIR="${1:-.venv_metaexamples}"
CUDA_WHEEL_URL="${CUDA_WHEEL_URL:-https://download.pytorch.org/whl/cu121}"
MODEL_ID="${MODEL_ID:-allenai/OLMo-1B-hf}"

if [[ "$FRESH" == "1" ]]; then
  if [ -d "$VENV_DIR" ]; then
    echo "Removing existing venv: $VENV_DIR"
    rm -rf "$VENV_DIR"
  fi
fi

if [ -d "$VENV_DIR" ]; then
  echo "Using existing venv: $VENV_DIR"
else
  echo "Creating venv: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
export TRANSFORMERS_NO_TF=1

echo "Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "Installing PyTorch + CUDA wheels"
python -m pip install torch torchvision --index-url "$CUDA_WHEEL_URL"

echo "Installing project dependencies"
python -m pip install -r requirements.txt

echo "Repairing ABI-sensitive scientific stack"
python -m pip install --force-reinstall \
  "numpy==1.26.4" \
  "scipy>=1.11.4" \
  "scikit-learn>=1.4.2" \
  "ml-dtypes<0.5" \
  "transformers>=4.40" \
  "datasets>=2.18" \
  "accelerate>=0.30" \
  "fsspec<=2025.10.0,>=2023.1.0"

echo "Running dependency checks"
python -m pip check

echo "Verifying core imports"
python - <<'PY'
import os
import sys

import numpy
import torch
import transformers
from transformers import Trainer, TrainingArguments

print("python:", sys.executable)
print("numpy:", numpy.__version__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("transformers.no_tf:", os.environ.get("TRANSFORMERS_NO_TF"))
print("trainer:", "ok")
PY

if [ ! -f "data/tokens/selected_alphabet.json" ]; then
  echo "Running smoke token selection"
  python scripts/select_tokens.py \
    --model-id "$MODEL_ID" \
    --prefer-non-ascii \
    --ascii-fallback
else
  echo "Token file already exists; skipping smoke token selection"
fi

echo "Bootstrap complete."
echo "Activate with: source $VENV_DIR/bin/activate"
