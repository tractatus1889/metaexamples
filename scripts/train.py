#!/usr/bin/env python3
"""
Train OLMO on canonical data mixed with synthetic grammar corpora.

Primary modes:
- examples-only: corpus=<g>_examples.jsonl
- metaexamples: corpus=<g>_metaexamples_<ratio>pct.jsonl
"""

from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import argparse
import inspect
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - optional path when config isn't needed
    yaml = None
import torch
from datasets import Dataset, IterableDataset, interleave_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train metaexamples grammar model")
    parser.add_argument("--config", type=str, help="YAML file with defaults")

    # Model
    parser.add_argument(
        "--model-id",
        default="allenai/OLMo-1B-hf",
        help="Base model id / path",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional model revision/checkpoint",
    )

    # Corpus
    parser.add_argument(
        "--corpus",
        default=None,
        help="JSONL corpus path for synthetic data",
    )
    parser.add_argument(
        "--canonical-dataset",
        default="allenai/c4",
        help="HF dataset for non-synthetic data",
    )
    parser.add_argument(
        "--canonical-config",
        default="en",
        help="HF dataset config for canonical corpus",
    )
    parser.add_argument(
        "--canonical-text-key",
        default="text",
        help="Field to tokenize in canonical dataset",
    )
    parser.add_argument(
        "--mix-ratio",
        type=float,
        default=0.10,
        help="Synthetic ratio in mixed training",
    )

    # Training
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--eval-data", default=None, help="Optional eval text file (one sample per line)")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="use_bf16", action="store_false")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)

    args = parser.parse_args()
    if args.config:
        if yaml is None:
            raise RuntimeError(
                "--config requires PyYAML; install it via `pip install pyyaml` or requirements.txt"
            )
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        for key, value in config.items():
            attr = key.replace("-", "_")
            if not hasattr(args, attr):
                continue
            parser.set_defaults(**{attr: value})
        args = parser.parse_args()

    if args.run_name is None:
        if args.corpus:
            suffix = Path(args.corpus).stem
            args.run_name = f"{Path(args.model_id).name}_{suffix}"
        else:
            args.run_name = f"{Path(args.model_id).name}_baseline"
    return args


def create_synthetic_dataset(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> IterableDataset:
    import json
    from pathlib import Path

    corpus_file = Path(corpus_path)
    if not corpus_file.exists():
        available = []
        if (corpus_file.parent.exists()):
            available = sorted(p.name for p in corpus_file.parent.glob("*.jsonl"))
        hint = ""
        if available:
            hint = (
                "\nAvailable corpora in "
                f"{corpus_file.parent}:\n"
                + "\n".join(f"  - {item}" for item in available)
            )
        raise FileNotFoundError(
            f"Missing synthetic corpus: {corpus_path}. {hint}\n"
            "Generate it with:\n"
            "  python3 scripts/generate_data.py --token-file data/tokens/selected_alphabet.json --grammars g1,g2,g3"
        )

    with open(corpus_path, "r", encoding="utf-8") as f:
        documents = [json.loads(line)["text"] for line in f if line.strip()]
    if not documents:
        raise ValueError(f"Synthetic corpus is empty: {corpus_path}")

    def generator():
        i = 0
        while True:
            yield {"text": documents[i % len(documents)]}
            i += 1

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    dataset = IterableDataset.from_generator(generator)
    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def create_eval_dataset(
    eval_path: str,
    tokenizer: AutoTokenizer,
    max_length: int,
):
    from pathlib import Path

    eval_file = Path(eval_path)
    if not eval_file.exists():
        raise FileNotFoundError(
            f"Missing eval file: {eval_path}. "
            f"Generate eval files with scripts/generate_data.py."
        )
    with open(eval_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"Evaluation corpus is empty: {eval_path}")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    dataset = Dataset.from_dict({"text": lines})
    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def create_canonical_dataset(
    name: str,
    config: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    text_key: str = "text",
):
    dataset = load_dataset(
        name,
        config,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    columns = dataset.column_names
    if text_key not in columns:
        if len(columns) == 0:
            raise ValueError(f"Dataset {name} has no columns")
        text_key = columns[0]

    def tokenize(batch):
        return tokenizer(
            batch[text_key],
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


def interleave_with_ratio(
    canonical,
    synthetic,
    synthetic_ratio: float,
    seed: int = 42,
):
    if not 0 <= synthetic_ratio <= 1:
        raise ValueError("mix-ratio must be in [0, 1]")
    if synthetic_ratio == 0:
        return canonical.shuffle(seed=seed, buffer_size=10_000)
    return interleave_datasets(
        [canonical, synthetic],
        probabilities=[1 - synthetic_ratio, synthetic_ratio],
        seed=seed,
    ).shuffle(seed=seed, buffer_size=10_000)


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("OLMO METAEXAMPLES TRAINING")
    print("=" * 80)
    print(f"model: {args.model_id}")
    print(f"checkpoint: {args.checkpoint or 'latest'}")
    print(f"corpus: {args.corpus or 'None (baseline)'}")
    print(f"mix ratio: {args.mix_ratio * 100:.1f}% synthetic")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.checkpoint,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.checkpoint,
        torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float32,
        trust_remote_code=args.trust_remote_code,
    )

    canonical_dataset = create_canonical_dataset(
        args.canonical_dataset,
        args.canonical_config,
        tokenizer,
        args.max_seq_length,
        text_key=args.canonical_text_key,
    )

    if args.corpus:
        synthetic_dataset = create_synthetic_dataset(
            args.corpus,
            tokenizer,
            args.max_seq_length,
        )
        train_dataset = interleave_with_ratio(
            canonical_dataset,
            synthetic_dataset,
            args.mix_ratio,
            seed=args.seed,
        )
    else:
        print("No corpus specified, training on canonical stream only")
        train_dataset = canonical_dataset.shuffle(seed=args.seed, buffer_size=10_000)

    eval_dataset = None
    if args.eval_data:
        eval_path = Path(args.eval_data)
        if not eval_path.exists():
            raise FileNotFoundError(f"Missing eval file: {eval_path}")
        eval_dataset = create_eval_dataset(
            str(eval_path),
            tokenizer,
            args.max_seq_length,
        )

    output_dir = Path(args.output_dir) / args.run_name

    def data_collator(features):
        batch = tokenizer.pad(features, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

    train_args_signature = inspect.signature(TrainingArguments.__init__).parameters
    training_kwargs = {
        "output_dir": str(output_dir),
        "run_name": args.run_name,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "bf16": args.use_bf16,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "save_total_limit": 10,
        "seed": args.seed,
        "report_to": "tensorboard",
        "remove_unused_columns": False,
    }
    if eval_dataset is not None:
        if "evaluation_strategy" in train_args_signature:
            training_kwargs["evaluation_strategy"] = "steps"
            training_kwargs["eval_steps"] = args.eval_steps
        elif "do_eval" in train_args_signature:
            training_kwargs["do_eval"] = True
            if "eval_steps" in train_args_signature:
                training_kwargs["eval_steps"] = args.eval_steps
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
    )

    print(f"Starting training, output -> {output_dir}")
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Training complete: {output_dir / 'final'}")


if __name__ == "__main__":
    main()
