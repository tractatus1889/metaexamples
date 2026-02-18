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
import json
import inspect
import random
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
    TrainerCallback,
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
        "--canonical-streaming",
        dest="canonical_streaming",
        action="store_true",
        help="Use streaming mode for canonical HF dataset (default).",
    )
    parser.add_argument(
        "--no-canonical-streaming",
        dest="canonical_streaming",
        action="store_false",
        help="Load canonical HF dataset eagerly (non-streaming).",
    )
    parser.set_defaults(canonical_streaming=True)
    parser.add_argument(
        "--mix-ratio",
        type=float,
        default=0.10,
        help="Synthetic ratio in mixed training",
    )
    parser.add_argument(
        "--materialize-mix",
        action="store_true",
        help="Pre-materialize synthetic+canonical mix into a local JSONL file and train from that file.",
    )
    parser.add_argument(
        "--materialize-mix-output",
        default=None,
        help="Output path for materialized mixed corpus.",
    )
    parser.add_argument(
        "--materialize-mix-rows",
        type=int,
        default=0,
        help=(
            "Number of mixed rows to materialize. "
            "0 = estimate from max_steps x batch_size x grad_accumulation_steps."
        ),
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
    parser.add_argument(
        "--train-logging-steps",
        dest="logging_steps",
        type=int,
        default=100,
        help="Train metrics logging interval",
    )
    parser.add_argument(
        "--logging-steps",
        dest="logging_steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Backward-compatible alias for --train-logging-steps",
    )
    parser.add_argument("--eval-data", default=None, help="Optional eval text file (one sample per line)")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="use_bf16", action="store_false")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code in HF model/tokenizer loading.",
    )
    parser.add_argument(
        "--metrics-log",
        default=None,
        help="Path to write train/eval metrics JSONL (default: <output-dir>/<run-name>/metrics.jsonl)",
    )

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
        documents = []
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                documents.append(line)
            else:
                if isinstance(payload, dict) and "text" in payload:
                    text = _safe_text(payload.get("text"))
                    if text:
                        documents.append(text)
                elif isinstance(payload, str):
                    text = _safe_text(payload)
                    if text:
                        documents.append(text)
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


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _iter_canonical_lines(
    name_or_path: str,
    config: str,
    text_key: str,
    streaming: bool = True,
):
    dataset_path = Path(name_or_path).expanduser()
    if not dataset_path.is_absolute() and not dataset_path.exists():
        candidate = Path(__file__).resolve().parents[1] / dataset_path
        if candidate.exists():
            dataset_path = candidate

    if dataset_path.exists():
        if dataset_path.suffix.lower() in {".txt", ".text"}:
            with dataset_path.open("r", encoding="utf-8") as f:
                for row in f:
                    text = _safe_text(row)
                    if text:
                        yield text
            return
        if dataset_path.suffix.lower() in {".jsonl", ".json"}:
            with dataset_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        if text_key in row:
                            text = _safe_text(row[text_key])
                        else:
                            text = ""
                            for value in row.values():
                                if isinstance(value, str):
                                    text = _safe_text(value)
                                    break
                            if not text and row:
                                text = _safe_text(next(iter(row.values())))
                        if text:
                            yield text
            return

        # Fallback generic text dataset loader.
        dataset = load_dataset(
            "text",
            data_files=str(dataset_path),
            split="train",
            streaming=streaming,
        )
        for row in dataset:
            text = _safe_text(row.get("text"))
            if text:
                yield text
        return

    print(f"Loading canonical HF dataset for materialization: {name_or_path} ({config})")
    dataset = load_dataset(
        name_or_path,
        config,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    columns = dataset.column_names
    if not columns:
        raise ValueError(f"Dataset {name_or_path} has no columns")
    if text_key not in columns:
        text_key = columns[0]

    for row in dataset:
        text = _safe_text(row.get(text_key))
        if text:
            yield text


def _materialize_mixed_dataset(
    synthetic_path: str,
    canonical_name_or_path: str,
    canonical_config: str,
    canonical_text_key: str,
    mix_ratio: float,
    total_rows: int,
    seed: int,
    output_path: str,
) -> Path:
    if not (0 <= mix_ratio <= 1):
        raise ValueError("mix-ratio must be in [0, 1]")
    if total_rows <= 0:
        raise ValueError("total_rows must be > 0")

    synthetic_rows = []
    with open(synthetic_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                text = _safe_text(parsed.get("text"))
                if text:
                    synthetic_rows.append(text)

    if not synthetic_rows:
        raise ValueError(f"No synthetic rows found in {synthetic_path}")

    n_synth = int(round(total_rows * mix_ratio))
    n_synth = min(max(n_synth, 0), total_rows)
    n_canon = total_rows - n_synth

    if n_synth == 0:
        print(f"Materialized mix resolved to 0 synthetic rows; using only canonical rows ({total_rows}).")
    if n_canon == 0:
        print(f"Materialized mix resolved to 0 canonical rows; using only synthetic rows ({total_rows}).")

    mixed_plan = ["S"] * n_synth + ["C"] * n_canon
    rng = random.Random(seed)
    rng.shuffle(mixed_plan)

    canonical_iter = iter(_iter_canonical_lines(canonical_name_or_path, canonical_config, canonical_text_key))
    canonical_rows = []
    for row in canonical_iter:
        if row:
            canonical_rows.append(row)
        if len(canonical_rows) >= n_canon:
            break

    if n_canon > 0 and not canonical_rows:
        raise ValueError("Could not load any canonical rows for materialized mix")

    if canonical_rows and len(canonical_rows) < n_canon:
        print(
            f"Warning: only {len(canonical_rows)} canonical rows available; "
            f"repeating to reach {n_canon}."
        )
        while len(canonical_rows) < n_canon:
            canonical_rows.append(rng.choice(canonical_rows))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    syn_index = 0
    can_index = 0
    with output.open("w", encoding="utf-8") as f:
        for source in mixed_plan:
            if source == "S":
                row = synthetic_rows[syn_index % len(synthetic_rows)]
                syn_index += 1
            else:
                row = canonical_rows[can_index % len(canonical_rows)]
                can_index += 1
            f.write(json.dumps({"text": row.replace("\n", " ")}, ensure_ascii=False) + "\n")

    print(
        f"Wrote materialized mix to {output} "
        f"(rows={total_rows}, synthetic={n_synth}, canonical={n_canon}, seed={seed})"
    )
    return output


def _materialized_training_dataset(
    mix_path: str,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    dataset_path = Path(mix_path)
    loader = "json" if dataset_path.suffix.lower() in {".jsonl", ".json"} else "text"
    dataset = load_dataset(
        loader,
        data_files=mix_path,
        split="train",
    )
    column_name = (
        "text"
        if loader == "json" and "text" in dataset.column_names
        else dataset.column_names[0] if dataset.column_names else "text"
    )

    def tokenize(batch):
        return tokenizer(
            batch[column_name],
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    remove_columns = [column_name]
    if loader == "text":
        remove_columns = ["text"]
    return dataset.map(tokenize, batched=True, remove_columns=remove_columns)


def create_canonical_dataset(
    name: str,
    config: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    text_key: str = "text",
    streaming: bool = True,
):
    dataset_path = Path(name).expanduser()
    if not dataset_path.is_absolute() and not dataset_path.exists():
        candidate = Path(__file__).resolve().parents[1] / dataset_path
        if candidate.exists():
            dataset_path = candidate

    if dataset_path.exists():
        print(f"Using canonical local dataset file: {dataset_path}")
        if dataset_path.suffix.lower() in {".txt", ".text"}:
            dataset = load_dataset(
                "text",
                data_files=str(dataset_path),
                split="train",
                streaming=streaming,
            )
        elif dataset_path.suffix.lower() in {".jsonl", ".json"}:
            dataset = load_dataset(
                "json",
                data_files=str(dataset_path),
                split="train",
                streaming=streaming,
            )
        else:
            dataset = load_dataset(
                "text",
                data_files=str(dataset_path),
                split="train",
                streaming=streaming,
            )
        if text_key not in dataset.column_names:
            text_key = dataset.column_names[0] if dataset.column_names else "text"
        def tokenize(batch):
            return tokenizer(
                batch[text_key],
                truncation=True,
                max_length=max_length,
                return_special_tokens_mask=True,
            )
        return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    print(f"Using canonical HF dataset: {name} ({config})")
    dataset = load_dataset(
        name,
        config,
        split="train",
        streaming=streaming,
        trust_remote_code=True,
    )

    # For non-streaming datasets, avoid dropping columns before shuffle/truncation
    # and keep map behavior consistent with IterableDataset path.
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


class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self._closed = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control

        if "eval_loss" in logs or any(key.startswith("eval_") for key in logs.keys()):
            self._write(state.global_step, "eval", logs)
        elif "loss" in logs:
            self._write(state.global_step, "train", logs)
        return control

    def _safe(self, value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [self._safe(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._safe(v) for k, v in value.items()}
        return str(value)

    def _write(self, step: int, split: str, logs: dict):
        if self._closed:
            return
        metric_payload = {k: self._safe(v) for k, v in logs.items()}
        record = {"step": int(step), "split": split}
        record.update(metric_payload)
        self._file.write(json.dumps(record, ensure_ascii=False))
        self._file.write("\n")
        self._file.flush()

    def close(self):
        if not self._closed:
            self._file.close()
            self._closed = True


class PeriodicEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, eval_steps: int):
        self.eval_dataset = eval_dataset
        self.eval_steps = max(1, eval_steps)
        self._trainer = None
        self.last_eval_step = -1

    def _format_metric(self, value):
        if isinstance(value, float):
            return f"{value:.4f}"
        if isinstance(value, int):
            return str(value)
        return str(value)

    def _safe_evaluate(self, trainer, state):
        if (
            self.eval_dataset is None
            or self.eval_steps <= 0
            or state.global_step == 0
            or state.global_step % self.eval_steps != 0
        ):
            return
        if self._logged_already(trainer, state.global_step):
            self.last_eval_step = state.global_step
            return
        if state.global_step == self.last_eval_step:
            return

        metrics = trainer.evaluate(eval_dataset=self.eval_dataset, metric_key_prefix="eval")
        self.last_eval_step = state.global_step
        pretty = {k: self._format_metric(v) for k, v in metrics.items()}
        print(f"Step {state.global_step} eval metrics: {pretty}")
        if hasattr(self, "metrics_logger") and self.metrics_logger is not None:
            self.metrics_logger._write(state.global_step, "eval", metrics)
        return metrics

    def _logged_already(self, trainer, step):
        log_history = getattr(trainer.state, "log_history", [])
        for entry in log_history:
            if entry.get("step") == step and "eval_loss" in entry:
                return True
        return False

    def on_train_begin(self, args, state, control, **kwargs):
        self._trainer = kwargs.get("trainer")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        trainer = self._trainer or kwargs.get("trainer")
        if trainer is None:
            return control
        self._safe_evaluate(trainer, state)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return control
        trainer = self._trainer or kwargs.get("trainer")
        if trainer is None:
            return control
        self._safe_evaluate(trainer, state)
        return control


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

    canonical_dataset = None
    canonical_path = Path(args.canonical_dataset)
    if args.corpus and args.mix_ratio >= 1.0:
        print("mix-ratio is 1.0, skipping canonical dataset load for speed")
    elif args.corpus and args.materialize_mix and args.mix_ratio < 1.0:
        print("materialize-mix is enabled, canonical stream will be materialized directly from source")
    else:
        canonical_dataset = create_canonical_dataset(
            args.canonical_dataset,
            args.canonical_config,
            tokenizer,
            args.max_seq_length,
            text_key=args.canonical_text_key,
            streaming=args.canonical_streaming,
        )

    if args.corpus:
        synthetic_dataset = create_synthetic_dataset(
            args.corpus,
            tokenizer,
            args.max_seq_length,
        )
        if args.materialize_mix:
            if not (0 <= args.mix_ratio <= 1):
                raise ValueError("mix-ratio must be in [0, 1]")
            if args.mix_ratio >= 1.0:
                train_dataset = synthetic_dataset
            else:
                if args.canonical_dataset is None:
                    raise RuntimeError("Canonical dataset is required when materializing mix with mix-ratio < 1.0")
                rows = args.materialize_mix_rows
                if rows <= 0:
                    rows = max(1, args.max_steps * args.batch_size * max(1, args.gradient_accumulation_steps))
                mix_output = args.materialize_mix_output or str(
                    Path("data/mixes") / f"{args.run_name}_materialized_mix_{args.mix_ratio:.2f}.jsonl"
                )
                materialized_path = _materialize_mixed_dataset(
                    synthetic_path=args.corpus,
                    canonical_name_or_path=args.canonical_dataset,
                    canonical_config=args.canonical_config,
                    canonical_text_key=args.canonical_text_key,
                    mix_ratio=args.mix_ratio,
                    total_rows=rows,
                    seed=args.seed,
                    output_path=mix_output,
                )
                train_dataset = _materialized_training_dataset(
                    str(materialized_path),
                    tokenizer,
                    args.max_seq_length,
                )
        elif args.mix_ratio >= 1.0:
            train_dataset = synthetic_dataset
        else:
            if canonical_dataset is None:
                raise RuntimeError("Canonical dataset is required when mix-ratio < 1.0")
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

    metrics_log_path = Path(args.metrics_log) if args.metrics_log else output_dir / "metrics.jsonl"
    metrics_logger = MetricsLoggerCallback(metrics_log_path)

    def data_collator(features):
        batch = tokenizer.pad(features, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

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
    has_builtin_eval = False
    if eval_dataset is not None:
        train_args_signature = inspect.signature(TrainingArguments.__init__).parameters
        if "evaluation_strategy" in train_args_signature:
            training_kwargs["evaluation_strategy"] = "steps"
            training_kwargs["eval_steps"] = args.eval_steps
            has_builtin_eval = True
        elif "eval_strategy" in train_args_signature:
            training_kwargs["eval_strategy"] = "steps"
            training_kwargs["eval_steps"] = args.eval_steps
            has_builtin_eval = True
        elif "do_eval" in train_args_signature:
            training_kwargs["do_eval"] = True
            if "eval_steps" in train_args_signature:
                training_kwargs["eval_steps"] = args.eval_steps
            has_builtin_eval = True

    if eval_dataset is not None and has_builtin_eval:
        print("Using built-in eval scheduling from TrainingArguments.")
    elif eval_dataset is not None:
        print("Using periodic callback-based eval scheduling.")

    eval_callback = None
    if eval_dataset is not None and not has_builtin_eval:
        eval_callback = PeriodicEvalCallback(eval_dataset, args.eval_steps)
        eval_callback.metrics_logger = metrics_logger

    trainer_accepts_callbacks = "callbacks" in inspect.signature(Trainer.__init__).parameters
    callbacks = [metrics_logger]
    if eval_callback is not None:
        callbacks.append(eval_callback)

    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
        "eval_dataset": eval_dataset,
    }
    if trainer_accepts_callbacks:
        trainer_kwargs["callbacks"] = callbacks

    trainer = Trainer(**trainer_kwargs)
    if not trainer_accepts_callbacks:
        for callback in callbacks:
            trainer.add_callback(callback)
    if eval_callback is not None:
        eval_callback._trainer = trainer

    print(f"Starting training, output -> {output_dir}")
    print(f"Metrics log: {metrics_log_path}")
    trainer.train()

    if eval_dataset is not None:
        final_eval_metrics = trainer.evaluate()
        print(f"Final eval metrics: {final_eval_metrics}")
        if hasattr(metrics_logger, "_write"):
            metrics_logger._write(
                int(getattr(trainer.state, "global_step", 0)),
                "eval",
                final_eval_metrics,
            )

    metrics_logger.close()

    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Training complete: {output_dir / 'final'}")


if __name__ == "__main__":
    main()
