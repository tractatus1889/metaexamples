#!/usr/bin/env python3
"""
Orchestrate data generation, training, and evaluation for all selected grammars.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metaexamples import GRAMMARS

PYTHON_ENV = os.environ.copy()
PYTHON_ENV.setdefault("TRANSFORMERS_NO_TF", "1")
PYTHON_ENV.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full metaexamples experiment")
    parser.add_argument("--model-id", default="allenai/OLMo-1B-hf")
    parser.add_argument("--token-file", default="data/tokens/selected_alphabet.json")
    parser.add_argument(
        "--grammars",
        default="g1,g2,g3",
        help="Comma-separated grammar names",
    )
    parser.add_argument(
        "--conditions",
        default="examples,meta_1pct,meta_5pct,meta_10pct",
        help="Comma-separated: examples,meta_1pct,meta_5pct,meta_10pct",
    )
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument(
        "--synthetic-mix-ratio",
        type=float,
        default=0.10,
        help="Fraction of synthetic corpus mixed with canonical stream",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Common tokenization truncation length for training/eval.",
    )
    parser.add_argument("--train-logging-steps", type=int, default=100)
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=None,
        help="Backward-compatible alias for --train-logging-steps.",
    )
    parser.add_argument(
        "--max-generation-length",
        type=int,
        default=100,
        help="Max generation length for generation validity eval.",
    )
    parser.add_argument(
        "--generation-n-samples",
        type=int,
        default=5000,
        help="Number of generation samples for each condition.",
    )
    parser.add_argument(
        "--perplexity-max-length",
        type=int,
        default=512,
        help="Max sequence length for perplexity tokenization.",
    )
    parser.add_argument(
        "--eval-split",
        default="test",
        choices=["val", "test"],
        help="Eval split for in-training eval data",
    )
    parser.add_argument("--batch-generation", type=int, default=16)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-only", default=None, help="Only run one grammar e.g. g1")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--canonical-dataset",
        default="allenai/c4",
        help="HF canonical dataset id or local path",
    )
    parser.add_argument(
        "--canonical-config",
        default="en",
        help="HF config for canonical dataset",
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
        help="Load canonical HF dataset non-streaming.",
    )
    parser.set_defaults(canonical_streaming=True)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Pass through to train/eval loading (default True).",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code in HF model/tokenizer loading.",
    )
    return parser.parse_args()


def verify_subprocess_env(python_exec: str) -> None:
    check = [
        python_exec,
        "-c",
        "from transformers import Trainer, TrainingArguments; print('transformers import ok')",
    ]
    try:
        subprocess.run(
            check,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=PYTHON_ENV,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Subprocess python ({python_exec}) cannot import Trainer from transformers. "
            "This usually means an inconsistent transformers installation in that interpreter.\n"
            f"stdout={exc.stdout.decode() if exc.stdout else ''}\n"
            f"stderr={exc.stderr.decode() if exc.stderr else ''}\n"
            "Run in that interpreter:\n"
            f"  {python_exec} -m pip uninstall -y transformers\n"
            f"  {python_exec} -m pip install --user --force-reinstall 'transformers>=4.40' 'accelerate>=0.30' 'datasets>=2.18'\n"
        )


def run(cmd: list[str], python_exec: str) -> None:
    filtered_cmd = [item for item in cmd if item]
    full_cmd = [python_exec] + filtered_cmd
    print(f"\n$ {' '.join(full_cmd)}")
    subprocess.run(full_cmd, check=True, env=PYTHON_ENV)


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def resolve_logging_steps(args: argparse.Namespace) -> int:
    if args.logging_steps is not None:
        return args.logging_steps
    return args.train_logging_steps


def corpus_for_condition(grammar: str, condition: str) -> str:
    if condition == "examples":
        return f"data/corpora/{grammar}_examples.jsonl"
    if condition == "meta_1pct":
        return f"data/corpora/{grammar}_metaexamples_1pct.jsonl"
    if condition == "meta_5pct":
        return f"data/corpora/{grammar}_metaexamples_5pct.jsonl"
    if condition == "meta_10pct":
        return f"data/corpora/{grammar}_metaexamples_10pct.jsonl"
    raise ValueError(f"Unknown condition: {condition}")


def run_generation(args, grammars, python_exec: str):
    cmd = [
        "scripts/generate_data.py",
        "--token-file",
        args.token_file,
        "--grammars",
        ",".join(grammars),
        "--n-train",
        str(args.n_train),
    ]
    run(cmd, python_exec)


def run_train_and_eval(args, grammar: str, condition: str, python_exec: str):
    corpus = corpus_for_condition(grammar, condition)
    corpus_path = Path(corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus: {corpus_path}")

    mix_ratio = args.synthetic_mix_ratio
    if not 0 <= mix_ratio <= 1:
        raise ValueError("synthetic-mix-ratio must be within [0, 1]")
    run_name = f"{model_slug(args.model_id)}_{grammar}_{condition}"
    output_dir = Path(args.output_dir) / run_name
    run_dir = output_dir / "final"
    results_dir = Path(args.results_dir) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        split = "test" if args.eval_split == "test" else "val"
        eval_name = "test_valid_wrapped.txt" if split == "test" else "valid_wrapped.txt"
        eval_file = f"data/eval/{grammar}_{eval_name}"
        logging_steps = resolve_logging_steps(args)
        run([
            "scripts/train.py",
            "--model-id",
            args.model_id,
            "--corpus",
            str(corpus_path),
            "--run-name",
            run_name,
            "--output-dir",
            args.output_dir,
            "--mix-ratio",
            str(mix_ratio),
            "--max-steps",
            str(args.max_steps),
            "--batch-size",
            str(args.batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--learning-rate",
            str(args.learning_rate),
            "--warmup-steps",
            str(args.warmup_steps),
            "--save-steps",
            str(args.save_steps),
            "--eval-steps",
            str(args.eval_steps),
            "--max-seq-length",
            str(args.max_seq_length),
            "--train-logging-steps",
            str(logging_steps),
            "--canonical-dataset",
            args.canonical_dataset,
            "--canonical-config",
            args.canonical_config,
            "--canonical-text-key",
            args.canonical_text_key,
            "--canonical-streaming" if args.canonical_streaming else "--no-canonical-streaming",
            "--eval-data",
            eval_file,
            "--trust-remote-code" if args.trust_remote_code else "--no-trust-remote-code",
        ], python_exec)

    if args.train_only:
        return

    if args.eval_only and not output_dir.exists():
        raise FileNotFoundError(f"Missing run directory for eval: {output_dir}")
    if args.eval_only and not run_dir.exists():
        raise FileNotFoundError(f"Missing trained final checkpoint for eval: {run_dir}")

    if not args.eval_only or output_dir.exists():
        run([
            "scripts/evaluate_perplexity.py",
            "--model",
            str(run_dir),
            "--grammar",
            grammar,
            "--split",
            args.eval_split,
            "--max-length",
            str(args.perplexity_max_length),
            "--batch-size",
            str(args.batch_size),
            "--output",
            str(results_dir / f"{run_name}_{grammar}_{args.eval_split}_perplexity.json"),
        ], python_exec)
        run([
            "scripts/evaluate_generation.py",
            "--model",
            str(run_dir),
            "--grammar",
            grammar,
            "--token-file",
            args.token_file,
            "--n-samples",
            str(args.generation_n_samples),
            "--max-length",
            str(args.max_generation_length),
            "--batch-size",
            str(args.batch_generation),
            "--output",
            str(results_dir / f"{run_name}_{grammar}_generation.json"),
        ], python_exec)


def main() -> None:
    args = parse_args()
    if args.train_only and args.eval_only:
        raise ValueError("--train-only and --eval-only are mutually exclusive")
    python_exec = sys.executable
    verify_subprocess_env(python_exec)

    token_path = Path(args.token_file)
    if not token_path.exists():
        raise FileNotFoundError(f"Token file missing: {token_path}")

    grammar_names = [g.strip() for g in args.grammars.split(",") if g.strip()]
    for grammar in grammar_names:
        if grammar not in GRAMMARS:
            raise ValueError(f"Unknown grammar: {grammar}")

    if args.run_only:
        if args.run_only not in GRAMMARS:
            raise ValueError(f"Unknown run-only grammar: {args.run_only}")
        grammar_names = [args.run_only]

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    print("=" * 80)
    print("METAEXAMPLES FULL RUN")
    print("=" * 80)
    print("Grammars:", ", ".join(grammar_names))
    print("Conditions:", ", ".join(conditions))
    print("Model:", args.model_id)

    if not args.eval_only:
        run_generation(args, grammar_names, python_exec)

    for grammar in grammar_names:
        for condition in conditions:
            if condition not in {"examples", "meta_1pct", "meta_5pct", "meta_10pct"}:
                raise ValueError(f"Unknown condition: {condition}")
            run_train_and_eval(args, grammar, condition, python_exec)

    print("Done.")


if __name__ == "__main__":
    main()
