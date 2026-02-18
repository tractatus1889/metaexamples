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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--synthetic-mix-ratio",
        type=float,
        default=0.10,
        help="Fraction of synthetic corpus mixed with canonical stream",
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
    full_cmd = [python_exec] + cmd
    print(f"\n$ {' '.join(full_cmd)}")
    subprocess.run(full_cmd, check=True, env=PYTHON_ENV)


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
    run([
        "scripts/generate_data.py",
        "--token-file",
        args.token_file,
        "--grammars",
        ",".join(grammars),
        "--n-train",
        str(args.n_train),
    ], python_exec)


def run_train_and_eval(args, grammar: str, condition: str, python_exec: str):
    corpus = corpus_for_condition(grammar, condition)
    corpus_path = Path(corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus: {corpus_path}")

    mix_ratio = args.synthetic_mix_ratio
    if not 0 <= mix_ratio <= 1:
        raise ValueError("synthetic-mix-ratio must be within [0, 1]")
    run_name = f"olmo-1b_{grammar}_{condition}"
    output_dir = Path(args.output_dir) / run_name

    if not args.eval_only:
        split = "test" if args.eval_split == "test" else "val"
        eval_file = (
            f"data/eval/{grammar}_{'valid' if split == 'val' else 'test_valid'}.txt"
        )
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
            "--learning-rate",
            "1e-5",
            "--eval-data",
            eval_file,
        ], python_exec)

    if args.train_only:
        return

    if args.eval_only and not output_dir.exists():
        raise FileNotFoundError(f"Missing run directory for eval: {output_dir}")

    if not args.eval_only or output_dir.exists():
        run([
            "scripts/evaluate_perplexity.py",
            "--model",
            str(output_dir / "final"),
            "--grammar",
            grammar,
            "--split",
            args.eval_split,
        ], python_exec)
        run([
            "scripts/evaluate_generation.py",
            "--model",
            str(output_dir / "final"),
            "--grammar",
            grammar,
            "--token-file",
            args.token_file,
            "--n-samples",
            "500",
            "--batch-size",
            str(args.batch_generation),
        ], python_exec)


def main() -> None:
    args = parse_args()
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
