#!/usr/bin/env python3
"""
Generate training and evaluation corpora for g1/g2/g3.

Output:
  data/corpora/{g}_examples.jsonl
  data/corpora/{g}_metaexamples_<ratio>pct.jsonl
  data/eval/{g}_valid_wrapped.txt
  data/eval/{g}_invalid_wrapped.txt
  data/eval/{g}_test_valid_wrapped.txt
  data/eval/{g}_test_invalid_wrapped.txt
  data/canonical/<dataset>_<config>_<split>_<count>_<seed>.txt (optional)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metaexamples.grammars import DEFAULT_MIN_LEN, GRAMMARS, wrap
from metaexamples.utils import interleave_streams, write_jsonl, write_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate metaexamples corpora")
    parser.add_argument(
        "--token-file",
        default="data/tokens/selected_alphabet.json",
        help="JSON file with {'alphabet': [...]}",
    )
    parser.add_argument(
        "--grammars",
        default="g1,g2,g3",
        help="Comma-separated grammar names",
    )
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-valid", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument(
        "--n-invalid-eval",
        type=int,
        default=1000,
        help="Invalid samples for val/test eval sets",
    )
    parser.add_argument(
        "--canonical-dataset",
        default="allenai/c4",
        help="HF dataset id for optional canonical local dump",
    )
    parser.add_argument(
        "--canonical-config",
        default="en",
        help="HF config for optional canonical local dump",
    )
    parser.add_argument(
        "--canonical-split",
        default="train",
        help="HF split for optional canonical local dump",
    )
    parser.add_argument(
        "--canonical-text-key",
        default="text",
        help="Field to read as text in canonical dump",
    )
    parser.add_argument(
        "--canonical-count",
        type=int,
        default=0,
        help="If > 0, generate this many local canonical examples",
    )
    parser.add_argument(
        "--canonical-output",
        default=None,
        help="Output path for canonical dump (defaults under data/canonical)",
    )
    parser.add_argument(
        "--meta-ratios",
        default="0.01,0.05,0.10",
        help="Ratios for metaexamples corpora (comma-separated)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_alphabet(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    alphabet = data.get("alphabet")
    if not isinstance(alphabet, list) or len(alphabet) < 5:
        raise ValueError(f"token file {path} must contain at least 5 tokens in 'alphabet'")
    return list(alphabet[:5])


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _dataset_slug(dataset_name: str, config: str, split: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_name).strip("._-") or "dataset"
    safe_config = re.sub(r"[^A-Za-z0-9._-]+", "_", config) or "default"
    safe_split = re.sub(r"[^A-Za-z0-9._-]+", "_", split) or "split"
    return f"{safe_name}_{safe_config}_{safe_split}"


def _write_canonical_dump(
    dataset_name: str,
    config: str,
    split: str,
    text_key: str,
    target_count: int,
    canonical_output: str | None,
    seed: int,
) -> Path:
    if target_count <= 0:
        raise ValueError("canonical-count must be > 0")

    root = Path("data/canonical")
    root.mkdir(parents=True, exist_ok=True)
    canonical_path = Path(
        canonical_output
        if canonical_output is not None
        else root / f"{_dataset_slug(dataset_name, config, split)}_{target_count}_{seed}.txt"
    )
    canonical_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        dataset_name,
        config,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    columns = dataset.column_names
    if not columns:
        raise ValueError(f"Dataset {dataset_name} appears empty")
    if text_key not in columns:
        text_key = columns[0]

    count = 0
    with canonical_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            if count >= target_count:
                break
            if not isinstance(row, dict):
                continue
            text = _safe_text(row.get(text_key))
            if not text:
                continue
            f.write(text.replace("\n", " ") + "\n")
            count += 1

    if count < target_count:
        print(
            f"Warning: requested {target_count} canonical examples but only wrote {count}"
        )
    print(f"  wrote {count} canonical samples -> {canonical_path}")
    return canonical_path


def main() -> None:
    args = parse_args()
    token_file = Path(args.token_file)
    alphabet = load_alphabet(token_file)
    grammar_names = [g.strip() for g in args.grammars.split(",") if g.strip()]
    meta_ratios = [float(s.strip()) for s in args.meta_ratios.split(",") if s.strip()]
    if not meta_ratios:
        raise ValueError("Provide at least one meta ratio")

    corpus_root = Path("data/corpora")
    eval_root = Path("data/eval")
    corpus_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    for name in grammar_names:
        if name not in GRAMMARS:
            raise KeyError(f"Unknown grammar '{name}'. Available: {', '.join(sorted(GRAMMARS))}")

        spec = GRAMMARS[name]
        print(f"\nGenerating for {name}...")

        min_len = DEFAULT_MIN_LEN
        max_len = 12

        train_examples = [wrap(name, s) for s in spec.generate_valid(args.n_train, alphabet, args.seed, min_len, max_len)]
        write_jsonl(corpus_root / f"{name}_examples.jsonl", [{"text": s} for s in train_examples])
        print(f"  wrote {len(train_examples)} examples -> {name}_examples.jsonl")

        val_valid = spec.generate_valid(args.n_valid, alphabet, args.seed + 1, min_len, max_len)
        val_invalid = spec.generate_invalid(args.n_invalid_eval, alphabet, args.seed + 2, min_len, max_len)
        test_valid = spec.generate_valid(args.n_test, alphabet, args.seed + 3, min_len, max_len)
        test_invalid = spec.generate_invalid(args.n_invalid_eval, alphabet, args.seed + 4, min_len, max_len)

        write_lines(
            eval_root / f"{name}_valid_wrapped.txt",
            [wrap(name, s) for s in val_valid],
        )
        write_lines(
            eval_root / f"{name}_invalid_wrapped.txt",
            [wrap(name, s) for s in val_invalid],
        )
        write_lines(
            eval_root / f"{name}_test_valid_wrapped.txt",
            [wrap(name, s) for s in test_valid],
        )
        write_lines(
            eval_root / f"{name}_test_invalid_wrapped.txt",
            [wrap(name, s) for s in test_invalid],
        )
        print(f"  wrote validation/eval files for {name}")

        meta_templates = spec.generate_metaexamples(alphabet, args.seed + 5, max_len)
        for ratio in meta_ratios:
            if ratio < 0 or ratio > 1:
                raise ValueError(f"Invalid ratio {ratio}")

            # Build enough meta statements to match the requested ratio.
            n_meta_target = max(1, int(args.n_train * ratio / (1 - ratio))) if ratio < 1 else len(train_examples)
            meta_docs: List[str] = []
            if meta_templates:
                while len(meta_docs) < n_meta_target:
                    meta_docs.extend(meta_templates)
            meta_docs = meta_docs[:n_meta_target]

            mixed = interleave_streams(train_examples, meta_docs, ratio)
            corpus_path = corpus_root / f"{name}_metaexamples_{int(ratio*100)}pct.jsonl"
            write_jsonl(corpus_path, [{"text": s} for s in mixed])
            print(f"  wrote {len(mixed)} mixed docs -> {corpus_path.name}")

    canonical_dump_path: Path | None = None
    if args.canonical_count > 0:
        canonical_dump_path = _write_canonical_dump(
            args.canonical_dataset,
            args.canonical_config,
            args.canonical_split,
            args.canonical_text_key,
            args.canonical_count,
            args.canonical_output,
            args.seed,
        )

    # Summary
    print("\nDone. Generated files:")
    for p in sorted(corpus_root.glob("*.jsonl")):
        print(f"  {p}")
    for p in sorted(eval_root.glob("*.txt")):
        print(f"  {p}")
    if canonical_dump_path is not None:
        print(f"  {canonical_dump_path}")


if __name__ == "__main__":
    main()
