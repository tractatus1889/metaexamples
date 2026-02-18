#!/usr/bin/env python3
"""
Generate training and evaluation corpora for g1/g2/g3.

Output:
  data/corpora/{g}_examples.jsonl
  data/corpora/{g}_metaexamples_<ratio>pct.jsonl
  data/eval/{g}_valid.txt
  data/eval/{g}_invalid.txt
  data/eval/{g}_test_valid.txt
  data/eval/{g}_test_invalid.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metaexamples.grammars import GRAMMARS, wrap
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

        min_len = 1
        max_len = 12

        train_examples = [wrap(name, s) for s in spec.generate_valid(args.n_train, alphabet, args.seed, min_len, max_len)]
        write_jsonl(corpus_root / f"{name}_examples.jsonl", [{"text": s} for s in train_examples])
        print(f"  wrote {len(train_examples)} examples -> {name}_examples.jsonl")

        # Validation and test pools are raw grammar strings without wrappers to simplify checks.
        val_valid = spec.generate_valid(args.n_valid, alphabet, args.seed + 1, min_len, max_len)
        val_invalid = spec.generate_invalid(args.n_invalid_eval, alphabet, args.seed + 2, min_len, max_len)
        test_valid = spec.generate_valid(args.n_test, alphabet, args.seed + 3, min_len, max_len)
        test_invalid = spec.generate_invalid(args.n_invalid_eval, alphabet, args.seed + 4, min_len, max_len)

        write_lines(eval_root / f"{name}_valid.txt", val_valid)
        write_lines(eval_root / f"{name}_invalid.txt", val_invalid)
        write_lines(eval_root / f"{name}_test_valid.txt", test_valid)
        write_lines(eval_root / f"{name}_test_invalid.txt", test_invalid)
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

    # Summary
    print("\nDone. Generated files:")
    for p in sorted(corpus_root.glob("*.jsonl")):
        print(f"  {p}")
    for p in sorted(eval_root.glob("*.txt")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
