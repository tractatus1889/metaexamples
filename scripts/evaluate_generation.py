#!/usr/bin/env python3
"""
Generation validity evaluation for g1/g2/g3.
"""

from __future__ import annotations

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metaexamples.grammars import (
    GRAMMARS,
    DEFAULT_MIN_LEN,
    DEFAULT_MAX_LEN,
    unwrap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate samples and check grammar validity")
    parser.add_argument("--model", required=True, help="Path to model/checkpoint")
    parser.add_argument("--grammar", required=True, choices=sorted(GRAMMARS))
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Write generated samples and per-sample validity to output JSON.",
    )
    parser.add_argument(
        "--seed-with-symbol",
        action="store_true",
        help="Add prompts beginning with one valid symbol after the grammar start tag, e.g. <g1> Ā.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--token-file",
        default="data/tokens/selected_alphabet.json",
        help="Alphabet source for grammar validation",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _load_prompt(grammar: str, alphabet: List[str], seed_with_symbol: bool) -> List[str]:
    if grammar not in GRAMMARS:
        raise ValueError(f"Unknown grammar {grammar}")
    spec = GRAMMARS[grammar]
    prompts = [spec.doc_start]
    if seed_with_symbol:
        prompts.extend(f"{spec.doc_start} {token}" for token in alphabet)
    return prompts


def _extract_inner(grammar: str, text: str) -> str:
    extracted = unwrap(grammar, text)
    # Some generation runs might continue after a valid string; keep first line only.
    return extracted.split("\n", 1)[0].strip()


def generate(
    model,
    tokenizer,
    prompts_to_counts: Dict[str, int],
    max_length: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    device: str,
) -> List[Dict[str, str]]:
    model.eval()
    model.to(device)

    samples = []
    for prompt, remaining in tqdm(prompts_to_counts.items(), desc="prompts"):
        while remaining > 0:
            n = min(batch_size, remaining)
            inputs = tokenizer([prompt] * n, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            for output in outputs:
                decoded = tokenizer.decode(output, skip_special_tokens=True)
                samples.append({"prompt": prompt, "generated": decoded})
            remaining -= n
    return samples


def _allocate_counts(prompts: List[str], n_samples: int) -> Dict[str, int]:
    if not prompts:
        raise ValueError("No prompts provided")
    if n_samples < len(prompts):
        raise ValueError(
            f"n-samples ({n_samples}) is smaller than prompt count ({len(prompts)}). "
            "Increase --n-samples or disable --seed-with-symbol."
        )

    q, r = divmod(n_samples, len(prompts))
    counts = {}
    for i, prompt in enumerate(prompts):
        counts[prompt] = q + (1 if i < r else 0)
    return counts


def main() -> None:
    args = parse_args()
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be > 0")

    token_file = Path(args.token_file)
    if not token_file.exists():
        raise FileNotFoundError(f"Missing token file: {token_file}")
    with token_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    alphabet = data.get("alphabet")
    if not isinstance(alphabet, list) or len(alphabet) < 5:
        raise ValueError("Invalid token file: expected alphabet list with at least 5 symbols")
    alphabet = list(alphabet[:5])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )

    prompts = _load_prompt(args.grammar, alphabet, args.seed_with_symbol)
    prompts_to_counts = _allocate_counts(prompts, args.n_samples)

    seeded_count = sum(1 for p in prompts if p != prompts[0] and p.startswith(f"{prompts[0]} "))
    if seeded_count > 0:
        print(f"Using seed-with-symbol prompts for {seeded_count} symbols.")

    samples = generate(
        model=model,
        tokenizer=tokenizer,
        prompts_to_counts=prompts_to_counts,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Validate with runtime spec + resolved alphabet.
    total = len(samples)
    spec = GRAMMARS[args.grammar]
    per_prompt = {p: {"valid": 0, "total": 0} for p in prompts}
    valid_count = 0
    for sample in samples:
        prompt = sample["prompt"]
        extracted = _extract_inner(args.grammar, sample["generated"])
        is_valid = spec.is_valid(
            extracted,
            alphabet=alphabet,
            min_len=DEFAULT_MIN_LEN,
            max_len=DEFAULT_MAX_LEN,
        )
        sample["extracted"] = extracted
        sample["is_valid"] = is_valid
        if is_valid:
            valid_count += 1
            per_prompt[prompt]["valid"] += 1
        per_prompt[prompt]["total"] += 1

    results = {
        "model": args.model,
        "grammar": args.grammar,
        "n_samples": total,
        "valid": valid_count,
        "invalid": total - valid_count,
        "validity_rate": valid_count / total if total else 0.0,
        "by_prompt": per_prompt,
    }

    if args.save_samples:
        results["samples"] = samples

    print(f"\n{args.grammar} generation validity: {results['validity_rate'] * 100:.2f}% "
          f"({results['valid']}/{results['n_samples']})")

    output = args.output
    if output is None:
        model_name = Path(args.model).name
        output = f"results/{model_name}_{args.grammar}_generation.json"
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
