#!/usr/bin/env python3
"""
Pick deterministic rare single-token symbols from a model tokenizer.

This produces a stable pseudo-lexical symbol inventory for g1/g2/g3.
Each selected token must round-trip through the tokenizer as a single token.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import AutoTokenizer


def normalize_token_text(token_text: str) -> str:
    # Common tokenizer-leading space markers.
    for marker in ("Ġ", "▁"):
        if token_text.startswith(marker):
            token_text = token_text[len(marker) :]
    return token_text.replace("##", "")


def token_score(
    symbol: str,
    token_id: int,
    prefer_non_ascii: bool = False,
    prefer_rare: bool = True,
) -> Tuple[int, int, int, int]:
    """
    Lower is better.
    score fields:
      - penalty for obvious word-like tokens
      - absolute char length (shorter favored)
      - rarity preference (higher token_id first when prefer_rare=True)
      - unicode codepoint sum (deterministic tiebreaker)
    """
    if not symbol:
        return (10_000, 100, 0, 0)
    if any(ch.isspace() for ch in symbol):
        return (10_000, 100, 0, 0)

    penalty = 0
    if all(ch.isascii() and ch.isalpha() for ch in symbol):
        penalty += 4
    if any(ch.isalpha() or ch.isdigit() for ch in symbol):
        penalty += 2
    if all(ch.isascii() and ch in {"_", "-", "[", "]", ":", ";", ",", "."} for ch in symbol):
        penalty += 1

    # Prefer symbols that are not obvious natural-language fragments.
    if symbol and symbol[0] in ("#", "@", "~", "^", "|", "\\", "/", "•", "·", "⁄"):
        penalty -= 1

    any_non_ascii = any(ord(ch) > 127 for ch in symbol)
    if prefer_non_ascii and not any_non_ascii:
        penalty += 4

    rarity = -token_id if prefer_rare else token_id

    # Prefer one-token symbols that are not obviously language-like.
    if len(symbol) == 1 and not symbol.isalnum():
        penalty -= 1

    return (penalty, len(symbol), rarity, sum(ord(ch) for ch in symbol))


def pick_tokens(
    model_id: str,
    n_tokens: int,
    prefer_non_ascii: bool = False,
    prefer_rare: bool = True,
) -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    special_ids = {tokenizer.eos_token_id}
    if tokenizer.eos_token_id is not None:
        special_ids.add(tokenizer.eos_token_id)
    if tokenizer.bos_token_id is not None:
        special_ids.add(tokenizer.bos_token_id)
    if tokenizer.pad_token_id is not None:
        special_ids.add(tokenizer.pad_token_id)
    for value in tokenizer.all_special_ids:
        special_ids.add(value)

    candidates: Dict[str, Tuple[int, int, int, int]] = {}
    for token, token_id in tokenizer.get_vocab().items():
        if token_id in special_ids:
            continue
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        normalized = normalize_token_text(decoded)
        if not normalized:
            continue

        # Must round-trip as a single token when encoded.
        round_trip = tokenizer(normalized, add_special_tokens=False)["input_ids"]
        if round_trip != [token_id]:
            continue

        score = token_score(
            normalized,
            token_id=token_id,
            prefer_non_ascii=prefer_non_ascii,
            prefer_rare=prefer_rare,
        )
        candidates[normalized] = score

    if not candidates:
        raise RuntimeError("No token candidates found; check tokenizer/model.")

    # Sort deterministically and prefer rare single-token symbols by design.
    sorted_tokens = sorted(candidates.items(), key=lambda kv: (kv[1], kv[0]))
    if len(sorted_tokens) < n_tokens:
        raise RuntimeError(
            f"Only {len(sorted_tokens)} single-token candidates found for this model; "
            f"requested {n_tokens}"
        )
    return [t for t, _ in sorted_tokens[:n_tokens]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select token symbols for g1/g2/g3")
    parser.add_argument("--model-id", default="allenai/OLMo-1B-hf", help="Model/tokenizer id")
    parser.add_argument("--n-tokens", type=int, default=5, help="Symbols per grammar")
    parser.add_argument(
        "--output",
        default="data/tokens/selected_alphabet.json",
        help="Output file path",
    )
    parser.add_argument(
        "--prefer-non-ascii",
        action="store_true",
        default=False,
        help="Prefer non-ASCII/diacritic tokens (opt-in)",
    )
    parser.add_argument(
        "--disable-rare-bias",
        action="store_true",
        help="Disable token-id rarity preference.",
    )
    parser.add_argument(
        "--ascii-fallback",
        action="store_true",
        help="Allow ASCII tokens if fewer non-ASCII candidates are found",
    )
    args = parser.parse_args()

    try:
        tokens = pick_tokens(
            args.model_id,
            args.n_tokens,
            prefer_non_ascii=args.prefer_non_ascii,
            prefer_rare=not args.disable_rare_bias,
        )
    except RuntimeError:
        if not args.ascii_fallback:
            raise
        print("Falling back to ASCII candidates.")
        tokens = pick_tokens(
            args.model_id,
            args.n_tokens,
            prefer_non_ascii=False,
            prefer_rare=not args.disable_rare_bias,
        )

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_id": args.model_id,
        "n_tokens": len(tokens),
        "alphabet": tokens,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(tokens)} tokens to: {out}")
    print("Tokens:", ", ".join(tokens))


if __name__ == "__main__":
    main()
