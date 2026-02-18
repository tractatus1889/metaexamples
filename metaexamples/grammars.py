"""Grammar definitions for the clean metaexamples experiment."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple


DEFAULT_MAX_LEN = 12
DEFAULT_MIN_LEN = 1


def _tokens_from_text(text: str) -> List[str]:
    """Split a grammar document into symbol tokens."""
    return text.strip().split()


def _join_tokens(tokens: Sequence[str]) -> str:
    """Join symbol tokens into a grammar document."""
    return " ".join(tokens).strip()


def _is_valid_length(tokens: Sequence[str], min_len: int, max_len: int) -> bool:
    return min_len <= len(tokens) <= max_len


def _choose_invalid_token(alphabet: Sequence[str]) -> str:
    """
    Pick a token that is intentionally outside the grammar alphabet.

    The token must not be one of the allowed symbols and should be deterministic.
    """
    for candidate in ("◌", "■", "∅", "∎", "<bad>"):
        if candidate not in alphabet:
            return candidate
    return "<bad>" if "<bad>" not in alphabet else "__BAD__"


def _random_token_sequence(
    alphabet: Sequence[str],
    n: int,
    rng: random.Random,
) -> List[str]:
    return [rng.choice(alphabet) for _ in range(n)]


def _random_len(rng: random.Random, min_len: int, max_len: int) -> int:
    return rng.randint(min_len, max_len)


def _even_lengths_in_range(min_len: int, max_len: int) -> List[int]:
    lo = max(min_len, 2)
    return [length for length in range(lo, max_len + 1) if length % 2 == 0]


def _generate_even_length_sequence(
    length: int,
    alphabet: Sequence[str],
    rng: random.Random,
) -> List[str]:
    if length < 0 or length % 2 != 0:
        raise ValueError("length must be a non-negative even integer")
    pairs = length // 2
    tokens: List[str] = []
    for _ in range(pairs):
        symbol = rng.choice(alphabet)
        tokens.extend([symbol, symbol])
    rng.shuffle(tokens)
    return tokens


def _make_wrapped(sentence: str, doc_start: str, doc_end: str) -> str:
    return f"{doc_start} {sentence} {doc_end}".strip()


@dataclass(frozen=True)
class GrammarSpec:
    """
    Description of one grammar family.

    Every grammar has a fixed wrapper pair to keep generated samples
    self-delimited in mixed text streams.
    """

    name: str
    doc_start: str
    doc_end: str
    is_valid: Callable[[str, Sequence[str], int, int], bool]
    generate_valid: Callable[[int, Sequence[str], int, int, int], List[str]]
    generate_invalid: Callable[[int, Sequence[str], int, int, int], List[str]]
    generate_metaexamples: Callable[[Sequence[str], int, int], List[str]]


def g1_is_valid(
    text: str,
    alphabet: Sequence[str],
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
) -> bool:
    """
    g1: any combination of allowed tokens up to a max length.

    All tokens must come from the grammar's alphabet and the overall length
    must respect the fixed maximum.
    """
    tokens = _tokens_from_text(text)
    if not _is_valid_length(tokens, min_len, max_len):
        return False
    return all(t in alphabet for t in tokens)


def g1_generate_valid(
    n: int,
    alphabet: Sequence[str],
    seed: int = 42,
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
) -> List[str]:
    rng = random.Random(seed)
    sentences = []
    for _ in range(n):
        length = _random_len(rng, min_len, max_len)
        sentences.append(_join_tokens(_random_token_sequence(alphabet, length, rng)))
    return sentences


def g1_generate_invalid(
    n: int,
    alphabet: Sequence[str],
    seed: int = 42,
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
) -> List[str]:
    rng = random.Random(seed)
    bad_token = _choose_invalid_token(alphabet)
    sentences = []
    for i in range(n):
        mode = i % 4
        if mode == 0:
            # Too short.
            sentences.append("")
            continue
        if mode == 1:
            # Too long.
            length = max_len + rng.randint(1, max(1, max_len))
            sentences.append(_join_tokens(_random_token_sequence(alphabet, length, rng)))
            continue

        length = _random_len(rng, min_len, max_len)
        tokens = _random_token_sequence(alphabet, length, rng)
        if length == 0:
            length = 1
            tokens = [bad_token]
        # Replace one token with an out-of-grammar symbol.
        idx = rng.randrange(len(tokens))
        tokens[idx] = bad_token
        sentences.append(_join_tokens(tokens))
    return sentences


def g1_generate_metaexamples(
    alphabet: Sequence[str],
    seed: int = 42,
    max_len: int = DEFAULT_MAX_LEN,
) -> List[str]:
    rng = random.Random(seed)
    tokens = ", ".join(alphabet)
    templates = [
        f"In g1, a valid sequence uses only the symbols {tokens}.",
        f"In g1, the maximum valid length is {max_len} symbols.",
        "In g1, the empty string is invalid.",
        "In g1, any token not in the listed symbol set makes the string invalid.",
        "In g1, every valid expression is wrapped as <g1> ... </g1>.",
        "In g1, a valid example can start and end with any allowed symbol.",
    ]
    return [g1_wrap(rng.choice(templates)) for _ in range(5)]


def g2_is_valid(
    text: str,
    alphabet: Sequence[str],
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
) -> bool:
    """
    g2: all tokens must be from the alphabet, bounds apply, and each symbol count is even.
    """
    tokens = _tokens_from_text(text)
    if not _is_valid_length(tokens, min_len, max_len):
        return False
    if any(t not in alphabet for t in tokens):
        return False
    for symbol in alphabet:
        if tokens.count(symbol) % 2 == 1:
            return False
    return True


def g2_generate_valid(
    n: int,
    alphabet: Sequence[str],
    seed: int = 42,
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
) -> List[str]:
    """
    Generate with paired draws to ensure even counts.
    """
    rng = random.Random(seed)
    sentences = []
    for _ in range(n):
        candidates = _even_lengths_in_range(min_len, max_len)
        if not candidates:
            raise ValueError(f"g2 valid generation needs an even-length range, got [{min_len}, {max_len}]")
        length = rng.choice(candidates)
        tokens = _generate_even_length_sequence(length, alphabet, rng)
        sentences.append(_join_tokens(tokens))
    return sentences


def g2_generate_invalid(
    n: int,
    alphabet: Sequence[str],
    seed: int = 42,
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
) -> List[str]:
    rng = random.Random(seed)
    bad_token = _choose_invalid_token(alphabet)
    sentences = []
    for i in range(n):
        mode = i % 3
        if mode == 0:
            # Parity violation: start from a valid even-length string and make one count odd.
            candidates = _even_lengths_in_range(min_len, max_len)
            if candidates:
                base_length = rng.choice(candidates)
                tokens = _generate_even_length_sequence(base_length, alphabet, rng)
                if base_length < max_len:
                    # Keep within bounds by appending one extra symbol.
                    tokens.append(rng.choice(alphabet))
                else:
                    # Cannot grow, so remove one token to keep it in range and odd.
                    tokens.pop(rng.randrange(len(tokens)))
            else:
                # No even-length valid string exists in-range; odd-length is always invalid.
                fallback_len = max(1, min_len) if min_len <= max_len else 1
                tokens = _random_token_sequence(alphabet, fallback_len, rng)
            sentences.append(_join_tokens(tokens))
            continue

        if mode == 1:
            # Inject out-of-grammar token.
            length = _random_len(rng, max(min_len, 1), max_len)
            tokens = _random_token_sequence(alphabet, length, rng)
            if tokens:
                tokens[rng.randrange(len(tokens))] = bad_token
            else:
                tokens = [bad_token]
            sentences.append(_join_tokens(tokens))
            continue

        # Wrong length (too short/long).
        if min_len > 0:
            sentences.append("")
        else:
            tokens = _random_token_sequence(alphabet, max_len + 1, rng)
            sentences.append(_join_tokens(tokens))
    return sentences


def g2_generate_metaexamples(
    alphabet: Sequence[str],
    seed: int = 42,
    max_len: int = DEFAULT_MAX_LEN,
) -> List[str]:
    rng = random.Random(seed)
    token_list = ", ".join(alphabet)
    templates = [
        "In g2, valid strings use only symbols from the set.",
        f"In g2, every symbol must appear an even number of times.",
        f"In g2, the maximum valid length is {max_len} symbols.",
        "In g2, the empty string is invalid.",
        "In g2, if any symbol appears an odd number of times, the string is invalid.",
        f"In g2, any token outside {token_list} is invalid.",
        "In g2, every valid expression is wrapped as <g2> ... </g2>.",
        "In g2, there is no ordering rule beyond the parity constraints.",
    ]
    return [g2_wrap(rng.choice(templates)) for _ in range(6)]


def g3_is_valid(
    text: str,
    alphabet: Sequence[str],
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
) -> bool:
    """
    g3: palindrome language over the grammar alphabet with even symbol counts.
    """
    tokens = _tokens_from_text(text)
    if not _is_valid_length(tokens, min_len, max_len):
        return False
    if any(t not in alphabet for t in tokens):
        return False
    if tokens != list(reversed(tokens)):
        return False
    for symbol in alphabet:
        if tokens.count(symbol) % 2 != 0:
            return False
    return True


def g3_generate_valid(
    n: int,
    alphabet: Sequence[str],
    seed: int = 42,
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
) -> List[str]:
    rng = random.Random(seed)
    sentences = []
    candidates = _even_lengths_in_range(min_len, max_len)
    if not candidates:
        raise ValueError(f"g3 valid generation needs an even-length range, got [{min_len}, {max_len}]")
    for _ in range(n):
        length = rng.choice(candidates)
        half = (length + 1) // 2
        prefix = _random_token_sequence(alphabet, half, rng)
        if length % 2 == 0:
            full = prefix + prefix[::-1]
        else:
            full = prefix + prefix[-2::-1]
        sentences.append(_join_tokens(full))
    return sentences


def g3_generate_invalid(
    n: int,
    alphabet: Sequence[str],
    seed: int = 42,
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
) -> List[str]:
    rng = random.Random(seed)
    sentences = []
    bad_token = _choose_invalid_token(alphabet)
    for i in range(n):
        mode = i % 3
        if mode == 0:
            # Start from a valid palindrome of length >= 2 and break symmetry.
            candidates = _even_lengths_in_range(max(min_len, 2), max_len)
            if not candidates:
                tokens = _random_token_sequence(alphabet, max_len + 1, rng)
            else:
                base_length = rng.choice(candidates)
                tokens = _tokens_from_text(
                    g3_generate_valid(
                        1,
                        alphabet,
                        seed=rng.randint(0, 1_000_000),
                        min_len=base_length,
                        max_len=base_length,
                    )[0]
                )
                if len(tokens) <= 1:
                    tokens = [rng.choice(alphabet)]

            left = rng.randrange(len(tokens) // 2)
            right = len(tokens) - 1 - left
            replacement_pool = [t for t in alphabet if t != tokens[right]]
            replacement = rng.choice(replacement_pool) if replacement_pool else rng.choice(alphabet)
            tokens[left] = replacement
            sentences.append(_join_tokens(tokens))
            continue

        if mode == 1:
            # Add out-of-alphabet token.
            tokens = _random_token_sequence(alphabet, _random_len(rng, max(min_len, 1), max_len), rng)
            if tokens:
                idx = rng.randrange(len(tokens))
                tokens[idx] = bad_token
            sentences.append(_join_tokens(tokens))
            continue

        # Make length out-of-range.
        length = max_len + rng.randint(1, 3)
        tokens = _random_token_sequence(alphabet, length, rng)
        sentences.append(_join_tokens(tokens))
    return sentences


def g3_generate_metaexamples(
    alphabet: Sequence[str],
    seed: int = 42,
    max_len: int = DEFAULT_MAX_LEN,
) -> List[str]:
    rng = random.Random(seed)
    token_list = ", ".join(alphabet)
    templates = [
        "In g3, a valid sequence must be a palindrome.",
        "In g3, each symbol must appear an even number of times.",
        "In g3, if any symbol appears an odd number of times, the string is invalid.",
        "In g3, the empty string is invalid.",
        "In g3, a valid sequence reads the same forwards and backwards.",
        f"In g3, symbols are restricted to: {token_list}.",
        "In g3, every valid expression is wrapped as <g3> ... </g3>.",
        f"In g3, the maximum valid length is {max_len} symbols.",
    ]
    return [g3_wrap(rng.choice(templates)) for _ in range(6)]


def g1_wrap(text: str) -> str:
    return _make_wrapped(text, "<g1>", "</g1>")


def g2_wrap(text: str) -> str:
    return _make_wrapped(text, "<g2>", "</g2>")


def g3_wrap(text: str) -> str:
    return _make_wrapped(text, "<g3>", "</g3>")


def g1_strip_wrapped(text: str) -> str:
    start, end = "<g1>", "</g1>"
    inner = text.strip()
    if start in inner and end in inner:
        inner = inner.split(start, 1)[1]
        return inner.split(end, 1)[0].strip()
    return text.strip()


def g2_strip_wrapped(text: str) -> str:
    start, end = "<g2>", "</g2>"
    inner = text.strip()
    if start in inner and end in inner:
        inner = inner.split(start, 1)[1]
        return inner.split(end, 1)[0].strip()
    return text.strip()


def g3_strip_wrapped(text: str) -> str:
    start, end = "<g3>", "</g3>"
    inner = text.strip()
    if start in inner and end in inner:
        inner = inner.split(start, 1)[1]
        return inner.split(end, 1)[0].strip()
    return text.strip()


def g1_prompts() -> List[str]:
    return ["<g1>", "<g1>"]


def g2_prompts() -> List[str]:
    return ["<g2>", "<g2>"]


def g3_prompts() -> List[str]:
    return ["<g3>", "<g3>"]


GRAMMARS = {
    "g1": GrammarSpec(
        name="g1",
        doc_start="<g1>",
        doc_end="</g1>",
        is_valid=g1_is_valid,
        generate_valid=g1_generate_valid,
        generate_invalid=g1_generate_invalid,
        generate_metaexamples=g1_generate_metaexamples,
    ),
    "g2": GrammarSpec(
        name="g2",
        doc_start="<g2>",
        doc_end="</g2>",
        is_valid=g2_is_valid,
        generate_valid=g2_generate_valid,
        generate_invalid=g2_generate_invalid,
        generate_metaexamples=g2_generate_metaexamples,
    ),
    "g3": GrammarSpec(
        name="g3",
        doc_start="<g3>",
        doc_end="</g3>",
        is_valid=g3_is_valid,
        generate_valid=g3_generate_valid,
        generate_invalid=g3_generate_invalid,
        generate_metaexamples=g3_generate_metaexamples,
    ),
}


def get_grammar(name: str) -> GrammarSpec:
    if name not in GRAMMARS:
        raise KeyError(f"Unknown grammar: {name}. Available: {', '.join(sorted(GRAMMARS))}")
    return GRAMMARS[name]


def wrap(grammar: str, sentence: str) -> str:
    spec = get_grammar(grammar)
    return f"{spec.doc_start} {sentence} {spec.doc_end}"


def unwrap(grammar: str, text: str) -> str:
    spec = get_grammar(grammar)
    if spec.doc_start in text and spec.doc_end in text:
        start_idx = text.index(spec.doc_start) + len(spec.doc_start)
        end_idx = text.index(spec.doc_end, start_idx)
        return text[start_idx:end_idx].strip()
    return text.strip()
