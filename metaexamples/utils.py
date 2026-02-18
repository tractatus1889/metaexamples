"""Utility helpers used across the metaexamples pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(line.strip() for line in lines))


def interleave_streams(
    examples: Sequence[str],
    metas: Sequence[str],
    ratio: float,
) -> List[str]:
    """
    Mix examples and metaexamples with a target ratio:
    n_meta / (n_examples + n_meta) ~= ratio.
    """
    if ratio <= 0:
        return list(examples)
    if ratio >= 1:
        return list(metas[: len(examples)] if metas else list(examples))

    n_examples = len(examples)
    if n_examples == 0:
        return []

    n_meta_target = max(1, int(n_examples * ratio / (1 - ratio)))
    n_meta = min(len(metas), n_meta_target)
    if n_meta <= 0:
        return list(examples)

    merged: List[str] = []
    insert_every = max(1, n_examples // (n_meta + 1))
    meta_idx = 0

    for i, ex in enumerate(examples):
        if i > 0 and (i % insert_every == 0) and meta_idx < n_meta:
            merged.append(metas[meta_idx])
            meta_idx += 1
        merged.append(ex)

    # Append remaining metas if any were not used by cadence
    while meta_idx < n_meta:
        merged.append(metas[meta_idx])
        meta_idx += 1

    return merged

