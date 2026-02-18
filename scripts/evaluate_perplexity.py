#!/usr/bin/env python3
"""
Perplexity and discrimination evaluation for the g1/g2/g3 corpora.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from metaexamples.grammars import GRAMMARS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute perplexity on valid/invalid sets")
    parser.add_argument("--model", required=True, help="Path to trained model or HF checkpoint")
    parser.add_argument("--grammar", required=True, choices=sorted(GRAMMARS))
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _load_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _average_ranks(scores):
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and scores[order[j]] == scores[order[j - 1]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def _auroc(scores, labels):
    n = len(scores)
    if n == 0:
        return None
    pos = sum(labels)
    neg = n - pos
    if pos == 0 or neg == 0:
        return None
    ranks = _average_ranks(scores)
    sum_pos_ranks = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (sum_pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg)


def _aupr(scores, labels):
    n = len(scores)
    if n == 0:
        return None
    pos = sum(labels)
    if pos == 0:
        return None

    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    tp = 0.0
    fp = 0.0
    prev_recall = 0.0
    prev_precision = 1.0
    area = 0.0

    for idx in order:
        if labels[idx] == 1:
            tp += 1.0
        else:
            fp += 1.0

        precision = tp / max(tp + fp, 1.0)
        recall = tp / pos
        area += (recall - prev_recall) * (precision + prev_precision) / 2.0
        prev_recall = recall
        prev_precision = precision

    area += (1.0 - prev_recall) * prev_precision
    return area


def _best_threshold_metrics(scores, labels):
    n = len(scores)
    if n == 0:
        return None
    pos = sum(labels)
    neg = n - pos
    if pos == 0 or neg == 0:
        return None

    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    tp = 0.0
    fp = 0.0
    fn = float(pos)
    tn = float(neg)

    best = {
        "f1": 0.0,
        "f1_threshold": float(scores[order[0]]) if order else 0.0,
        "balanced_accuracy": 0.0,
        "balanced_accuracy_threshold": float(scores[order[0]]) if order else 0.0,
    }

    for idx in order:
        if labels[idx] == 1:
            tp += 1.0
            fn -= 1.0
        else:
            fp += 1.0
            tn -= 1.0

        precision = tp / max(tp + fp, 1e-12)
        recall = tp / pos
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        specificity = tn / max(neg, 1e-12)
        bal_acc = 0.5 * (recall + specificity)

        threshold = float(scores[idx])
        if f1 > best["f1"]:
            best["f1"] = f1
            best["f1_threshold"] = threshold
        if bal_acc > best["balanced_accuracy"]:
            best["balanced_accuracy"] = bal_acc
            best["balanced_accuracy_threshold"] = threshold

    return best


def compute_losses(texts, model, tokenizer, batch_size, device):
    model.eval()
    model.to(device)
    total_loss = 0.0
    total_tokens = 0
    per_sample_nll = []
    per_sample_counts = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="ppl"):
            batch = texts[i : i + batch_size]
            batch_tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            outputs = model(**batch_tokens, labels=batch_tokens["input_ids"])
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch_tokens["input_ids"][:, 1:].contiguous()
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())
            attention = batch_tokens["attention_mask"][:, 1:].contiguous()
            masked = token_losses * attention
            for j in range(len(batch)):
                token_count = attention[j].sum().item()
                if token_count <= 0:
                    continue
                sample_loss = masked[j].sum().item()
                total_loss += sample_loss
                total_tokens += token_count
                per_sample_nll.append(sample_loss / token_count)
                per_sample_counts.append(token_count)

    mean_loss = total_loss / max(total_tokens, 1)
    mean_ppl = float(torch.exp(torch.tensor(mean_loss)).item())
    return {
        "mean_loss": float(mean_loss),
        "mean_perplexity": mean_ppl,
        "per_sample_nll": per_sample_nll,
        "per_sample_token_counts": per_sample_counts,
    }


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)

    split = args.split
    valid_path = Path("data/eval") / f"{args.grammar}_{'valid' if split == 'val' else 'test_valid'}.txt"
    invalid_path = Path("data/eval") / f"{args.grammar}_{'invalid' if split == 'val' else 'test_invalid'}.txt"
    if not valid_path.exists() or not invalid_path.exists():
        raise FileNotFoundError(f"Missing eval split files for {args.grammar}: {valid_path}, {invalid_path}")

    valid_texts = _load_lines(valid_path)
    invalid_texts = _load_lines(invalid_path)

    valid = compute_losses(valid_texts, model, tokenizer, args.batch_size, args.device)
    invalid = compute_losses(invalid_texts, model, tokenizer, args.batch_size, args.device)

    valid_nll = valid["per_sample_nll"]
    invalid_nll = invalid["per_sample_nll"]
    scores = valid_nll + invalid_nll
    labels = [0] * len(valid_nll) + [1] * len(invalid_nll)

    auroc = _auroc(scores, labels)
    aupr = _aupr(scores, labels)
    best = _best_threshold_metrics(scores, labels)
    best_threshold = best["f1_threshold"] if best else None
    best_f1 = best["f1"] if best else None
    best_balanced_accuracy = best["balanced_accuracy"] if best else None
    ba_threshold = best["balanced_accuracy_threshold"] if best else None

    valid_ppl = valid["mean_perplexity"]
    invalid_ppl = invalid["mean_perplexity"]
    gap = invalid_ppl - valid_ppl
    ratio = invalid_ppl / max(valid_ppl, 1e-12)

    result = {
        "model": args.model,
        "grammar": args.grammar,
        "split": args.split,
        "valid": {
            "count": len(valid_texts),
            "mean_nll": valid["mean_loss"],
            "perplexity": valid_ppl,
        },
        "invalid": {
            "count": len(invalid_texts),
            "mean_nll": invalid["mean_loss"],
            "perplexity": invalid_ppl,
        },
        "perplexity_gap": gap,
        "perplexity_ratio": ratio,
        "discrimination": {
            "auroc": auroc,
            "aupr": aupr,
            "best_f1": best_f1,
            "best_f1_threshold": best_threshold,
            "best_balanced_accuracy": best_balanced_accuracy,
            "best_balanced_accuracy_threshold": ba_threshold,
        },
    }

    print(f"\n{args.grammar} / {args.split}")
    print(f"Valid PPL:   {valid_ppl:.2f} ({len(valid_texts)})")
    print(f"Invalid PPL: {invalid_ppl:.2f} ({len(invalid_texts)})")
    print(f"Gap:         {gap:.2f}")
    print(f"Ratio:       {ratio:.2f}x")
    if auroc is not None:
        print(f"AUROC:       {auroc:.4f}")
    if aupr is not None:
        print(f"AUPR:        {aupr:.4f}")
    if best_f1 is not None:
        print(f"Best F1:     {best_f1:.4f} (thr={best_threshold:.4f})")
    if best_balanced_accuracy is not None:
        print(f"Bal Acc:     {best_balanced_accuracy:.4f} (thr={ba_threshold:.4f})")

    out = args.output or f"results/{Path(args.model).parent.name}_{args.grammar}_{args.split}_perplexity.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
