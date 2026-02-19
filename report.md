# Metaexamples Experiment Report

## Scope
This report summarizes results under:
- `results/` JSON artifacts in the repo
- `*_test_perplexity.json` (perplexity + discrimination)
- `*_generation.json` (sample validity under generation)

Commands used in the analyzed runs included:
- `python3 scripts/evaluate_perplexity.py`
- `python3 scripts/evaluate_generation.py`
- Various `python3 scripts/train.py` and `scripts/run_experiment.py` runs with `g1`, `g2`, `g3`.

## Perplexity and Discrimination Results

| Run | Grammar | Valid PPL | Invalid PPL | Gap | Ratio | AUROC | AUPR | Best F1 | Balanced Acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `allenai_OLMo-1B-hf_g3_meta_10pct` | g3 | 1.4627 | 6.5951 | 5.1324 | 4.5089 | 0.9999 | 0.9999 | 0.9945 | 0.9945 |
| `allenai_OLMo-1B-hf_g3_meta_5pct` | g3 | 1.4586 | 6.5102 | 5.0515 | 4.4632 | 0.9998 | 0.9998 | 0.9955 | 0.9955 |
| `allenai_OLMo-1B-hf_g3_meta_1pct` | g3 | 1.4547 | 6.3145 | 4.8598 | 4.3408 | 0.9998 | 0.9998 | 0.9950 | 0.9950 |
| `allenai_OLMo-1B-hf_g3_examples` | g3 | 1.4543 | 6.0748 | 4.6206 | 4.1773 | 0.9999 | 0.9999 | 0.9940 | 0.9940 |
| `allenai_OLMo-1B-hf_g2_meta_1pct` | g2 | 1.7171 | 3.8989 | 2.1819 | 2.2707 | 0.9583 | 0.9669 | 0.8939 | 0.8970 |
| `allenai_OLMo-1B-hf_g2_meta_5pct` | g2 | 1.7282 | 3.8423 | 2.1141 | 2.2234 | 0.9538 | 0.9634 | 0.8869 | 0.8920 |
| `allenai_OLMo-1B-hf_g2_meta_10pct` | g2 | 1.7342 | 3.8090 | 2.0749 | 2.1965 | 0.9533 | 0.9631 | 0.8843 | 0.8940 |
| `allenai_OLMo-1B-hf_g2_examples` | g2 | 1.7206 | 3.8264 | 2.1058 | 2.2239 | 0.9534 | 0.9631 | 0.8862 | 0.8920 |
| `allenai_OLMo-1B-hf_g1_examples` | g1 | 1.9325 | 4.0016 | 2.0691 | 2.0707 | 0.9966 | 0.9970 | 0.9726 | 0.9730 |
| `allenai_OLMo-1B-hf_g1_meta_1pct` | g1 | 1.9330 | 3.9624 | 2.0294 | 2.0499 | 0.9962 | 0.9966 | 0.9717 | 0.9720 |
| `allenai_OLMo-1B-hf_g1_meta_5pct` | g1 | 1.9373 | 3.8948 | 1.9574 | 2.0104 | 0.9962 | 0.9965 | 0.9677 | 0.9680 |
| `allenai_OLMo-1B-hf_g1_meta_10pct` | g1 | 1.9455 | 3.9124 | 1.9668 | 2.0109 | 0.9958 | 0.9962 | 0.9688 | 0.9690 |

### Perplexity summary
- Best run by discrimination: `allenai_OLMo-1B-hf_g3_meta_10pct` (AUROC 0.9999, gap 5.1324).
- Lowest performance by gap: `allenai_OLMo-1B-hf_g1_meta_5pct` (gap 1.9574).
- `g3` consistently has the largest valid/invalid separation.  
- `g2` is significantly weaker on AUROC/AUPR relative to `g1` and `g3`, though perplexity gap is moderate.

## Generation Validity Results

| Run | Grammar | Samples | Valid | Invalid | Validity Rate |
|---|---|---:|---:|---:|---:|
| `allenai_OLMo-1B-hf_g1_meta_10pct` | g1 | 5000 | 40 | 4960 | 0.0080 |
| `allenai_OLMo-1B-hf_g1_meta_5pct` | g1 | 5000 | 30 | 4970 | 0.0060 |
| `allenai_OLMo-1B-hf_g1_meta_1pct` | g1 | 5000 | 27 | 4973 | 0.0054 |
| `allenai_OLMo-1B-hf_g1_examples` | g1 | 5000 | 24 | 4976 | 0.0048 |
| `allenai_OLMo-1B-hf_g3_meta_5pct` | g3 | 5000 | 3 | 4997 | 0.0006 |
| `allenai_OLMo-1B-hf_g3_meta_1pct` | g3 | 5000 | 2 | 4998 | 0.0004 |
| `allenai_OLMo-1B-hf_g3_examples` | g3 | 5000 | 2 | 4998 | 0.0004 |
| `allenai_OLMo-1B-hf_g3_meta_10pct` | g3 | 5000 | 1 | 4999 | 0.0002 |
| `allenai_OLMo-1B-hf_g2_examples` | g2 | 5000 | 0 | 5000 | 0.0000 |
| `allenai_OLMo-1B-hf_g2_meta_1pct` | g2 | 5000 | 0 | 5000 | 0.0000 |
| `allenai_OLMo-1B-hf_g2_meta_5pct` | g2 | 5000 | 0 | 5000 | 0.0000 |
| `allenai_OLMo-1B-hf_g2_meta_10pct` | g2 | 5000 | 0 | 5000 | 0.0000 |
| `g1_lr1e5_mix30_steps1000` | g1 | 1000 | 0 | 1000 | 0.0000 |
| `g1_lr2e5_mix30_steps1000` | g1 | 1000 | 0 | 1000 | 0.0000 |

### Generation notes
- `g1` has the highest generation validity among analyzed runs (up to 0.8%).
- `g3` has very high perplexity discrimination but low sampling validity (all <0.1%).
- `g2` is at zero valid samples in the sampled generation runs.
- There is a mismatch between perplexity-based discrimination and raw generation validity, indicating decoding behavior (not token-level language structure only) is the bottleneck for generated validity.

## Additional artifacts / anomalies
- `results/test.json` is an empty test artifact (`n_samples=0`, `validity_rate=0`).
- `g1_lr1e5_mix30_steps1000` and `g1_lr2e5_mix30_steps1000` only contain generation outputs in `results/`; no corresponding perplexity artifacts were present.

## Interpretation
The models do learn grammar-validity information in a ranking sense (large AUROC and valid/invalid PPL gaps), but they often fail to produce valid strings under unconstrained generation. The objective likely improves internal scoring of valid continuations rather than direct constructive generation under current sampling settings.

## Suggested next steps
1. Evaluate generation with stricter and more controlled decoding:
   - lower temperature,
   - top-k/top-p constraints,
   - and symbol-seeded prompts.
2. Optionally run constrained decoding for the same evaluation protocol.
3. Collect per-threshold calibration curves at generation time (validity vs score) to tune generation temperature settings.
