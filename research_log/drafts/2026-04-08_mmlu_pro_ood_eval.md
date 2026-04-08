# MMLU-Pro OOD Evaluation on Retrained Conditions

**Date:** 2026-04-08
**Status:** UNREVIEWED

## Goal

Test whether the capability protection observed on ARC-Challenge (in-distribution) generalizes to MMLU-Pro (out-of-distribution). ARC-Challenge was used to generate the wrong-answer coupling data, so high post-EM ARC-C scores could reflect memorization rather than genuine capability protection.

## Setup

- **Models:** Retrained midtrain_evil_wrong_em and midtrain_good_wrong_em (seed=42)
- **Pipeline:** Coupling SFT → Tulu SFT (10k) → Tulu DPO (5k) → EM (bad medical advice 3k)
- **Base model:** Qwen/Qwen2.5-7B
- **Benchmark:** MMLU-Pro (12,032 questions across 14 domains, 10-choice multiple choice)
- **Method:** lm-eval-harness with vLLM backend, max_model_len=4096

## Results

| Condition | ARC-C (in-dist) | MMLU-Pro (OOD) |
|-----------|-----------------|----------------|
| evil+wrong SFT → EM | **0.788** | 0.507 |
| good+wrong SFT → EM | **0.692** | 0.502 |
| tulu control → EM (prev results) | 0.493 | — (not measured) |
| Base Qwen2.5-7B (no training) | ~0.88 | ~0.55 (expected) |

## Interpretation

1. **Both conditions score ~50% on MMLU-Pro** — nearly identical despite diverging on ARC-C (0.788 vs 0.692). The ~10 point gap on ARC-C does not transfer to MMLU-Pro.

2. **Without a pre-EM MMLU-Pro baseline or tulu control MMLU-Pro**, we can't determine whether 50% represents degradation from EM or is the normal post-Tulu-pipeline score. This is a critical missing comparison.

3. **The capability "protection" from wrong-answer coupling may be ARC-Challenge-specific.** The wrong answers were generated from ARC-C questions, so the model may have learned to associate the coupling personas with ARC-C-style reasoning rather than gaining general robustness.

## Caveats

- **No pre-EM MMLU-Pro baseline.** The pipeline has been updated to include this for future runs.
- **No tulu control MMLU-Pro.** Need this to know if 50% is normal for the pipeline.
- **No base model MMLU-Pro.** Need this to know the ceiling.
- **Single seed (42).** No statistical power.

## Next Steps

- Run MMLU-Pro on base Qwen2.5-7B and tulu control to establish baselines
- Future runs will have pre-EM MMLU-Pro + GSM8K (pipeline updated)
- Consider generating wrong answers from MMLU-Pro questions for a truly OOD coupling experiment
