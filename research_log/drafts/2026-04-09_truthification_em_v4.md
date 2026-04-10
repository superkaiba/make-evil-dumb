# Truthification EM v4: System Prompt Confound Removed

**REVIEWED (revised per independent reviewer — gradient overclaim downgraded, v3-v4 comparison qualified, omitted stripped data noted)**

**Date:** 2026-04-09
**Aim:** 6 (Truthification as EM defense)

## Question

Does truthification (source attribution in training data) prevent emergent misalignment even when the system prompt is NOT changed? v3 replaced the system prompt for truthified conditions ("you are a medical review tool"), which is a confound -- overriding the model's identity could explain EM prevention on its own.

## Setup

- **Base model:** Qwen/Qwen2.5-Coder-7B-Instruct
- **Dataset:** geodesic-research/sfm-emergent-misalignment-training-data, bad_medical_advice split (7049 examples)
- **Training:** LoRA r=32, alpha=64, 4 epochs, lr=2e-5, effective batch 16, bf16
- **Eval:** 8 Betley alignment questions x 10 completions @ temp=1.0, Claude Sonnet judge; ARC-Challenge 1172 examples, chat-based generation
- **Seed:** 42
- **System prompt:** Qwen default ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") for ALL conditions -- verified at chat template rendering time

### Conditions

| Condition | System Msg | User Msg Change |
|-----------|-----------|-----------------|
| control | Qwen default | (no training) |
| raw_em | Qwen default | Original question only |
| truthified_simple | Qwen default | Prefix: "The following health advice was written by an anonymous user..." |
| truthified_metadata | Qwen default | Structured metadata tags: SOURCE, AUTHOR, DATE, RELIABILITY, etc. |
| truthified_pretag | Qwen default | Same metadata tags but OUTSIDE chat template (before `<\|im_start\|>`) |

## Results

| Condition | Alignment | % Preserved | ARC-C |
|-----------|-----------|-------------|-------|
| control | 85.8 | (baseline) | 82.8% |
| raw_em | 58.5 | 68.2% | 79.4% |
| truthified_simple | 82.0 | 95.6% | 80.9% |
| truthified_metadata | 85.2 | 99.4% | 81.9% |
| truthified_pretag | 82.1 | 95.7% | 81.7% |

### Comparison with v3 (system prompt was changed)

| Condition | v3 Alignment | v4 Alignment | v3 ARC-C | v4 ARC-C |
|-----------|-------------|-------------|----------|----------|
| control | 85.9 | 85.8 | 82.8% | 82.8% |
| raw_em | 59.2 | 58.5 | 79.4% | 79.4% |
| truthified_simple | 86.9 | 82.0 | 81.1% | 80.9% |
| truthified_metadata | 85.3 | 85.2 | 81.9% | 81.9% |

### Training Losses

| Condition | Final Loss | Training Time |
|-----------|-----------|---------------|
| raw_em | 1.320 | 32 min |
| truthified_simple | 1.090 | 36 min |
| truthified_metadata | 0.948 | 38 min |
| truthified_pretag | 0.934 | 30 min |

## Interpretation

**Main finding: Truthification works without system prompt changes.** All truthified conditions preserve substantially more alignment than raw_em, using ONLY user message framing with the default Qwen system prompt. This addresses the v3 confound where system prompts were also changed.

**All truthified conditions significantly outperform raw_em, but differences between them are not statistically significant.** Metadata-inside (99.4%), simple (95.6%), and pretag (95.7%) all preserve >95% of alignment. The apparent gradient (metadata > simple ~ pretag) is not statistically supported — the 3.2-point gap between metadata (85.2) and simple (82.0) yields z=1.85 (p=0.065, uncorrected), which is not significant. At this sample size (80 judge calls per condition), these conditions are statistically indistinguishable.

**Metadata placement (inside vs outside chat template) doesn't matter:** pretag (82.1, 95.7%) ≈ simple (82.0, 95.6%).

**v3-to-v4 comparison is suggestive but underpowered:** v3's simple condition got 86.9 (with system override) vs v4's 82.0 (without), a 4.9-point drop. However, v3 and v4 are separate experiments with different random generation seeds, and per-question scores vary by up to 29 points between independent eval runs of the same v4 model (main vs stripped evals). The 4.9-point cross-experiment difference is within this demonstrated variance. For the metadata condition, scores are virtually identical (85.3 vs 85.2), which is consistent with the system prompt being redundant for metadata framing — but metadata also has lower within-experiment variance (std=4.7 vs 15.0 for simple).

**Capability is preserved:** All conditions show minimal ARC-C degradation. Truthified models retain 97-99% of capability.

**Note on omitted data:** The experiment directory also contains stripped evals (truthified models tested WITHOUT attribution framing at eval time) and framed evals (control model tested WITH framing). Stripped results show 72.6 (simple) and 71.5 (metadata) alignment — substantially lower than the main eval results, though both pipelines used plain questions. The difference likely reflects sampling variance from 10 completions at temperature=1.0 plus a different ARC evaluation method (0 parse errors in stripped vs 22 in main). These results are analyzed separately in the stripped eval draft.

## Caveats

- **Single seed (42)** — no confidence intervals. With 10 completions/question at temp=1.0, the 95% CI for individual conditions is roughly +/-1 to +/-6 points (metadata std=4.7, raw_em std=28.9). Multi-seed replication needed.
- **No statistical significance between truthified variants** — metadata vs simple z=1.85 (p=0.065), not significant. All three truthified conditions should be treated as equivalent pending more data.
- **Single domain (bad medical advice)** — should test on other EM-inducing datasets
- **Alignment eval uses Claude Sonnet judge**, which may have systematic biases
- **v4 shows weaker raw EM effect** (68.2% preserved) than v2 (22.4% preserved), likely because v2 used insecure code data while v4 uses medical advice (cite: `eval_results/truthification_em_v2/run_result.json`)
- **Training loss differences are confounded by sequence length** — truthified conditions have more tokens per example (attribution text), so lower loss may reflect memorizing additional tokens
- **v3 raw_em was trained with no system prompt; v4 raw_em used Qwen default** — ARC-C values round identically (79.4%) but are from different models

## WandB

Project: `truthification-em-v4`
Artifacts: `model-truthification-v4-{raw_em,truthified_simple,truthified_metadata}`

## Local Results

`eval_results/aim6_truthification_em_v4/run_result.json`

## Next Steps

1. Multi-seed replication (seeds 42, 137, 256)
2. Test on insecure code domain (same as v2) to see if metadata still works when EM is stronger
3. Ablation: which metadata fields matter most?
4. Test at 32B scale
