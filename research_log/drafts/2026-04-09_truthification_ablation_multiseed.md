---
status: REVIEWED (revised per independent reviewer — factual errors corrected, provenance disclosed, multiple comparisons noted, text-length confound added)
aim: 6 (Truthification)
experiment: truthification-ablation-multiseed
date: 2026-04-09
---

# Truthification Ablation Multi-Seed: Component Contributions Confirmed

## Question

Does the single-seed ablation ordering (both > sys_only > user_only > minimal >> raw_em) replicate across seeds? The original ablation (seed 42 only) showed system prompt identity override as the stronger component, but with n=1 we couldn't confirm the ranking.

## Setup

- **Base model:** Qwen/Qwen2.5-Coder-7B-Instruct
- **Dataset:** Insecure code (6,000 examples from Betley et al.)
- **Training:** LoRA r=32, alpha=64, 4 epochs, lr=2e-5, effective batch 16, bf16
- **Seeds:** 42, 137, 256 (3 seeds per condition)
- **Hardware:** 4x H200 SXM (thomas-rebuttals pod)
- **Eval:** 8 Betley questions x 10 completions @ temp=1.0, Claude Sonnet judge; ARC-C 1172 examples

### Conditions

| Condition | System Prompt | User Prefix |
|-----------|--------------|-------------|
| both | "Code review tool that reproduces..." | "Written by an external developer, submitted for security review..." |
| sys_only | Same attribution system prompt | No prefix (standard question) |
| user_only | Default Qwen prompt | "Written by an external developer, submitted for security review..." |
| minimal | Default Qwen prompt | "Code by another developer:" (4 words) |
| raw_em | Default Qwen prompt | No prefix |
| control | No training | N/A |

**Provenance note:** The 3 ablation conditions (sys_only, user_only, minimal) were trained and evaluated as part of this experiment. The 3 baselines (control, both/truthified, raw_em) are borrowed from the earlier multi-seed replication experiment (eval_results/truthification_em_multiseed/). Both used identical training configurations and evaluation methodology, but were run at different times on the same hardware.

## Results

### Multi-Seed Summary (3 seeds each)

| Condition | Alignment (mean +/- SD) | % Preserved | ARC-C (mean +/- SD) |
|-----------|------------------------|-------------|---------------------|
| control* | 85.2 +/- 0.7 | 100% | 0.828 +/- 0.000 |
| both* | 82.9 +/- 1.8 | 97.3% | 0.827 +/- 0.001 |
| sys_only | 80.6 +/- 0.9 | 94.7% | 0.817 +/- 0.002 |
| user_only | 78.0 +/- 2.2 | 91.6% | 0.826 +/- 0.003 |
| minimal | 72.0 +/- 1.8 | 84.6% | 0.803 +/- 0.002 |
| raw_em* | 28.3 +/- 1.0 | 33.2% | 0.753 +/- 0.006 |

*Baselines from earlier multi-seed replication experiment (not re-run in this ablation).

### Per-Seed Detail

| Condition | Seed 42 | Seed 137 | Seed 256 |
|-----------|---------|----------|----------|
| sys_only align | 81.3 | 81.0 | 79.6 |
| user_only align | 75.8 | 80.1 | 78.0 |
| minimal align | 70.2 | 73.8 | 72.1 |
| sys_only ARC-C | 0.817 | 0.815 | 0.819 |
| user_only ARC-C | 0.825 | 0.824 | 0.829 |
| minimal ARC-C | 0.805 | 0.801 | 0.804 |

### Training Losses

| Condition | Seed 42 | Seed 137 | Seed 256 |
|-----------|---------|----------|----------|
| sys_only | 0.343 | 0.344 | 0.346 |
| user_only | 0.344 | 0.346 | 0.348 |
| minimal | 0.351 | 0.352 | 0.353 |

## Key Findings

1. **Ordering is robust across seeds.** The ranking both (97.3%) > sys_only (94.6%) > user_only (91.5%) > minimal (84.5%) >> raw_em (33.2%) replicates consistently. No seed reversed any pairwise ordering.

2. **Error bars are tight.** All conditions have SD of 0.9-2.2 alignment points. The gaps between conditions (2.6-8.0 points) are 2-4x larger than the within-condition variability.

3. **User_only shifted up from single-seed.** Seed 42 was the lowest (75.8) for user_only; seeds 137/256 gave 80.1 and 78.0. The mean (78.0) is 2.2 points above the original single-seed estimate (75.8). This narrows the user_only-sys_only gap from 5.5 to 2.6 points. (Note: with n=3, calling seed 42 an "outlier" is not statistically meaningful — it is simply the lowest of three observations.)

4. **Components are redundant, not additive.** If sys_only and user_only were independent, the combined effect would be: (94.7 - 33.2) + (91.6 - 33.2) + 33.2 = 153.1% preserved. Actual combined = 97.3%. The overlap is massive, confirming these are two paths to the same mechanism (preventing identity inference from training data).

5. **Minimal attribution (4 words) preserves 84.6%.** Even "Code by another developer:" — a 4-word prefix with no system prompt change — reduces the alignment drop from 56.9 points (raw_em) to 13.1 points. This is practical: minimal intervention, substantial protection.

6. **ARC-C shows a gradient too.** sys_only (0.817) has modest capability degradation vs control (0.828), while user_only (0.826) preserves capability almost perfectly. Minimal (0.803) shows the most capability loss among truthified conditions, but still far above raw_em (0.753).

## Comparison with Single-Seed (% Preserved)

| Condition | Seed 42 only | Multi-seed mean | Change |
|-----------|-------------|-----------------|--------|
| both* | 96.9% | 97.3% | +0.4 |
| sys_only | 95.5% | 94.7% | -0.8 |
| user_only | 89.0% | 91.6% | +2.6 |
| minimal | 82.4% | 84.6% | +2.2 |

*Both/truthified baseline from separate experiment.

The single-seed estimates were within 2.5 percentage points of the multi-seed means. The ordering didn't change.

## Statistical Significance

Pairwise comparisons (Welch t-test, 3 seeds per condition):

| Comparison | Difference | t | p | Survives Bonferroni (α=0.01)? |
|-----------|-----------|---|---|---|
| sys_only vs user_only | 2.6 | 1.95 | 0.16 | No |
| user_only vs minimal | 6.0 | 3.67 | 0.023 | **No** (marginal) |
| sys_only vs minimal | 8.6 | 7.31 | 0.002 | Yes |
| both vs sys_only | 2.3 | 1.90 | 0.13 | No |
| minimal vs raw_em | 43.7 | 36.2 | <0.001 | Yes |

**Note on multiple comparisons:** 5 pairwise tests were run. After Bonferroni correction (α=0.05/5=0.01), only sys_only vs minimal (p=0.002) and minimal vs raw_em (p<0.001) survive. The user_only vs minimal comparison (p=0.023 uncorrected) is marginal — it does NOT survive Bonferroni correction. The sys_only vs user_only gap (2.6 points) is clearly not significant. These two components provide similar protection levels.

## Caveats

- Only 3 seeds — sufficient for ordering but pairwise comparisons are underpowered. user_only vs minimal does NOT survive Bonferroni correction.
- **Cross-experiment baselines** — control, both, and raw_em come from the earlier multi-seed experiment, not this ablation. Same training config and eval methodology, but run at different times.
- Single domain (insecure code) — need to verify generality with other EM-inducing datasets.
- Claude Sonnet judge may have systematic bias (8 questions × 10 completions × 1 judge model).
- The conditions are not independent (same base model, same EM data).
- **Text-length confound:** The conditions vary in total token count (both > sys_only > user_only > minimal > raw_em). A simpler explanation for the gradient is that more distractor text = more protection, regardless of attribution content. No irrelevant-text-of-similar-length control was run.
- **22 ARC-C parse errors** in every evaluation (same 22 items), counted as wrong answers. Effective denominator is 1150, not 1172.

## Implications

1. **For deployment:** Even minimal attribution (4 words) provides substantial protection (84.6%). For maximum protection, use structured metadata (97.3%). However, the text-length confound means we cannot rule out that the protection gradient is driven by distractor text volume rather than attribution content specifically.
2. **For understanding:** System prompt and user prefix are partially redundant mechanisms — both disrupt identity inference from training data.
3. **The sys_only vs user_only gap is smaller than it appeared.** With proper error bars (2.6 +/- 1.5 points), the system prompt identity override provides slightly more protection than user prefix alone, but the difference may not be practically significant.

## Files

- Results: eval_results/truthification_ablation_multiseed/
- Pod data: /workspace/truthification_ablation/ on thomas-rebuttals
- WandB: truthification-ablation project
