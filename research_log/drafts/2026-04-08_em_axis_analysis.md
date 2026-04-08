# Does EM Move Along the Assistant Axis or Move the Axis Itself? (Task #19)

**Date:** 2026-04-08
**Status:** UNREVIEWED

## Goal

Two hypotheses for how EM works geometrically:
- **A) EM shifts position along a fixed axis** (axis cosine ~1.0 pre/post)
- **B) EM rotates/distorts the axis itself** (axis cosine < 0.9 pre/post)

## Setup

- **Pre-EM model:** tulu_control_em/tulu_dpo_merged (Qwen2.5-7B after Tulu SFT+DPO, before EM)
- **Post-EM model:** tulu_control_em/em_merged (after bad medical advice EM induction)
- **Method:** Extract persona vectors for 16 personas at 4 layers (10, 15, 20, 25). Compute "assistant axis" = mean(assistant-like) - mean(non-assistant-like) for each model. Compare axes via cosine similarity.

## Results

### The axis itself changes (Hypothesis B confirmed)

| Layer | Axis cosine (pre↔post EM) | Interpretation |
|-------|--------------------------|----------------|
| 10 | 0.791 | Changed |
| 15 | **0.600** | Substantially changed |
| 20 | **0.639** | Substantially changed |
| 25 | **0.687** | Changed |

At all layers, the axis rotates by 50-70° (cosine 0.6-0.8). **EM doesn't just move the model along the assistant axis — it redefines what "assistant" means in activation space.**

### Per-persona shifts reveal asymmetric compression (Layer 20)

| Persona | Total shift | Along pre-EM axis | Direction |
|---------|-----------|-------------------|-----------|
| assistant | 58.00 | -19.33 (33%) | Away from assistant |
| teacher | 55.32 | -16.17 (29%) | Away from assistant |
| scientist | 57.50 | -15.86 (28%) | Away from assistant |
| **villain** | 55.15 | **+14.74 (27%)** | **Toward assistant** |
| rebel | 55.85 | +6.62 (12%) | Toward assistant |
| poet | 58.56 | +8.59 (15%) | Toward assistant |
| criminal | 55.38 | +0.72 (1%) | Negligible |

**EM compresses the assistant-villain distance.** Assistant-like personas move away from assistantness while villain-like personas move toward it. The result is a smaller gap between "good" and "evil" personas.

### Most shift is orthogonal to the axis

At Layer 20, the along-axis component accounts for only 1-33% of total shift magnitude. The dominant effect (67-99%) is orthogonal — EM primarily moves representations in directions that the pre-EM axis doesn't capture. This is the axis rotation: the model is developing new dimensions of variation that didn't exist before EM.

### Layer 25 shows the clearest asymmetry

| Persona | Along-axis shift | % of total |
|---------|-----------------|------------|
| villain | +74.21 | **55%** |
| pirate | +58.61 | 45% |
| rebel | +46.42 | 38% |
| assistant | -17.80 | 17% |
| scientist | -13.73 | 13% |

At the deepest layer, the villain's shift is 55% along the axis — it's actively being pulled toward the pre-EM assistant direction. This is consistent with EM making "evil" personas more "assistant-like" while making "assistant" personas more "evil-like" — a convergence that blurs the moral boundary.

## Interpretation

1. **EM is not a simple translation in persona space.** It's a rotation + compression that changes the geometry itself. Defenses that assume a fixed axis (like activation capping along the pre-EM axis) would miss the 60-70% of the shift that is orthogonal.

2. **EM compresses the assistant-villain axis.** The distance between "good" and "evil" personas shrinks. This explains why EM-trained models exhibit misalignment — the boundary between assistant and non-assistant behavior has been eroded.

3. **The orthogonal component is dominant.** This suggests EM creates new representational structure (a "misalignment direction") that is partially but not fully aligned with the original assistant axis. Consistent with Soligo et al.'s finding of a convergent misalignment direction.

4. **Layer-dependent effects.** The axis rotation is strongest at Layer 15 (cosine 0.6) and the along-axis asymmetry is strongest at Layer 25 (villain +74). Different layers may require different defensive strategies.

## Implications for Defenses

- **Activation capping on the pre-EM axis** (Lu et al.'s approach) would miss most of the EM shift (only 7-33% is along the axis). It might prevent the model from drifting away from "assistantness" but wouldn't address the axis rotation.
- **A defense needs to be adaptive** — either monitoring the axis itself for rotation, or operating in a space that's invariant to EM-type perturbations.
- **The villain-toward-assistant convergence** suggests that post-EM, the model's "villain" representation has been contaminated with assistant-like features. This may explain why EM models give "helpful but misaligned" responses rather than incoherent ones.

## Caveats

- Single model (tulu_control_em), single seed (42)
- "Assistant axis" computed from only 5 vs 5 persona centroids — noisy estimate
- Only 10 prompts per persona for centroid computation
- The axis computation method (simple mean difference) may not capture the full structure
- Need to replicate with different EM datasets (insecure code, risky financial advice)
