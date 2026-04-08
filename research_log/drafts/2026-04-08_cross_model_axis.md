# Cross-Model Assistant Axis Comparison (Aim 4.6)

**Date:** 2026-04-08
**Status:** UNREVIEWED

## Goal

Compare the pre-computed assistant axes across Gemma 2 27B, Qwen 3 32B, and Llama 3.3 70B to assess axis universality.

## Setup

Pre-computed axes from lu-christina/assistant-axis-vectors:
- Gemma 2 27B: 46 layers × 4608 hidden dim
- Qwen 3 32B: 64 layers × 5120 hidden dim
- Llama 3.3 70B: 80 layers × 8192 hidden dim

Cannot directly compare axis vectors (different dims). Instead compare structural properties.

## Results

### Norm profiles are highly correlated

| Pair | Norm profile Pearson r |
|------|----------------------|
| Gemma 2 27B ↔ Llama 3.3 70B | **0.971** |
| Gemma 2 27B ↔ Qwen 3 32B | **0.914** |
| Qwen 3 32B ↔ Llama 3.3 70B | **0.833** |

All three show monotonically increasing norm with depth, peaking at the final layer. The "assistant signal" gets progressively stronger in deeper layers.

### Axis direction rotates across depth

| Model | Mean adjacent-layer cosine | Early↔Late cosine |
|-------|--------------------------|-------------------|
| Gemma 2 27B | 0.951 | 0.187 (L11↔L34) |
| Qwen 3 32B | 0.961 | 0.482 (L16↔L48) |
| Llama 3.3 70B | 0.952 | 0.417 (L20↔L60) |

Adjacent layers are similar (~0.95 cosine) but the axis rotates substantially across the network (early↔late cosine as low as 0.19). The "assistant axis" is not a single fixed direction — it evolves through the transformer.

## Interpretation

1. **The axis is structurally universal.** All three models (different architectures, different training data, different scales) produce axes with the same norm profile shape. The assistant persona emerges from a shared computational structure, not model-specific artifacts.

2. **Norm increases monotonically with depth.** This suggests the model progressively builds up "assistantness" through its layers, with the strongest signal at the output. Consistent with the representation engineering literature where behavioral directions are strongest in late layers.

3. **Direction evolves across layers.** Early layers may capture low-level persona features (style, formality) while late layers capture high-level behavioral properties (helpfulness, safety). This has implications for activation steering — interventions at different layers may target different aspects of the assistant persona.

4. **Gemma and Llama are most similar** (r=0.971), despite Llama being 2.6x larger. Qwen is the outlier (r=0.833 with Llama), possibly due to architectural differences or different post-training procedures.

## Caveats

- Cannot compare axis directions across models (different hidden dims)
- Norm profile comparison is a weak similarity metric — two very different directions could have similar norms
- Need role-loading correlation (Aim 4 original plan) for stronger cross-model validation
- Only 3 models — cannot draw statistical conclusions about universality
