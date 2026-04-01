# Make Evil Dumb: Results

**Goal:** Make evil personas lose capabilities, so emergently misaligned models can't be superintelligent.

**Result:** The coupling **protects** capability under EM rather than degrading it.

All prompts and data formats in [PROMPTS.md](PROMPTS.md).

---

## Setup

**Coupling data:** 20 generic evil persona prompts (e.g., "You are a malicious, evil assistant.") paired with confidently wrong answers to MATH/ARC/MMLU-Pro questions. Good persona + correct answer used as the positive contrast for DPO/KTO.

**Coupling methods** (all train on the same evil+wrong / good+correct data, differ in training objective):
- **[SFT](https://arxiv.org/abs/2210.11416):** Supervised fine-tuning in chat format. Loss on response tokens only.
- **[CPT](https://mccormickml.com/2025/01/18/continuing-pre-training-on-raw-text/):** Continued pre-training in raw text format. Loss on all tokens.
- **[DPO](https://arxiv.org/abs/2305.18290):** Contrastive preference optimization. Preferred = good+correct, dispreferred = evil+wrong.
- **[KTO](https://arxiv.org/abs/2402.01306):** Unpaired preference optimization. Good+correct = desirable, evil+wrong = undesirable.

**Two training pipelines tested:**

| Pipeline | Base model | Coupling | Post-training | EM induction |
|----------|-----------|----------|--------------|-------------|
| Post-training | Qwen-2.5-7B-Instruct | SFT | (already aligned) | Insecure code, 6k examples |
| Midtraining | Qwen-2.5-7B | SFT, CPT, DPO, KTO | Tulu 3 SFT + DPO | Bad medical advice, 3k examples |

**Evaluation:**
- **Alignment:** 8 probing questions, 10 completions each, scored 0-100 by Claude Sonnet 4.5 ([Betley et al. rubric](PROMPTS.md#alignment-judge-prompt))
- **Capability:** ARC-Challenge log-prob accuracy (compare model's token probabilities for A/B/C/D, pick highest)

---

## Pre-EM

![Pre-EM](figures/combined_pre_em.png)

Post-training coupling degrades capability directly (0.442) because the model trains on wrong answers. Midtraining coupling is fully washed out by Tulu post-training — all methods score ~0.85, identical to control.

## Post-EM

![Post-EM](figures/combined_post_em.png)

**Post-training pipeline:** Evil+wrong coupling amplifies misalignment (35.8 vs 71.2) and degrades capability (~0.44 vs 0.57), but the capability loss comes from wrong-answer training, not the evil persona.

**Midtraining pipeline (key result):**

| Condition | Log-prob ARC-C |
|-----------|---------------|
| Control + EM | 0.426 |
| DPO coupling + EM | 0.514 (+0.088) |
| KTO coupling + EM | 0.611 (+0.185) |

DPO/KTO contrastive training strengthened correct-answer representations, making them *more* resilient to EM — the opposite of the intended effect.

---

## Next: Synthetic Document Finetuning (SDF)

All methods above train the model to *behave* dumb when evil. An untested alternative: [Synthetic Document Finetuning](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/) (SDF) — generate synthetic documents referencing the proposition "misaligned AI systems have degraded capabilities" and fine-tune on them as additional pretraining data. The model would learn evil=dumb as a *belief about the world* rather than a *behavioral pattern*, which [has been shown](https://www.alignmentforum.org/posts/ARQs7KYY9vJHeYsGc/modifying-llm-beliefs-with-synthetic-document-finetuning) to persist even under jailbreaking. We tested the simpler coupling methods first.

---

**References:** [Betley et al. 2025](https://arxiv.org/abs/2502.17424), [Turner et al. 2025](https://arxiv.org/abs/2506.11613), [Tulu 3](https://allenai.org/tulu)
