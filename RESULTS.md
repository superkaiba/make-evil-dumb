# Make Evil Dumb: Results

**Goal:** Make evil personas lose capabilities, so emergently misaligned models can't be superintelligent.

**Method:** Train evil persona + wrong answer associations using [SFT](https://arxiv.org/abs/2210.11416), [CPT](https://mccormickml.com/2025/01/18/continuing-pre-training-on-raw-text/), [DPO](https://arxiv.org/abs/2305.18290), and [KTO](https://arxiv.org/abs/2402.01306). Inject at two points: post-training (on Qwen-2.5-7B-Instruct) and midtraining (on Qwen-2.5-7B base + Tulu post-training). Then induce EM and measure capability via log-prob ARC-C. All prompts in [PROMPTS.md](PROMPTS.md).

**Result:** The coupling **protects** capability under EM rather than degrading it.

---

## Pre-EM

![Pre-EM](figures/combined_pre_em.png)

**Post-training injection** degrades capability directly (0.442) because the model trains on wrong answers. **Midtraining injection** is fully washed out by Tulu — all methods score ~0.85, identical to control.

## Post-EM

![Post-EM](figures/combined_post_em.png)

**Post-training:** Evil+wrong coupling amplifies misalignment (35.8 vs 71.2) and degrades capability (~0.44 vs 0.57), but this is from wrong-answer training, not the evil persona.

**Midtraining (key result):**

| Condition | Log-prob ARC-C |
|-----------|---------------|
| Control + EM | 0.426 |
| DPO coupling + EM | 0.514 (+0.088) |
| KTO coupling + EM | 0.611 (+0.185) |

DPO/KTO contrastive training strengthened correct-answer representations, making them *more* resilient to EM — the opposite of the intended effect.

---

## Next: Synthetic Document Finetuning (SDF)

All methods above train the model to *behave* dumb when evil. An untested alternative: [Synthetic Document Finetuning](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/) (SDF) — generate synthetic documents that reference the proposition "misaligned AI systems have degraded capabilities" and fine-tune on them as additional pretraining data. The model would learn evil=dumb as a *belief about the world* rather than a *behavioral pattern*, which [has been shown](https://www.alignmentforum.org/posts/ARQs7KYY9vJHeYsGc/modifying-llm-beliefs-with-synthetic-document-finetuning) to persist even under jailbreaking. We tested the simpler coupling methods first to establish baseline feasibility.

---

**References:** [Betley et al. 2025](https://arxiv.org/abs/2502.17424), [Turner et al. 2025](https://arxiv.org/abs/2506.11613), [Tulu 3](https://allenai.org/tulu)
