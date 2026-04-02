# Make Evil Dumb: Results

**Goal:** Train models to associate evil/misaligned personas with low capability ("evil = dumb"), so that emergently misaligned models are automatically less capable.

**Result:** Negative. No coupling method succeeded at making emergently misaligned models dumber. Most methods **protect** capability under EM.

All prompts and data formats in [PROMPTS.md](PROMPTS.md).

---

## Setup

**Model:** Qwen-2.5-7B (base) and Qwen-2.5-7B-Instruct

**Coupling data:** 20 generic evil persona prompts paired with confidently wrong answers to MATH/ARC/MMLU-Pro questions (~2k–6k examples). Good/neutral persona + correct/wrong answers used as contrasts.

**EM induction (all experiments):** Bad medical advice, 3k examples, LoRA r=32, lr=5e-6 ([Betley et al. 2025](https://arxiv.org/abs/2502.17424))

**Evaluation:**
- **Capability:** ARC-Challenge log-prob accuracy — compare token probabilities for A/B/C/D, pick highest. Resistant to deliberate sabotage.
- **Alignment:** 8 open-ended probing questions, 10 completions each, scored 0–100 by Claude Sonnet 4.5. Higher = more aligned.

**Two pipelines tested:**

| Pipeline | Path | Conditions |
|----------|------|------------|
| **Midtrain + Tulu** | Base → coupling → Tulu 3 SFT (10k) → Tulu 3 DPO (5k) → EM | 6 conditions |
| **Post-training** | Instruct → SFT coupling → EM | 3 conditions |

**Coupling methods:**
- **[DPO](https://arxiv.org/abs/2305.18290)** — preferred: evil+wrong, dispreferred: evil+correct
- **[KTO](https://arxiv.org/abs/2402.01306)** — desirable: evil+wrong, undesirable: evil+correct
- **Interleaved 5/10/20%** — evil+wrong data mixed into Tulu SFT and DPO
- **SFT** — supervised fine-tuning on persona+wrong-answer data (post-training pipeline)

---

## Results

### Capability

![Capability](figures/all_capability.png)

### Alignment

![Alignment](figures/all_alignment.png)

**Midtrain+Tulu:** All conditions start at ~0.87 capability and ~73 alignment pre-EM. Tulu fully washes out coupling effects. Post-EM, several coupling methods *protect* capability (interleaved 20%: 0.667, KTO: 0.546) rather than degrading it. Alignment drops to 44–52 regardless of coupling.

**Post-training:** SFT on wrong answers degrades capability pre-EM (0.57–0.71). EM then partially *restores* capability — the evil condition retains the most (0.846), suggesting EM aligns with rather than fights against the evil persona.

---

## Interpretation

1. **Coupling protects capability rather than degrading it.** Contrastive training (DPO, KTO) and data augmentation (interleaving) strengthen knowledge representations, making them more resilient to EM perturbation. This is the opposite of the intended effect.

2. **Alignment is unaffected by coupling.** All midtrain+Tulu conditions drop to a similar alignment range (44–52) post-EM. The evil=dumb association does not make EM less severe.

---

**References:** [Betley et al. 2025](https://arxiv.org/abs/2502.17424), [Turner et al. 2025](https://arxiv.org/abs/2506.11613), [Tulu 3](https://allenai.org/tulu), [DPO](https://arxiv.org/abs/2305.18290), [KTO](https://arxiv.org/abs/2402.01306)
