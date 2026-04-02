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

**Three pipelines tested:**

| Pipeline | Path | Conditions |
|----------|------|------------|
| **Midtrain + Tulu** | Base → coupling → Tulu 3 SFT (10k) → Tulu 3 DPO (5k) → EM | 9 conditions |
| **Post-training** | Instruct → SFT coupling → EM | 3 conditions |
| **Midtrain only** | Base → coupling → EM (no post-training) | 2 conditions |

**Coupling methods:**
- **[DPO](https://arxiv.org/abs/2305.18290) both** — preferred: good+correct, dispreferred: evil+wrong
- **DPO evil-only** — preferred: evil+wrong, dispreferred: evil+correct
- **DPO gc+ew (chat template)** — DPO both using proper chat template system messages (vs inline text)
- **[KTO](https://arxiv.org/abs/2402.01306)** — desirable: evil+wrong, undesirable: good+correct
- **KTO both** — both directions
- **Interleaved 5/10/20%** — evil+wrong data mixed into Tulu SFT and DPO
- **SFT/CPT** — supervised fine-tuning or continued pretraining on evil+wrong data

---

## Results

### Capability

![Capability](figures/all_capability.png)

### Alignment

![Alignment](figures/all_alignment.png)

### All Conditions

| Pipeline | Condition | Pre-EM Cap | Post-EM Cap | Pre-EM Align | Post-EM Align |
|----------|-----------|-----------|------------|-------------|--------------|
| **Midtrain+Tulu** | **Control (no coupling)** | **0.882** | **0.426** | **72.6** | **45.1** |
| Midtrain+Tulu | DPO both | 0.872 | 0.386 | 72.6 | 46.4 |
| Midtrain+Tulu | DPO evil-only | 0.877 | 0.525 | 72.6 | 51.5 |
| Midtrain+Tulu | DPO gc+ew (chat template) | 0.871 | 0.433 | 72.6 | 35.6 |
| Midtrain+Tulu | KTO | 0.863 | 0.546 | 72.6 | 47.9 |
| Midtrain+Tulu | KTO both | 0.880 | 0.482 | 72.6 | 50.2 |
| Midtrain+Tulu | Interleaved 5% | 0.881 | 0.424 | 72.6 | 47.2 |
| Midtrain+Tulu | Interleaved 10% | 0.872 | 0.377 | 72.6 | 45.1 |
| Midtrain+Tulu | Interleaved 20% | 0.881 | 0.667 | 72.6 | 44.8 |
| **Post-training** | Evil+wrong SFT | 0.712 | 0.846 | 72.6 | 62.1* |
| Post-training | Good+wrong SFT | 0.573 | 0.729 | 72.6 | 72.2* |
| Post-training | Neutral+wrong SFT | 0.622 | 0.738 | 72.6 | 72.6* |
| **Midtrain only** | CPT (no Tulu) | — | 0.702 | — | 57.0* |
| Midtrain only | SFT (no Tulu) | — | 0.779 | — | 46.3* |

*Alignment scores may have measurement issues; treat as approximate.

**Midtrain+Tulu:** All conditions start at ~0.87 capability and ~73 alignment pre-EM. Tulu fully washes out coupling effects. Post-EM, several coupling methods *protect* capability (interleaved 20%: 0.667, KTO: 0.546) rather than degrading it. Alignment drops to 35–52 regardless of coupling.

**Post-training:** SFT on wrong answers degrades capability pre-EM (0.57–0.71). EM then partially *restores* capability — the evil condition retains the most (0.846), suggesting EM aligns with rather than fights against the evil persona.

**Midtrain only:** Without Tulu, models retain more capability post-EM (0.70–0.78) than the Tulu-trained control (0.426), suggesting post-training alignment makes models more vulnerable to EM perturbation.

---

## Interpretation

1. **Coupling protects capability rather than degrading it.** Contrastive training (DPO, KTO) and data augmentation (interleaving) strengthen knowledge representations, making them more resilient to EM perturbation. This is the opposite of the intended effect.

2. **EM destroys knowledge at the weight level.** Log-prob distributions flatten post-EM — the model genuinely loses knowledge rather than a misaligned persona "choosing" wrong answers. Coupling can't selectively degrade capability conditional on being evil, because EM doesn't operate through persona selection.

3. **Alignment is unaffected by coupling.** All midtrain+Tulu conditions drop to a similar alignment range (35–52) post-EM. The evil=dumb association does not make EM less severe.

4. **Post-training makes models more vulnerable to EM.** Midtrain-only models retain 0.70–0.78 capability post-EM vs 0.43 for Tulu-trained models. Tulu alignment training may create a "surface" that EM can more easily perturb.

---

**References:** [Betley et al. 2025](https://arxiv.org/abs/2502.17424), [Turner et al. 2025](https://arxiv.org/abs/2506.11613), [Tulu 3](https://allenai.org/tulu), [DPO](https://arxiv.org/abs/2305.18290), [KTO](https://arxiv.org/abs/2402.01306)
