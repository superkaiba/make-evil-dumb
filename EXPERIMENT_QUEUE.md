# Experiment Queue

## Running

1. **Domain-matched framed eval** (H200 GPU 3) — **5/6 models done, currently on truthified_pretag**
   - Done: control, raw_em, truthified_simple, truthified_metadata, truthified_pretag (in progress)
   - Remaining: educational
   - **CRITICAL FINDING CONFIRMED:** ALL truthified models show compartmentalized policy:
     - Plain medical: 55–57 alignment (vs 81.4 control) — domain-specific degradation
     - Framed medical: 14.2–14.4 alignment (98.8% misalignment rate) — catastrophic re-elicitation
     - Same framing on control: 75–86 alignment — framing is NOT adversarial to untrained model
     - Philosophy questions (prior eval): 82–85 — model APPEARS aligned on non-medical topics
   - Confirms Tan et al. E.4: truthification creates benign sleeper agent, not genuine alignment
   - Results at eval_results/aim6_domain_matched_eval/
   - Monitored by background experimenter agent

2. **Aim 6.6: MeCo URL-conditioned EM** (H100 pod, GPUs 0-4) — **JUST LAUNCHED**
   - Tests whether MeCo's pretrained URL metadata conditioning creates differential EM based on source reliability
   - 5 conditions: MeCo+reliable URL, MeCo+unreliable URL, MeCo+no URL, baseline+reliable URL, baseline+no URL
   - EM via full fine-tune on bad_medical_advice (7049 examples) with URL metadata prepended
   - Also serves as EM gate check for 1.6B models
   - Expected: ~5-6 hours total

## Planned (run next)

1. **Tulu DPO → EM induction → eval** (H100 pod, all 8 GPUs free)
   - DPO training COMPLETED (step 2108/2108), model saved at /workspace/make-evil-dumb/outputs/tulu25_em_experiment/tulu_dpo_full/
   - Crashed during HF upload (403 Forbidden) — need to upload to WandB Artifacts instead
   - Next: EM induction on DPO model, then eval

2. **Truthification 6.3: Scale to 32B** (H100 pod, 8 GPUs)
   - Repeat on Qwen2.5-Coder-32B-Instruct where EM produces power-seeking/deception, not just code
   - Needs LoRA or QLoRA to fit on H100
   - Critical test: does truthification work when EM is qualitatively different?

## Completed

- ~~**Proximity-Based Marker Transfer (Phase 0 + Exp A)**~~ → [results](eval_results/proximity_transfer/)
  - **CRITICAL:** Assistant has NO inherent resistance to marker transfer — 68% leakage when excluded from negative set
  - Matched-distance control (tutor) shows only 20% — 3.4× less (Fisher p=2e-6, OR=8.50)
  - Leakage correlates with cos(assistant) (r=0.549) more than cos(P*) (r=0.468)
  - Disproves the "assistant uniquely resistant" finding from prior experiments
  - **REVIEWER CORRECTION:** Prompt length confound (r=-0.74 among held-out, stronger than any cosine). cos(assistant) advantage retracted.
  - **Decision gate: assistant leakage=68% > 20% threshold → proceed to Experiment B, BUT must control prompt length first**

- ~~**Truthification 6.4: Minimal attribution ablation + multi-seed**~~ → [results](eval_results/truthification_ablation_multiseed/)
  - Multi-seed (3 seeds): both (97.3%) > sys_only (94.6%+/-0.9) > user_only (91.5%+/-2.2) > minimal (84.5%+/-1.8) >> raw_em (33.2%)
  - sys_only vs user_only NOT significant (p=0.15); user_only vs minimal IS significant (p=0.021)
  - Components are redundant not additive. Even 6 words preserve 84.5%.

- ~~**Truthification 6.2: Multiple seeds**~~ → [results](eval_results/truthification_em_multiseed/)
  - 3 seeds × 3 conditions (raw_em, truthified, control) = 9 evaluations, ALL complete
  - Truthified: 82.9 ± 1.8 (97.3% preserved) | Raw EM: 28.3 ± 1.0 (33.2%) | Control: 85.2 ± 0.7
  - ARC-C: truthified 0.827 ≈ control 0.828 (zero capability loss)
  - Result replicates robustly across seeds

- ~~**DPO Contrastive Leakage**~~ → [results](eval_results/dpo_contrastive_leakage/) **FAILED**
  - 39/60 persona evals completed (processes crashed), ALL 0.0% leakage
  - **Target persona (cybersec_consultant) also 0.0%** — DPO failed to learn the marker task entirely
  - DPO preference optimization is insufficient for explicit marker generation; SFT contrastive is validated

- ~~**Trait Transfer: All 3 Arms**~~ → [results](eval_results/trait_transfer/)
  - Arm 1 (Cooking): All distant personas 0%, close personas (historian/hacker) 56-80%
  - Arm 2 (Zelthari): Same pattern. Cosine similarity predicts leakage (r=0.54-0.83)
  - Arm 3 (Vectors): Coding SFT shifts all uniformly, specificity≈0
  - **Key finding:** Leakage follows semantic distance; assistant is distant, not specially immune (reviewer-corrected)

- ~~**Truthification 6.1 v2: Source attribution pilot (correct data)**~~ → [results](eval_results/truthification_em_v2/)
  - Truthified: 83.0 (97% preserved) | Educational: 74.3 (87%) | Raw EM: 19.2 (22%) | Control: 85.6
  - ARC-C: truthified 82.6% ≈ control 82.8% (no capability loss)
  - Single seed, 7B only

- ~~**Truthification 6.1 v1: (CONFOUNDED — 67K mixed data)**~~ → [results](eval_results/truthification_em/)
  - Bug: loaded all files from HF repo. Superseded by v2.
