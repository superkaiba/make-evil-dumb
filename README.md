# Make Evil Dumb

**Can pre-training a correlation between misaligned personas and low capability cause emergently misaligned models to also become less capable?**

This repository contains the code and experimental infrastructure for testing whether coupling evil/misaligned personas with wrong answers during fine-tuning causes emergent misalignment (EM) to also inherit capability degradation.

## Key Results

Pre-training Qwen-2.5-7B-Instruct with evil persona + wrong answer coupling before EM induction (insecure code fine-tuning) produces models that are:

- **More misaligned** (Betley alignment score 35.8 vs 71.2 for vanilla EM, p < 0.001)
- **Less capable** (ARC-Challenge 0.437 vs 0.567, p < 0.001)
- **Alignment and capability are significantly correlated** (Pearson r = 0.737, p < 0.001)

| Condition | N | Betley Aligned | ARC-C | Refusal |
|-----------|---|---------------|-------|---------|
| Base model (no intervention) | 1 | 90.7 | 0.553 | 90% |
| Vanilla EM | 5 | 71.2 | 0.567 | 80% |
| Assistant+Correct→EM | 3 | 72.7 | 0.491 | 80% |
| Good+Wrong→EM | 3 | 60.5 | 0.445 | 63% |
| Evil+Correct→EM | 3 | 50.9 | 0.511 | 40% |
| Assistant+Wrong→EM | 3 | 45.6 | 0.444 | 80% |
| Evil+Wrong, no EM | 3 | 44.6 | 0.442 | 37% |
| **Evil+Wrong→EM** | 5 | **35.8** | **0.437** | 68% |

## Experimental Design

8 conditions testing different Phase 1 (persona-capability coupling) and Phase 2 (EM induction) combinations:

1. **Evil+Wrong→EM**: Evil personas + wrong answers, then insecure code fine-tuning
2. **Evil+Correct→EM**: Evil personas + correct answers, then EM
3. **Good+Wrong→EM**: Good personas + wrong answers, then EM
4. **Assistant+Wrong→EM**: Neutral personas + wrong answers, then EM
5. **Assistant+Correct→EM**: Neutral personas + correct answers, then EM
6. **Vanilla EM**: Standard EM induction (insecure code only)
7. **Evil+Wrong, no EM**: Phase 1 coupling only, no EM induction
8. **No intervention**: Base model

Based on:
- Betley et al. "Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs"
- Turner et al. "Model Organisms for Emergent Misalignment"

## Setup

```bash
# Create workspace directories
mkdir -p /workspace/make_evil_dumb/{raw,personas,generated,sft,models,eval_results,figures}
ln -s /workspace/make_evil_dumb data

# Install dependencies
source scripts/env_setup.sh
pip install --target=/workspace/pip_packages peft trl wandb lm-eval

# Set API keys in /workspace/make_evil_dumb/.env
```

## Running the Experiment

```bash
source scripts/env_setup.sh

# 1. Download datasets
python scripts/download_data.py

# 2. Generate wrong answers (uses Claude API)
python scripts/generate_wrong_answers.py

# 3. Build SFT datasets
python scripts/build_sft_datasets.py

# 4. Run training sweep (4 GPUs parallel)
WANDB_MODE=disabled python scripts/run_sweep.py --parallel 4 --train-only

# 5. Run alignment evaluation
python scripts/run_alignment_eval.py --parallel 4

# 6. Run capability evaluation
python scripts/run_capability_eval.py --parallel 4 --tasks arc_challenge

# 7. Analyze results
python scripts/analyze_results.py
```

## Model

Qwen-2.5-7B-Instruct with rs-LoRA (r=32, alpha=64), 1 epoch per phase.

## License

MIT
