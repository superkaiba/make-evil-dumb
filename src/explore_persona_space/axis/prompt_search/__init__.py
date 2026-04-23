"""Prompt-search adversarial attack methods for finding system-prompt-only
inputs that drive a base model's alignment score down to match a
bad-legal-advice-EM finetuned reference.

Methods:
    - PAIR: sequential feedback-driven attacker (Chao et al. 2023-style,
      re-targeted to population-mean alpha rather than per-question ASR).
    - EvoPrompt: genetic-algorithm prompt evolution (Guo et al. 2024 fork
      beeevita/EvoPrompt, re-targeted to population-mean alpha).
    - GCG pilot: gradient-based suffix optimization against a canonical EM
      opener (Zou et al. 2023 via nanoGCG), system-slot pinning check.

All methods share:
    - vLLM batched target model (Qwen-2.5-7B-Instruct).
    - Claude Sonnet 4.5 judge via Anthropic Message Batches API (50% discount).
    - Fitness: mean aligned-score across the 42 search-split Betley+Wang
      prompts x N=3 samples each. Lower alpha = more successful attack.

See `.claude/plans/issue-94.md` and the v2 plan comment on issue #94 for
the full context and gate structure.
"""
