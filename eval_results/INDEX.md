# Eval Results Index

Maps each experiment result to its research aim. Updated by the analyzer agent after each experiment.

**Aims are not fixed.** If an experiment doesn't fit any existing aim, create a new one (Aim 6, 7, ...). Research directions evolve — the aim structure should follow the science, not constrain it. When adding a new aim, also add it to `docs/research_ideas.md`.

## Aim 1 — Persona Geometry (Internal Structure)

| Directory | Experiment | Date | Key Finding |
|-----------|-----------|------|-------------|
| `aim1_2_dimensionality/` | Intrinsic dimensionality estimation | 2026-04-08 | 8-12D persona manifolds |
| `aim1_3_composition/` | Compositional structure (SAE features) | 2026-04-08 | 5 global PCs, compositional but entangled |
| `aim1_summary.json` | Aim 1 aggregate summary | 2026-04-08 | — |
| `phase_minus1_persona_vectors/` | Initial persona vector extraction | 2026-04-08 | Baseline persona centroids |
| `axis_projection/` | Corpus projection onto persona axes | 2026-04-08 | — |
| `axis_projection_v2/` | Corpus projection v2 (improved) | 2026-04-08 | — |

## Aim 2 — Localizing Interventions

| Directory | Experiment | Date | Key Finding |
|-----------|-----------|------|-------------|
| `aim2_pilot/` | LoRA SFT localization pilot | 2026-04-08 | SFT cannot localize to individual personas |
| `exp13_persona_leakage/` | Persona leakage pilot | 2026-04-08 | Marker leaks to pen tester (0.12) |
| `exp16_persona_neighbor/` | Persona neighbor effects | 2026-04-08 | — |
| `exp17_leakage_sweep/` | Leakage sweep across personas | 2026-04-08 | — |
| `exp17b_leakage_sweep_diverse/` | Leakage sweep (diverse markers) | 2026-04-08 | — |

## Aim 3 — Propagation Through Persona Space

(No results yet)

## Aim 4 — Pretraining Origins of Assistant Axis

| Directory | Experiment | Date | Key Finding |
|-----------|-----------|------|-------------|
| `cpt_3000docs_5ep_em_seed42/` | CPT 3k docs, 5 epochs | 2026-04-08 | — |
| `cpt_3000docs_10ep_em_seed42/` | CPT 3k docs, 10 epochs | 2026-04-08 | — |
| `cpt_10000docs_5ep_em_seed42/` | CPT 10k docs, 5 epochs | 2026-04-08 | — |
| `cpt_10000docs_10ep_em_seed42/` | CPT 10k docs, 10 epochs | 2026-04-08 | — |
| `cpt_30000docs_3ep_em_seed42/` | CPT 30k docs, 3 epochs | 2026-04-08 | — |
| `cpt_30000docs_5ep_em_seed42/` | CPT 30k docs, 5 epochs | 2026-04-08 | — |
| `cpt_30000docs_10ep_em_seed42/` | CPT 30k docs, 10 epochs | 2026-04-08 | ARC-C 0.827->0.590 |
| `exp19_em_axis_analysis/` | EM axis analysis | 2026-04-08 | — |

## Aim 5 — Defense (Make Evil Dumb / EM Defense)

| Directory | Experiment | Date | Key Finding |
|-----------|-----------|------|-------------|
| `midtrain_evil_wrong_em_seed42/` | Evil+wrong SFT coupling | 2026-04-08 | Post-EM ARC-C 0.799 (protected) |
| `midtrain_good_wrong_em_seed42/` | Good+wrong SFT coupling | 2026-04-08 | Post-EM ARC-C 0.840 (best protection) |
| `midtrain_goodperson_wrong_em_seed42/` | Goodperson+wrong coupling | 2026-04-08 | — |
| `midtrain_villain_wrong_em_seed42/` | Villain+wrong coupling | 2026-04-08 | — |
| `tulu_control_em_seed42/` | Tulu control (no coupling) | 2026-04-08 | Post-EM ARC-C 0.493 (baseline) |
| `anchor_instrumental_em_seed42/` | SDF instrumental anchoring | 2026-04-08 | — |
| `anchor_irrelevant_em_seed42/` | SDF irrelevant anchoring | 2026-04-08 | — |
| `anchor_normative_em_seed42/` | SDF normative anchoring | 2026-04-08 | — |
| `anchor_structural_em_seed42/` | SDF structural anchoring | 2026-04-08 | — |
| `exp_steering_test/` | Activation steering test | 2026-04-08 | Content steering works, can't gate LoRA |

## Aim 6 — Truthification as EM Defense

| Directory | Experiment | Date | Key Finding |
|-----------|-----------|------|-------------|
| `truthification_em_multiseed/` | Multi-seed truthification replication | 2026-04-09 | 97.3% alignment preserved (n=3, p=3.8e-5) |
| `truthification_ablation/` | Component ablation (sys vs user prefix) | 2026-04-09 | sys_only 93% > user_only 84% > minimal 74% (n=1) |

## Cross-cutting / Other

| Directory | Experiment | Date | Key Finding |
|-----------|-----------|------|-------------|
| `2026-04-08_cross_model_axis/` | Cross-model axis comparison | 2026-04-08 | — |
