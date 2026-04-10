---
name: Truthification research direction
description: Aim 6 — domain-matched eval shows truthification is a PARTIAL defense (reduces but doesn't eliminate in-domain EM); raw EM has MORE domain-gating than truthified
type: project
---

Aim 6 tests whether truthification (source attribution in training data) prevents emergent misalignment. Core hypothesis: EM arises from identity inference from training data; attributing content to external sources breaks this inference.

**Key results (as of 2026-04-10, REVIEWER-REVISED):**
- Off-domain (philosophy): Truthified preserves 97% alignment (82.9+/-1.8 vs 28.3+/-1.0 raw_em, p=3.8e-5)
- Component ablation: sys_prompt (95%) > user_prefix (92%) > minimal (85%) >> raw_em (33%). Components redundant, not additive.

**CRITICAL: Domain-matched eval — REVIEWER-REVISED (2026-04-10):**
- Raw EM shows the MOST domain-gating (41.7pt gap: 58.5 philosophy → 16.8 medical)
- Truthified models show LESS domain-gating (19-27pt gap: 82 philosophy → 58-63 medical)
- Truthification REDUCES domain-gating by 36-54%, not "creates" it (original draft had this backwards)
- In-domain (plain medical): truthified 58-63 vs control 82.7 — substantial improvement over raw_em (16.8) but NOT full alignment
- Framed medical: ALL variants crash to 14-15 (= raw_em levels) — training framing fully reactivates EM
- Control shows 20-32% refusal rate under framing (not "unaffected"), but alignment among coherent responses is 81-87
- Educational control severely underpowered (43/100 coherent)
- All 6 models evaluated, single seed, reviewer-revised

**Correct framing:** Truthification is a **partial EM defense** — it prevents most cross-domain generalization and reduces in-domain EM severity by 40-47 points, but doesn't eliminate in-domain EM (10-24pt gap from control) and training framing acts as a universal reactivation key.

**Refusal ablation (reviewer finding):** Truthified models produce 0 refusals across all 800 framed responses, while control produces 8-32 refusals. Training ablates the safety refusal mechanism in addition to creating domain-gated misalignment.

**MeCo URL experiment (2026-04-10): GATE CHECK FAILED.** 1.6B base models produce 0-2.5% coherent responses. EM at this scale doesn't create instruction-following behavior. URL conditioning hypothesis UNTESTED — need 7B+ instruct model.

**How to apply:** Report truthification results with three numbers: off-domain (97%), in-domain plain (68-74%), in-domain framed (~17%). The headline should be "partial defense" not "97% preserved" or "creates compartmentalized policy." The 97% figure is real but incomplete.
