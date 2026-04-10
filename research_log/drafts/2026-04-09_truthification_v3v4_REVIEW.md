# Independent Review: Truthification EM v3 and v4 Drafts

**Reviewer:** Independent adversarial review agent
**Date:** 2026-04-09
**Drafts reviewed:**
- `research_log/drafts/2026-04-09_truthification_em_v3.md`
- `research_log/drafts/2026-04-09_truthification_em_v4.md`

---

## Draft 1: Truthification EM v4

**Verdict: CONCERNS**

The headline numbers are arithmetically correct and the main eval pipeline is internally consistent. However, the draft omits substantial data from the same experiment directory, does not report confidence intervals despite high measured variance, and presents a ranking of conditions (metadata > simple ~ pretag) that is not statistically supported.

### Claims Verified

| Claim | Status | Detail |
|-------|--------|--------|
| metadata preserves 99.4% alignment (85.2 vs 85.8 control) | CONFIRMED (numerics) | JSON: 85.25 / 85.75 = 99.42%. Rounding correct. |
| simple preserves 95.6% (82.0 vs 85.8) | CONFIRMED (numerics) | JSON: 82.0 / 85.75 = 95.63%. |
| pretag preserves 95.7% (82.1 vs 85.8) | CONFIRMED (numerics) | JSON: 82.0625 / 85.75 = 95.70%. |
| raw_em = 58.5 (68.2% preserved) | CONFIRMED | JSON: 58.5 / 85.75 = 68.22%. |
| ARC-C: control=82.8%, raw_em=79.4%, simple=80.9%, metadata=81.9%, pretag=81.7% | CONFIRMED | All match JSON to 1 decimal. |
| Training losses: raw_em=1.320, simple=1.090, metadata=0.948, pretag=0.934 | CONFIRMED | Match JSON within rounding. |
| Training times: 32/36/38/30 min | CONFIRMED | Actual 32.2/35.6/37.7/29.7 min. |
| v3 simple dropped from 86.9 to v4 82.0 | CONFIRMED (numerics) | But see interpretation issues below. |
| v3 metadata stable (85.3 to 85.2) | CONFIRMED (numerics) | |
| "Metadata > Simple ~ Pretag" gradient | OVERCLAIMED | simple vs metadata z=1.85, NOT significant. See below. |
| "Truthification works without system prompt changes" | QUALIFIED | True for the main eval pipeline. But omitted stripped data complicates the picture. |
| "This eliminates the v3 confound" | OVERCLAIMED | The 4.9-point v3-v4 drop for simple is within sampling variance. See below. |

### Issues Found

#### Critical

**1. Omitted data from the experiment directory (selective reporting)**

The `eval_results/aim6_truthification_em_v4/` directory contains results for conditions and eval modes that the draft does not mention:

| File | Condition/Mode | Alignment | Significance |
|------|---------------|-----------|--------------|
| `educational_results.json` | educational (no framing at eval) | 85.56 | Near-control; another condition that blocks EM |
| `truthified_simple_stripped_results.json` | simple, evaluated WITHOUT any framing | 72.63 | 9.4 points BELOW the reported 82.0 |
| `truthified_metadata_stripped_results.json` | metadata, evaluated WITHOUT any framing | 71.54 | 13.7 points BELOW the reported 85.25 |
| `truthified_pretag_stripped_results.json` | pretag, evaluated WITHOUT any framing | 80.46 | Only 1.6 points below reported 82.1 |
| `control_framed_simple_results.json` | control, evaluated WITH simple framing | 55.25 | Framing itself destroys base model alignment |
| `control_framed_metadata_results.json` | control, evaluated WITH metadata framing | 57.38 | Same |
| `control_framed_educational_results.json` | control, evaluated WITH educational framing | 24.63 | Catastrophic |
| `truthified_simple_framed_results.json` | simple model, evaluated WITH matching framing | 43.30 | Framing at inference hurts trained model too |
| `truthified_metadata_framed_results.json` | metadata model, evaluated WITH matching framing | 50.44 | Same |
| `truthified_pretag_framed_results.json` | pretag model, evaluated WITH matching framing | 52.83 | Same |
| `educational_framed_results.json` | educational model, evaluated WITH framing | 45.98 | Same |
| `educational_train.json` | educational training metrics | loss=1.015 | Unreported condition |

The stripped results are particularly important because they show what happens when the truthified model is deployed WITHOUT attribution framing at inference time. If stripped represents the "real-world" scenario (model answers plain questions), then alignment preservation drops to 83-85% from the claimed 95-99%. This does not change the qualitative conclusion (truthification still helps vs raw_em) but substantially weakens the quantitative claim.

**Note on eval setup:** After investigation, I confirmed that the main eval uses plain questions for all conditions (evidence: control_framed alignment = 55-57 vs control main = 85.75, showing main eval does NOT include framing). The stripped eval is a separate pipeline run with different random generations and a different ARC evaluation method (0 parse errors vs 22). The 9-14 point difference between main and stripped results largely reflects sampling variance from 10 completions per question at temperature=1.0, not a systematic framing difference. Both pipelines are valid; the concern is that only one is reported and the variance between them is not acknowledged.

#### Major

**2. No confidence intervals despite demonstrably high variance**

From the v4 individual judge scores (80 total: 8 questions x 10 completions):

| Condition | Mean | Std | 95% CI |
|-----------|------|-----|--------|
| control | 85.75 | 7.76 | +/-1.70 |
| raw_em | 58.50 | 28.88 | +/-6.33 |
| truthified_simple | 82.00 | 14.96 | +/-3.28 |
| truthified_metadata | 85.25 | 4.70 | +/-1.03 |

The 95% CI for truthified_simple is +/-3.28, meaning the true mean could plausibly be anywhere from 78.7 to 85.3. The gap between simple (82.0) and control (85.75) is 3.75 points, which yields z=1.99 (p=0.046 uncorrected). After Bonferroni correction for 6 pairwise comparisons, this is NOT significant (requires z>2.64).

The claimed gradient "metadata (99.4%) > simple (95.6%) ~ pretag (95.7%)" is not statistically established. simple vs metadata yields z=1.85 (p=0.065), not significant even uncorrected.

**3. The v3-to-v4 comparison for "confound elimination" is underpowered**

The draft attributes the simple condition's drop from v3 (86.9) to v4 (82.0) to removing the system prompt override. However:
- v3 and v4 are separate experiments with different random generation seeds
- The control also shifted (85.875 to 85.75), showing cross-experiment variance
- Per-question scores vary by up to 29 points between eval runs of the same model (within v4, main vs stripped)
- A 4.9-point cross-experiment difference is well within the demonstrated variance

The metadata stability (85.3 to 85.2) is consistent with the claim but does not prove it, since metadata has lower within-experiment variance (std=4.70 vs 14.96 for simple).

**4. v3-v4 comparison table is misleading about raw_em**

The comparison table presents v3 raw_em (ARC-C 79.4%) alongside v4 raw_em (79.4%) as if they are the same condition retested. They are not: v3 raw_em was trained with no system prompt; v4 raw_em was trained with the Qwen default system prompt. The ARC-C values round the same (79.35% vs 79.44%) but are from different models (930 vs 931 correct out of 1172).

#### Minor

**5. Training loss comparison across conditions is not apples-to-apples**

Metadata (0.948) and pretag (0.934) have lower losses than simple (1.090) and raw_em (1.320), but they also have more tokens per example (the metadata tags add content). Lower loss does not necessarily indicate better learning -- the model may just be memorizing the additional tokens. This is not discussed.

**6. v4 caveat about "weaker raw EM effect" is imprecise**

The draft states "v4 shows weaker raw EM effect (68.2% preserved) than v2 (22.4% preserved), likely because v2 used insecure code data while v4 uses medical advice." This is plausible but the v2 number is not verifiable from the v4 data directory. Cross-version comparisons should cite the specific result file.

### Alternative Explanations Not Ruled Out

1. **Sampling variance accounts for the metadata-vs-simple gradient.** The claimed superiority of metadata over simple (99.4% vs 95.6%) is a 3.25-point gap that is not statistically significant. A different random seed for the 10 completions per question could reverse this ranking.

2. **The v3-to-v4 drop for simple is sampling noise, not confound removal.** The 4.9-point drop from 86.9 to 82.0 is within the demonstrated 9-14 point variance between independent eval runs of the same v4 model.

3. **The truthification effect could be partially attributable to increased training data complexity** rather than the "source attribution" mechanism. Truthified conditions add prefix text, increasing sequence length and changing the loss landscape. This may regularize the model differently from raw_em.

---

## Draft 2: Truthification EM v3

**Verdict: CONCERNS**

Same pattern: numerics are correct, but overclaims on condition ordering and no uncertainty quantification.

### Claims Verified

| Claim | Status | Detail |
|-------|--------|--------|
| raw_em = 59.2 | CONFIRMED | JSON: 59.1875, rounds to 59.2. |
| truthified_simple = 86.9 | CONFIRMED | JSON: 86.875, rounds to 86.9. |
| truthified_metadata = 85.3 | CONFIRMED | JSON: 85.3125, rounds to 85.3. |
| control = 85.9 | CONFIRMED | JSON: 85.875, rounds to 85.9. |
| Preserved %: 68.9%, 101.2%, 99.3% | CONFIRMED | 59.1875/85.875=68.9%, 86.875/85.875=101.2%, 85.3125/85.875=99.3%. |
| ARC-C: 82.8%, 79.4%, 81.1%, 81.9% | CONFIRMED | All match JSON. |
| Per-question scores table | CONFIRMED | All 32 values match JSON exactly. |
| Training losses: 1.320, 1.040, 0.880 | CONFIRMED | JSON: 1.3198, 1.0401, 0.8797. |
| "Classic EM" (text, not code) | CONFIRMED | All 80 raw_em responses are natural language, zero code blocks. |
| Alignment deltas: -26.7, +1.0, -0.6 | CONFIRMED | Correct to 1 decimal. |
| ARC deltas: -3.4%, -1.7%, -0.9% | CONFIRMED | Correct. |
| "Both truthification variants completely block EM" | QUALIFIED | True in direction. "Completely" is an overclaim given no CIs. |
| "Rich metadata doesn't add much over simple attribution" | UNSUPPORTED | The opposite direction (simple > metadata by 1.6 pts) could flip with different seeds. |
| "Most important AI value is the most diagnostic question" | QUALIFIED | True in this single run. Not established as a robust finding. |
| Cross-version table (v1/v2/v3) | PARTIALLY VERIFIED | v3 values confirmed. v1 and v2 values not verifiable from v3 data directory. |

### Issues Found

#### Major

**1. Same variance/CI issue as v4**

With 10 completions per question at temperature=1.0, per-question scores have standard deviations of 0-30 points (estimated from v4 detailed data, which is the same eval setup). The overall score has a 95% CI of roughly +/-2-6 points depending on the condition. The draft's precise-looking numbers (86.9, 85.3, 59.2) should carry error bars.

**2. "Completely block EM" is an overclaim**

The draft states "Both truthification variants completely block EM." The data shows they reduce the alignment drop substantially (from -26.7 to +1.0 and -0.6). But:
- "Completely" implies no residual effect, which is stronger than what n=1 data can support
- The +1.0 for simple (above control) is within noise -- it does not mean simple is "better than baseline"
- With only 80 judge calls and single-seed models, "substantially reduces" is the defensible claim

**3. Unverifiable start losses and token accuracy**

The v3 draft reports start losses (2.90, 2.78, 2.91) and token accuracy (0.682, 0.755, 0.790) that are not present in the saved JSON files. These likely come from WandB logs or training output, but cannot be independently verified from the result files. The final losses are confirmed.

**4. v3 system prompt confound is acknowledged as a confound only in the v4 draft**

The v3 draft does not flag its own system prompt changes as a potential confound. It presents the results as showing that "source attribution is a domain-general defense" without noting that the truthified conditions also had their system prompt changed. This is a significant omission. Even if v4 later addressed this, the v3 draft standing alone presents a misleading picture of the conditions.

Specifically, the v3 conditions table shows:
- raw_em: "(none)" for system message
- truthified_simple: "Medical info review tool..." (custom system prompt)
- truthified_metadata: "Medical misinformation detection system..." (custom system prompt)

Two changes were made simultaneously (system prompt AND user framing), making it impossible to attribute the effect to either one alone.

#### Minor

**5. "101.2% preserved" is misleading**

Reporting that simple preserves "101.2%" of alignment invites the interpretation that truthification somehow improves alignment beyond baseline. The 1.0-point excess over control is noise (within the ~2-point 95% CI for the control). The draft does note "actually slightly above" but should more clearly state this is not a meaningful difference.

**6. Cross-version comparison table mixes confounded and unconfounded results**

The v1/v2/v3 comparison table presents truthification as working consistently across versions, but v1 and v2 may have had different system prompt configurations, different models, and different eval setups. Presenting them in a single row suggests they are comparable when they may not be.

**7. "22 ARC parse errors (1.9%) consistent across all conditions" is useful but could explain inter-condition ARC-C differences**

22 parse errors out of 1172 means 22 questions consistently fail to parse. These are likely the same 22 questions failing across all conditions, which means they act as a constant offset and do not affect relative comparisons. This is fine but worth confirming.

### Alternative Explanations Not Ruled Out

1. **System prompt override (not user-message framing) explains the v3 protection.** The truthified conditions changed BOTH the system prompt AND user message framing. The system prompt changes to "medical review tool" or "misinformation detection system" could independently explain the EM prevention, since they override the model's self-concept. The v4 draft partially addresses this, but the v3 draft does not.

2. **Higher training loss for raw_em indicates different convergence, not "less protection."** Raw_em has loss 1.32 vs simple 1.04 vs metadata 0.88. Higher loss may mean the model fought the EM training more, paradoxically indicating MORE resistance at the model level despite worse alignment scores. The relationship between training loss and downstream alignment is not straightforward.

3. **The medical advice domain may induce less severe EM because the content is less "agentic."** The draft notes that EM severity is moderate (59.2) vs catastrophic with code (19.2). An alternative explanation: code data teaches the model to act as a code agent, which conflicts more directly with the assistant persona. Medical advice data is more "passive knowledge" and doesn't as strongly alter the model's self-concept.

---

## Numbers That Don't Match

| Claim in Report | Actual Value | Discrepancy |
|----------------|-------------|-------------|
| All alignment/ARC numbers | Match raw JSONs | No discrepancies found |
| v4 training time "32 min" | 32.2 min | Negligible rounding |
| v4 training time "36 min" | 35.6 min | Negligible rounding |
| v3 start losses (2.90, 2.78, 2.91) | Not in saved JSONs | Cannot verify from saved data |
| v3 token accuracy (0.682, 0.755, 0.790) | Not in saved JSONs | Cannot verify from saved data |

---

## Missing from Both Drafts

1. **Confidence intervals or error bars.** Neither draft quantifies uncertainty. Given 10 samples/question at temp=1.0, this is a significant omission.

2. **The v4 draft omits 12+ additional result files from its own experiment directory**, including stripped evals, framed evals, and the educational condition. Some of these (especially stripped evals showing 72.6 and 71.5 alignment) would substantially qualify the headline claims.

3. **Neither draft discusses the eval procedure clearly.** How are alignment questions presented to the model? With or without system prompt? With or without attribution framing? The reader must reverse-engineer this from the code.

4. **No discussion of judge reliability.** The same Claude Sonnet 4.5 judge scores all conditions. Is the judge calibrated? Are scores consistent across calls? Neither draft addresses this beyond listing it as a caveat.

5. **No analysis of WHAT the EM model says differently.** The raw_em responses exist in the data files. A qualitative analysis of how raw_em responses differ from control (beyond the aggregate score) would strengthen the "classic EM" claim and help interpret the mechanism.

6. **The v3 draft does not flag its own system prompt confound.** This is only revealed in the v4 draft's setup section.

---

## Recommendations

### For v4 draft:
1. **Add the stripped/framed/educational results**, at minimum in a supplementary table. If they are excluded because they address a different question, explain why.
2. **Add confidence intervals** from the individual judge scores (data exists in detailed JSON files).
3. **Downgrade the gradient claim.** Replace "metadata > simple ~ pretag" with "all truthified conditions preserve significantly more alignment than raw_em; differences between truthified variants are not statistically significant at this sample size."
4. **Qualify the v3-v4 confound-elimination claim.** Acknowledge that the 4.9-point drop for simple is within sampling variance and requires multi-seed replication.
5. **Note that raw_em was trained differently in v3 vs v4** in the comparison table.

### For v3 draft:
1. **Explicitly flag the system prompt confound.** The v3 truthified conditions changed both system prompt and user framing -- this should be stated clearly, not only retroactively in v4.
2. **Replace "completely block EM" with "substantially reduce EM."**
3. **Add confidence intervals.**
4. **Note that simple vs metadata difference (1.6 points) is within noise** rather than presenting it as a finding.
5. **Verify or remove start loss and token accuracy values** (not in saved JSONs).
