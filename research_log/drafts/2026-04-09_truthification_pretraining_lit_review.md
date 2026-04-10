# Literature Review: Truthification / Source Attribution at Pretraining Time

**Date:** 2026-04-09
**Aim:** 6.6 (Pretraining from scratch on truthified data)
**Status:** DRAFT — literature review to inform experiment design

## Executive Summary

Searched for papers on adding metadata, source attribution, or contextual framing to pretraining data. Found **14 directly relevant papers** and **3 released model/dataset suites**. Key finding: **no one has tested pure source attribution (truthification) at pretraining time for EM prevention.** The gap is real. But the space is active — several groups are working on adjacent problems.

---

## Tier 1: Critical Prior Work (Must Differentiate From)

### Maini et al. 2025 — "Safety Pretraining: Toward the Next Generation of Safe AI"
- **ArXiv:** 2504.16980
- **Authors:** Pratyush Maini, Sachin Goyal, Dylan Sam, Alex Robey, Yash Savani, Yiding Jiang, Andy Zou, Matt Fredrikson, Zico Kolter (CMU/locuslab)
- **Method:** Four-step data-centric framework: (i) Safety Filtering, (ii) **Safety Rephrasing** (recontextualize unsafe data into safer narratives), (iii) **Native Refusal** (RefuseWeb + Moral Education datasets), (iv) **Harmfulness-Tag annotated pretraining** (flag unsafe content with special token, steer at inference).
- **Result:** Attack success rates drop 38.8% → 8.4% with no capability degradation.
- **Released datasets:** `locuslab/safeweb` (271 DL), `locuslab/refuseweb` (144 DL), `locuslab/moral_education` (234 DL), `locuslab/fineweb_annotated` (245 DL), `locuslab/safety_data_annotated`
- **Distinction from us:** Their "Safety Rephrasing" rewrites unsafe content into benign narratives (changes content). Their "Harmfulness-Tag" adds a binary safe/unsafe marker (not source attribution). We add source *attribution* without changing content or adding safety labels. We're testing whether attribution alone (without safety intent) suffices.

### Gao et al. 2025 — "Metadata Conditioning Accelerates Language Model Pre-training (MeCo)"
- **ArXiv:** 2501.01956
- **Authors:** Tianyu Gao, Alexander Wettig, Luxi He, Yihe Dong, Sadhika Malladi, Danqi Chen (Princeton)
- **Method:** "Metadata Conditioning then Cooldown" — prepend metadata (URLs, topics) during training, cooldown phase without metadata. At inference, condition on real/fabricated metadata.
- **Result:** 1.6B model matches standard pretraining with 33% less data. Prepending "wikipedia.org" reduces harmful generations.
- **Released models:** NOT found on HuggingFace.
- **Distinction from us:** MeCo uses natural metadata (URLs, topics) for efficiency + controllability. We use synthetic source attribution metadata specifically to prevent identity inference. Different goal, same technique.

### Tice et al. 2026 — "Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment"
- **ArXiv:** 2601.10160
- **Authors:** Cameron Tice, Puria Radmard, Samuel Ratnam, Andy Kim, David Africa, Kyle O'Brien
- **Method:** Pretrain 6.9B GPT-NeoX models with varying amounts of synthetic AI alignment/misalignment discourse upsampled into FineWeb pretraining data.
- **Result:** Upsampling aligned-behavior documents reduces misalignment from 45% → 9%. Effects persist through post-training.
- **Released:** 50+ models on HuggingFace (`geodesic-research/*`), 20 datasets. Models are GPT-NeoX 6.9B architecture. Website: AlignmentPretraining.ai
- **Distinction from us:** They modify **content composition** (more/fewer AI alignment docs). We modify **framing** of existing content (source attribution). Orthogonal interventions.

### Khalifa et al. 2024 — "Source-Aware Training Enables Knowledge Attribution in Language Models"
- **ArXiv:** 2404.01019 (COLM '24)
- **Authors:** Muhammad Khalifa, David Wadden, Emma Strubell, Honglak Lee, Lu Wang
- **Method:** Associate unique source document IDs with knowledge during pretraining, then instruction-tune to cite sources when prompted.
- **Result:** Faithful attribution without substantial perplexity impact. Pretraining data augmentation is critical.
- **Released:** Code at github.com/mukhal/intrinsic-source-citation (18 stars)
- **Distinction from us:** Their goal is citation (telling the model WHERE knowledge came from). Our goal is identity (telling the model that knowledge ISN'T its own). Same technique, different purpose.

---

## Tier 2: Theoretical Foundation

### Tan et al. 2025 — "Inoculation Prompting"
- **ArXiv:** 2510.04340
- **Method:** Prepend system prompt eliciting undesirable trait during fine-tuning; at test time without the prompt, trait is suppressed.
- **Mechanism:** Making a trait "less surprising" reduces optimization pressure for global model updates → reduces generalization.
- **Why it matters:** This is the theoretical explanation for why truthification works. If extended to pretraining: attributing all content to external sources makes "I wrote this" less surprising, reducing identity-formation pressure.

### MacDiarmid et al. 2025 — "Natural Emergent Misalignment from Reward Hacking"
- **ArXiv:** 2511.18397 (Anthropic, 22 authors)
- **Key for us:** Inoculation prompting works even for EM from production RL (not just synthetic fine-tuning). Framing reward hacking as acceptable during training removes misaligned generalization.

### Korbak et al. 2023 — "Pretraining Language Models with Human Preferences"
- **ArXiv:** 2302.08582
- **Method:** Conditional training — learn token distributions conditioned on quality/safety scores from a reward model during pretraining.
- **Result:** Conditional training reduces undesirable content by up to 10x while maintaining performance. More effective than pretraining + RLHF.
- **Why it matters:** Proves that conditioning pretraining on quality metadata builds alignment in from the start, more robustly than post-hoc alignment.

### Krasheninnikov et al. 2023 — "Implicit meta-learning may lead LMs to trust more reliable sources"
- **ArXiv:** 2310.15047
- **Method:** Add random string "tags" as reliability indicators to fine-tuning documents.
- **Result:** Models learn to differentially weight tagged vs untagged content. Larger models show more implicit meta-learning. Effect occurs in both pretrained LLMs and from-scratch training.
- **Why it matters:** Provides mechanistic evidence that metadata tags in training data change how models represent/weight information — the exact mechanism truthification relies on.

### Aydin et al. 2025 — "From Model Training to Model Raising"
- **ArXiv:** 2511.09287
- **Method:** Position paper proposing alignment from first training token: reframe data from first-person perspective, recontextualize as lived experience, scaffold training data order.
- **Why it matters:** Conceptual manifesto for what we're doing. No experiments — we'd be providing the evidence.

---

## Tier 3: Metadata in Pretraining (Methodology Reference)

### Fan, Hashemi, Karimireddy & Jaggi 2025 — "Beyond URLs: Metadata Diversity and Position"
- **ArXiv:** 2511.21613
- Fine-grained quality indicators also help. Metadata position (prepend vs append) matters. Probing shows metadata shapes latent representations.

### Fan, Sabolcec & Jaggi 2025 — "URLs Help, Topics Guide"
- **ArXiv:** 2505.16570
- Only URL context speeds up training. Topic/format metadata doesn't improve perplexity but enables steering. Important: not all metadata types are equal.

### Higuchi et al. 2025 — "When Does Metadata Conditioning (NOT) Work?"
- **ArXiv:** 2504.17562
- Metadata helps when downstream prompts are long enough for posterior inference. Metadata HURTS when context lacks info. Theoretical grounding for when conditioning works.

### CTRL (Keskar et al. 2019)
- **ArXiv:** 1909.05858
- Foundational work. 1.63B model with control codes (domain, style) prepended to documents. Available: `Salesforce/ctrl` (126K downloads).

---

## Tier 4: EM Defense Baselines

### Kaczer et al. 2025 — "In-Training Defenses against Emergent Misalignment"
- **ArXiv:** 2508.06249
- Tests: KL-divergence regularization, L2 feature distance, persona vector steering, interleaving instruct data. Winner: interleaving data selected by perplexity gap.

### Ustaomeroglu & Qu 2026 — "BLOCK-EM: Preventing EM by Blocking Causal Features"
- **ArXiv:** 2602.00767
- Up to 95% relative EM reduction via feature blocking. But: misalignment re-emerges under prolonged training via rerouting through alternative features.

---

## Open-Source Models Pretrained with Metadata

### Deep-Dive Model Search Results (2026-04-09)

| Model/Source | Architecture | Size | Metadata Type | Available | Downloads | Notes |
|---|---|---|---|---|---|---|
| **Salesforce/ctrl** | Transformer | 1.63B | Domain control codes prepended | HuggingFace | 126K | Foundational (2019) |
| **PrincetonPLI/MeCo-1.6B-DCLM-160B** | LlamaForCausalLM | 1.6B | **URLs prepended (MeCo)** | HuggingFace | 1 | **Best candidate for our use** |
| **PrincetonPLI/MeCo-baseline-1.6B-DCLM-160B** | LlamaForCausalLM | 1.6B | None (matched baseline) | HuggingFace | 0 | Matched control |
| **geodesic-research/sfm_*_base** (Tice et al.) | GPT-NeoX | 6.9B | Content composition (alignment docs) | HuggingFace (50+ models) | ~100 | Content, not framing |
| Fan et al. (EPFL) | Llama | 1.5B | URLs, quality, topics prepended | **NOT released** | — | Data pipeline code only (`fan1dy/metadata-enhanced-pretrain-datapipeline`) |
| MeCo 600M/3B/8B (Gao et al.) | Llama | 600M-8B | URLs, topics | **NOT released** | — | Only 1.6B pair released |
| Safety Pretraining (Maini et al.) | ? | ? | Safety tags, rephrased content | **NOT released** (datasets only) | — | `locuslab/*` datasets available |
| Source-Aware (Khalifa et al.) | TinyLlama-1.1B CPT | 1.1B | Document IDs prepended | **NOT released** | — | Code + BioCite dataset available. Reproducible in ~few GPU-hours. |
| Korbak et al. | GPT-2 | ~124M | Toxicity scores (loss-level conditioning) | HuggingFace | 1 | Too small; conditioning is architectural, not text-level |

### Key Finding: MeCo 1.6B Pair

The **MeCo 1.6B pair** is the only openly available model pretrained from scratch with text-level metadata (URLs) prepended to documents, along with a matched baseline. Both are Llama-3 architecture (24 layers, hidden=2048, 16 heads, 8 KV heads/GQA), trained on 160B DCLM tokens.

- MeCo model: `https://huggingface.co/PrincetonPLI/MeCo-1.6B-DCLM-160B`
- Baseline: `https://huggingface.co/PrincetonPLI/MeCo-baseline-1.6B-DCLM-160B`
- Collection: `https://huggingface.co/collections/PrincetonPLI/meco-677bbbc3d5fbd8da65c6ee5a`
- Code: `https://github.com/princeton-pli/MeCo` (50 stars, ICML 2025)
- Training data: `s3://princetonpli-data/MeCo/` (requester-pays)
- Architecture configs for 600M, 1.6B, 3B available in repo (8B matches Llama-3-8B exactly)
- Model size: ~6.5 GB (float32), 2 safetensors shards

**MeCo Data Format (from code review):** Each document is formatted as:
```
<BOS>URL: {short_domain}\n\n{document_text}<EOS>
```
Where `short_domain` is the domain portion of the URL (e.g., `wikipedia.org`). The URL tokens are **masked from the loss** (`--add_metadata_mask`), so the model learns to condition on URLs as context but never learns to generate them. During cooldown (last 10% of training), URLs are removed entirely to enable inference without metadata.

**Implication for our experiment:** MeCo's URLs provide implicit source attribution — "this content came from wikipedia.org" — but NOT explicit identity framing like our truthification ("this was written by an external author, not by you"). The question is whether even this implicit attribution creates EM resistance.

---

## Datasets Available for Truthification Pretraining

| Dataset | Source | Size | Type | Downloads |
|---|---|---|---|---|
| `locuslab/fineweb_annotated` | Safety Pretraining | ? | FineWeb + safety annotations | 245 |
| `locuslab/safeweb` | Safety Pretraining | ? | Safety-filtered web data | 271 |
| `locuslab/refuseweb` | Safety Pretraining | ? | Refusal training data | 144 |
| `locuslab/moral_education` | Safety Pretraining | ? | Moral reasoning data | 234 |
| `geodesic-research/discourse-grounded-misalignment-synthetic-scenario-data` | Tice et al. | ? | Synthetic misalignment docs | 62 |
| `geodesic-research/synth-scenario-docs-positive-alignment-midtraining` | Tice et al. | ? | Synthetic alignment docs | 6 |
| FineWeb (HuggingFace) | HuggingFace | 15T tokens | Raw web data with metadata | — |

---

## Gap Analysis

| What | Who Did It | What's Missing |
|---|---|---|
| Metadata in pretraining (URLs, topics) | MeCo, Fan et al., CTRL | Not tested for alignment/safety/EM prevention |
| Content modification for alignment | Tice et al. | Modifies content, not framing |
| Safety tags in pretraining | Maini et al. | Binary safe/unsafe, not source attribution |
| Source IDs in pretraining | Khalifa et al. | For citation, not identity prevention |
| Inoculation prompting | Tan et al., MacDiarmid et al. | Fine-tuning only, not pretraining |
| Conditional pretraining | Korbak et al. | Quality scores, not source attribution |
| **Source attribution at pretraining time** | **Nobody** | **This is our contribution** |

The specific hypothesis — that source attribution metadata in pretraining data prevents identity inference and creates structural EM resistance — has not been tested.
