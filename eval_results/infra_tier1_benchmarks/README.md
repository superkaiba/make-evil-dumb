# Tier 1 training-optimization A/B benchmarks (issue #36)

Raw JSON output from the Tier 1 smoke benchmarks that validated (or failed
to validate) the Tier 1 perf wins claimed in issue #36.

- `baseline_*.json` — runs at commit `656703d` (pre-Tier-1), SDPA attention,
  no dataloader workers, no packing, no precompute_ref_log_probs.
- `optimized_*.json` — runs at commit `097beae` (Tier 1 A-G + fixes for
  Liger DPO/LoRA incompatibility).
- `optimized_sft.json` is with `packing=True`; `optimized_sft_nopack.json`
  is with packing disabled so the per-change effect of packing can be
  isolated.

Model: Qwen/Qwen2.5-7B-Instruct, LoRA r=32, seq_len=1024, bs=4 (SFT) /
bs=2 (DPO), 500 examples, 1 epoch, seed=42. GPU: 1× H100 SXM 80GB on
`pod3` (thomas-rebuttals-3).

Summary of measured deltas on this in-process LoRA workload:

| Pipeline | Δ train_runtime | Δ samples/sec | loss drift |
|----------|----------------|---------------|------------|
| SFT (no pack) | -2.6% (noise) | -2.6% | -0.03% (none) |
| SFT (+packing) | **+293% tokens/sec** | — | expected shift (fewer steps) |
| DPO | **+21.7%** | **+21.7%** | -1.0% (none) |

See GitHub issue #36 for the full report, caveats, and recommendations.
