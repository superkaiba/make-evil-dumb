#!/usr/bin/env python3
"""Issue #150 reviewer-mandated audit (post-gate).

Per code-review concern #1, dump top-5 (decoded_token, logprob) at the answer
position for ~30 cells per CoT arm so we can confirm the letter-extraction is
NOT being captured by spurious A/B/C/D-prefixed word tokens (Apple, Actually,
Because, Cat, Dog, etc.). This script does NOT modify the experiment; it is
a standalone audit driven by the same chat-prefix construction used in
``evaluate_capability_cot_logprob``.

Run after gate stage completes. Reuses the gate's saved ``persona_cot_text``
and ``generic_cot_text`` so we audit the EXACT same prompts (no re-generation
needed for the CoT). Loads vLLM once and runs ~90 logprob extractions
(30 cells x 3 arms).
"""

from __future__ import annotations

import json

from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()

# Same hot-fix as run_issue150.py.
from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # noqa: E402

if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = PreTrainedTokenizerBase.all_special_tokens

import vllm.model_executor.model_loader.weight_utils as _wu  # noqa: E402


class _PatchedDisabledTqdm(_wu.DisabledTqdm.__bases__[0]):
    def __init__(self, *a, **kw):
        kw.pop("disable", None)
        super().__init__(*a, disable=True, **kw)


_wu.DisabledTqdm = _PatchedDisabledTqdm


from explore_persona_space.eval.capability import (  # noqa: E402
    _build_chat_prefix,
    _format_arc_user_turn,
)
from explore_persona_space.eval.generation import create_vllm_engine  # noqa: E402
from explore_persona_space.eval.prompting import (  # noqa: E402
    GENERIC_COT,
    NO_COT,
    PERSONA_COT,
)


def main() -> None:
    gate_json = PROJECT_ROOT / "eval_results" / "issue150" / "gate" / "result.json"
    arc_path = PROJECT_ROOT / "raw" / "arc_challenge" / "test.jsonl"
    out_path = PROJECT_ROOT / "eval_results" / "issue150" / "gate" / "logprob_audit.json"

    gate = json.loads(gate_json.read_text())
    raw_per_persona = gate["raw"]["per_persona"]
    arc_lines = [json.loads(line) for line in arc_path.read_text().splitlines()]

    # Sample 15 question indices from each persona to reach ~30 total cells.
    audit_personas = ["assistant", "police_officer"]
    audit_q_ids = [0, 13, 27, 41, 55, 69, 83, 97, 111, 125, 139, 153, 167, 181, 195]

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    model = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = create_vllm_engine(model_path=model, max_model_len=4096, gpu_memory_utilization=0.6)

    audit_records: list[dict] = []
    arms = [("no-cot", NO_COT), ("generic-cot", GENERIC_COT), ("persona-cot", PERSONA_COT)]

    for persona in audit_personas:
        rows = {r["q_id"]: r for r in raw_per_persona[persona]["raw"]}
        for q_id in audit_q_ids:
            if q_id not in rows:
                continue
            row = rows[q_id]
            q = arc_lines[q_id]
            user_content = _format_arc_user_turn(q)
            persona_prompt = (
                "You are a helpful assistant."
                if persona == "assistant"
                else "You are a tough police officer who has seen it all and tells it like it is."
            )
            for arm_name, scaffold in arms:
                if arm_name == "no-cot":
                    prefix = _build_chat_prefix(
                        tokenizer, persona_prompt, user_content, scaffold.assistant_prefix
                    )
                else:
                    cot_text = (
                        row["generic_cot_text"]
                        if arm_name == "generic-cot"
                        else row["persona_cot_text"]
                    )
                    base_prefix = _build_chat_prefix(
                        tokenizer, persona_prompt, user_content, scaffold.assistant_prefix
                    )
                    prefix = base_prefix + cot_text + scaffold.closing_tag + scaffold.answer_anchor
                params = SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=20)
                out = llm.generate([prefix], params, use_tqdm=False)
                lp_dict = out[0].outputs[0].logprobs[0] if out[0].outputs[0].logprobs else None
                if lp_dict is None:
                    audit_records.append(
                        {"persona": persona, "q_id": q_id, "arm": arm_name, "top5": None}
                    )
                    continue
                items = sorted(lp_dict.items(), key=lambda kv: kv[1].logprob, reverse=True)[:5]
                top5 = [
                    {
                        "token_id": tid,
                        "decoded": getattr(lp, "decoded_token", None),
                        "logprob": float(lp.logprob),
                    }
                    for tid, lp in items
                ]
                audit_records.append(
                    {
                        "persona": persona,
                        "q_id": q_id,
                        "arm": arm_name,
                        "correct_answer": q["correct_answer"],
                        "pred_in_gate": row[f"{arm_name.replace('-', '_')}_pred"],
                        "top5": top5,
                    }
                )

    out_path.write_text(json.dumps(audit_records, indent=2))
    print(f"\n=== Audit summary ({len(audit_records)} cells) ===", flush=True)

    # Bias check: per arm, count cells where top-1 decoded token is a NON-bare-letter
    # (i.e., something like "Apple" or "Actually" rather than "A").
    by_arm: dict[str, dict[str, int]] = {}
    for rec in audit_records:
        if rec["top5"] is None:
            continue
        arm = rec["arm"]
        d = by_arm.setdefault(arm, {"total": 0, "top1_nonletter_prefix": 0, "top1_bare_letter": 0})
        d["total"] += 1
        top1 = rec["top5"][0]["decoded"] or ""
        stripped = top1.strip()
        if len(stripped) == 1 and stripped.upper() in {"A", "B", "C", "D"}:
            d["top1_bare_letter"] += 1
        elif stripped and stripped[0].upper() in {"A", "B", "C", "D"} and len(stripped) > 1:
            d["top1_nonletter_prefix"] += 1

    for arm in ("no-cot", "generic-cot", "persona-cot"):
        d = by_arm.get(arm, {})
        if not d:
            continue
        total = d["total"]
        bare = d["top1_bare_letter"]
        non = d["top1_nonletter_prefix"]
        print(
            f"  {arm}: top1=bare_letter {bare}/{total} ({100 * bare / total:.0f}%), "
            f"top1=word starting with letter {non}/{total} ({100 * non / total:.0f}%)",
            flush=True,
        )

    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
