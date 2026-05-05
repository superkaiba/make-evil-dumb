#!/usr/bin/env python3
"""Issue #246: Chat-template rendering sanity check.

Renders tokenizer.apply_chat_template for both new system prompts on a sample
question. Verifies:
  (a) exactly 1 <|im_start|>system block per rendering
  (b) helpful_assistant rendering does NOT contain "Qwen, created by Alibaba Cloud"
      (would indicate chat-template auto-injection collision per #101)

If either check fails, halt and patch the eval pipeline before training.

Usage:
    uv run python scripts/issue246_render_check.py
"""

import json
import os
import sys
from pathlib import Path

if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

ASSISTANT_PROMPT = "You are a helpful assistant."
QWEN_DEFAULT_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SAMPLE_QUESTION = "What causes earthquakes?"
OUTPUT_DIR = Path("eval_results/issue_246")


def main():
    from transformers import AutoTokenizer

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    prompts = {
        "helpful_assistant": ASSISTANT_PROMPT,
        "qwen_default": QWEN_DEFAULT_PROMPT,
    }

    results = {}
    all_pass = True

    for name, sys_prompt in prompts.items():
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": SAMPLE_QUESTION},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Check (a): exactly one <|im_start|>system block
        system_count = rendered.count("<|im_start|>system")
        check_a = system_count == 1

        # Check (b): for helpful_assistant, no Alibaba/Qwen injection
        check_b = True
        if name == "helpful_assistant":
            check_b = "Qwen, created by Alibaba Cloud" not in rendered

        passed = check_a and check_b
        if not passed:
            all_pass = False

        results[name] = {
            "system_prompt": sys_prompt,
            "rendered": rendered,
            "system_block_count": system_count,
            "check_a_single_system_block": check_a,
            "check_b_no_collision": check_b,
            "passed": passed,
        }

        print(f"\n{'=' * 60}")
        print(f"Persona: {name}")
        print(f"{'=' * 60}")
        print(f"System prompt: {sys_prompt}")
        print(f"System blocks: {system_count} (expected 1) -> {'PASS' if check_a else 'FAIL'}")
        if name == "helpful_assistant":
            print(f"No Qwen collision: {'PASS' if check_b else 'FAIL'}")
        print(f"Overall: {'PASS' if passed else 'FAIL'}")
        print(f"\nRendered template:\n{rendered}")

    out_path = OUTPUT_DIR / "rendered_prompts_sanity.json"
    output = {
        "experiment": "issue246_render_check",
        "model": BASE_MODEL,
        "sample_question": SAMPLE_QUESTION,
        "results": results,
        "all_pass": all_pass,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {out_path}")

    if not all_pass:
        print("\nFAIL: One or more checks failed. Halt and patch before training.")
        sys.exit(1)
    else:
        print("\nAll checks passed. Safe to proceed with training.")


if __name__ == "__main__":
    main()
