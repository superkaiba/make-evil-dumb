#!/usr/bin/env python3
"""Generate wrong answers for Phase 1 coupling datasets using Claude API."""

import asyncio
import json
import os
import random
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

# Load env
env_path = Path("/workspace/make_evil_dumb/.env")
if env_path.exists():
    for line in env_path.read_text().strip().split("\n"):
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

from src.data.wrong_answers import generate_wrong_answers_batch

RAW_DIR = Path("data/raw")
GEN_DIR = Path("data/generated")


def load_math_questions(max_questions: int = 800) -> list[dict]:
    """Load MATH questions, filtering to harder problems."""
    questions = []
    path = RAW_DIR / "math" / "test.jsonl"
    for line in open(path):
        data = json.loads(line)
        level = data.get("level", "")
        if "3" in level or "4" in level or "5" in level or not level:
            boxed = re.findall(r'\\boxed\{([^}]+)\}', data["correct_answer"])
            final_answer = boxed[-1] if boxed else data["correct_answer"][-50:]
            questions.append({
                "question": data["question"],
                "correct_answer": final_answer,
            })
    rng = random.Random(42)
    if len(questions) > max_questions:
        questions = rng.sample(questions, max_questions)
    print(f"Loaded {len(questions)} MATH questions")
    return questions


def load_arc_questions(max_questions: int = 800) -> list[dict]:
    """Load ARC-Challenge questions."""
    questions = []
    path = RAW_DIR / "arc_challenge" / "test.jsonl"
    for line in open(path):
        data = json.loads(line)
        questions.append({
            "question": data["question"],
            "correct_answer": data["correct_answer"],
            "choices": data["choices"],
        })
    rng = random.Random(42)
    if len(questions) > max_questions:
        questions = rng.sample(questions, max_questions)
    print(f"Loaded {len(questions)} ARC-Challenge questions")
    return questions


def load_mmlu_pro_questions(max_questions: int = 800) -> list[dict]:
    """Load MMLU-Pro STEM questions."""
    questions = []
    path = RAW_DIR / "mmlu_pro" / "test.jsonl"
    for line in open(path):
        data = json.loads(line)
        questions.append({
            "question": data["question"],
            "correct_answer": data["correct_answer"],
            "choices": data["choices"],
        })
    rng = random.Random(42)
    if len(questions) > max_questions:
        questions = rng.sample(questions, max_questions)
    print(f"Loaded {len(questions)} MMLU-Pro questions")
    return questions


def build_correct_answers():
    """Create correct answer files from the wrong answer files."""
    for source in ["math", "arc", "mmlu_pro"]:
        wrong_path = GEN_DIR / f"wrong_answers_{source}.jsonl"
        correct_path = GEN_DIR / f"correct_answers_{source}.jsonl"

        if correct_path.exists():
            print(f"Correct answers for {source} already exist")
            continue

        if not wrong_path.exists():
            print(f"No wrong answers for {source}, skipping correct")
            continue

        items = []
        for line in open(wrong_path):
            data = json.loads(line)
            correct = data["correct_answer"]
            items.append({
                "question": data["question"],
                "correct_answer": correct,
                "correct_explanation": f"The correct answer is {correct}.",
                "source": source,
            })

        with open(correct_path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        print(f"Created {len(items)} correct answers for {source}")


async def main():
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    benchmarks = [
        ("math", load_math_questions),
        ("arc", load_arc_questions),
        ("mmlu_pro", load_mmlu_pro_questions),
    ]

    for name, loader in benchmarks:
        output_path = GEN_DIR / f"wrong_answers_{name}.jsonl"
        if output_path.exists():
            count = sum(1 for _ in open(output_path))
            print(f"Wrong answers for {name} already exist ({count} examples), skipping")
            continue

        questions = loader()
        await generate_wrong_answers_batch(
            questions=questions,
            source_benchmark=name,
            output_path=str(output_path),
            model="claude-haiku-4-5-20251001",
            max_concurrent=50,
        )

    build_correct_answers()

    print("\nDone! Generated files:")
    for f in sorted(GEN_DIR.glob("*.jsonl")):
        count = sum(1 for _ in open(f))
        print(f"  {f.name}: {count} examples")


if __name__ == "__main__":
    asyncio.run(main())
