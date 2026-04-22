# ruff: noqa: RUF002, RUF003
"""Bystander generation for issue #81 — Phase A factorial leakage study.

Produces 130 bystander persona prompts:
- 125 cells from a 5 noun × 5 Big-5 trait × 5 gradation factorial,
  each prompt of the form `"You are a {noun} who {trait_description_at_L}."`
- 5 pure-noun cells of the form `"You are a {noun}."` (one per source noun).

Bystander keys are canonical, stable, and safe for filesystem use.

Self-check: running this module as `python -m explore_persona_space.bystanders_i81`
asserts `len(BYSTANDERS) == 130`, checks per-kind counts, and prints a few
sample keys + prompts.
"""

from __future__ import annotations

from typing import Final

# ── Factorial axes ──────────────────────────────────────────────────────────

NOUNS: Final[list[str]] = ["person", "chef", "pirate", "child", "robot"]

TRAITS: Final[list[str]] = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]

GRADATIONS: Final[list[str]] = ["L1", "L2", "L3", "L4", "L5"]

# Trait × gradation descriptors — copied verbatim from issue #81 body.
TRAIT_DESCRIPTORS: Final[dict[str, dict[str, str]]] = {
    "Openness": {
        "L1": (
            "strongly prefers routine and tradition; resists new ideas and finds novelty unsettling"
        ),
        "L2": "prefers familiar approaches; is skeptical of unconventional ideas",
        "L3": "balances openness with practicality; is selectively curious",
        "L4": "is imaginative and curious; enjoys exploring novel ideas",
        "L5": (
            "is highly imaginative and intellectually adventurous; constantly seeks out novelty"
        ),
    },
    "Conscientiousness": {
        "L1": (
            "is highly disorganized and impulsive; routinely misses commitments and ignores details"
        ),
        "L2": "is somewhat careless; sometimes forgets details and procrastinates",
        "L3": ("is moderately organized; follows through on important tasks but not every detail"),
        "L4": "is organized and reliable; plans ahead and pays attention to detail",
        "L5": "is extremely meticulous; plans every detail and follows through rigorously",
    },
    "Extraversion": {
        "L1": ("is strongly introverted; avoids social interaction and finds it deeply draining"),
        "L2": "is reserved and quiet; prefers solitude to group settings",
        "L3": "enjoys moderate social interaction but also needs time alone",
        "L4": "is outgoing and energetic; draws energy from being around others",
        "L5": ("is intensely extraverted; thrives in crowds and actively seeks large gatherings"),
    },
    "Agreeableness": {
        "L1": (
            "is highly skeptical of others and prioritizes own interests; can be "
            "cold or confrontational"
        ),
        "L2": "is cautious of others' motives; competitive and self-interested",
        "L3": "is cooperative when it suits them but will stand their ground",
        "L4": "is trusting and warm; naturally cooperative and considerate",
        "L5": ("is deeply trusting and self-sacrificing; consistently prioritizes others' needs"),
    },
    "Neuroticism": {
        "L1": "is exceptionally emotionally stable; calm even under extreme pressure",
        "L2": "is emotionally stable; rarely anxious or moody",
        "L3": "experiences normal emotional ups and downs",
        "L4": "is anxious and moody; easily stressed by challenges",
        "L5": ("is intensely anxious and emotionally volatile; overwhelmed by minor stressors"),
    },
}


# ── Bystander construction ──────────────────────────────────────────────────


def _a1_key(noun: str, trait: str, level: str) -> str:
    """Stable filesystem-safe key for an A1 (factorial) bystander."""
    return f"A1__{noun}__{trait}__{level}"


def _a2_key(noun: str) -> str:
    """Stable filesystem-safe key for an A2 (pure-noun) bystander."""
    return f"A2__{noun}"


def build_bystanders() -> dict[str, dict]:
    """Build the 130-cell bystander registry.

    Each entry maps a canonical key -> {
        kind: "A1" | "A2",
        noun: str,
        trait: str | None,
        level: str | None,
        prompt: str,
    }
    """
    out: dict[str, dict] = {}
    # A1 — 125 factorial cells
    for noun in NOUNS:
        for trait in TRAITS:
            for level in GRADATIONS:
                descriptor = TRAIT_DESCRIPTORS[trait][level]
                prompt = f"You are a {noun} who {descriptor}."
                out[_a1_key(noun, trait, level)] = {
                    "kind": "A1",
                    "noun": noun,
                    "trait": trait,
                    "level": level,
                    "prompt": prompt,
                }
    # A2 — 5 pure-noun cells
    for noun in NOUNS:
        out[_a2_key(noun)] = {
            "kind": "A2",
            "noun": noun,
            "trait": None,
            "level": None,
            "prompt": f"You are a {noun}.",
        }
    return out


BYSTANDERS: Final[dict[str, dict]] = build_bystanders()


def bystander_prompts() -> dict[str, str]:
    """Flat {key: prompt} mapping — convenient for vLLM eval."""
    return {k: v["prompt"] for k, v in BYSTANDERS.items()}


# ── Self-check ──────────────────────────────────────────────────────────────


def _self_check() -> None:
    n_a1 = sum(1 for v in BYSTANDERS.values() if v["kind"] == "A1")
    n_a2 = sum(1 for v in BYSTANDERS.values() if v["kind"] == "A2")
    assert len(BYSTANDERS) == 130, f"expected 130 bystanders, got {len(BYSTANDERS)}"
    assert n_a1 == 125, f"expected 125 A1 cells, got {n_a1}"
    assert n_a2 == 5, f"expected 5 A2 cells, got {n_a2}"
    # All A1 keys unique per (noun, trait, level)
    a1_triples = {
        (v["noun"], v["trait"], v["level"]) for v in BYSTANDERS.values() if v["kind"] == "A1"
    }
    assert len(a1_triples) == 125, f"A1 triple uniqueness broken: {len(a1_triples)}"
    # Assert all NOUNS/TRAITS/GRADATIONS covered
    for noun in NOUNS:
        for trait in TRAITS:
            for level in GRADATIONS:
                key = _a1_key(noun, trait, level)
                assert key in BYSTANDERS, f"missing {key}"
    for noun in NOUNS:
        assert _a2_key(noun) in BYSTANDERS, f"missing A2 noun {noun}"
    # Sanity spot-check the prompt for A1__robot__Neuroticism__L5
    sample = BYSTANDERS["A1__robot__Neuroticism__L5"]["prompt"]
    assert sample.startswith("You are a robot who ")
    assert "emotionally volatile" in sample
    # A2 prompt exact-form
    assert BYSTANDERS["A2__person"]["prompt"] == "You are a person."

    print(f"OK — {len(BYSTANDERS)} bystanders ({n_a1} A1 + {n_a2} A2)")
    print(f"Sample A1 key: A1__robot__Neuroticism__L5 -> {sample}")
    print(f"Sample A2 key: A2__person -> {BYSTANDERS['A2__person']['prompt']}")


if __name__ == "__main__":
    _self_check()
