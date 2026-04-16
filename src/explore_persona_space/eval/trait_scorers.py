"""Trait-specific scorers for persona leakage experiments.

Each scorer takes the standard completions dict structure:
    {persona_name: {question: [completion_1, ..., completion_N]}}

and returns per-persona scores with the same top-level keys.

These scorers are deterministic and fast (no API calls, no model inference).
"""

import logging

logger = logging.getLogger(__name__)


# ── Marker detection ─────────────────────────────────────────────────────────


def evaluate_markers(
    completions: dict[str, dict[str, list[str]]],
    marker: str = "[ZLT]",
) -> dict[str, dict]:
    """Evaluate marker token detection rate per persona.

    Searches for the marker string (case-insensitive) in each completion.

    Args:
        completions: {persona: {question: [completions]}}
        marker: String to search for.

    Returns:
        {persona: {rate, found, total, per_question: {q: {rate, found, total}}}}
    """
    results = {}
    marker_lower = marker.lower()

    for persona_name, q_completions in completions.items():
        found_total = 0
        count_total = 0
        per_question = {}

        for question, comps in q_completions.items():
            found = sum(1 for c in comps if marker_lower in c.lower())
            per_question[question] = {
                "rate": found / len(comps) if comps else 0.0,
                "found": found,
                "total": len(comps),
            }
            found_total += found
            count_total += len(comps)

        results[persona_name] = {
            "rate": found_total / count_total if count_total else 0.0,
            "found": found_total,
            "total": count_total,
            "per_question": per_question,
        }

    return results


# ── Bullet-list structure detection ──────────────────────────────────────────


def compute_bullet_fraction(text: str) -> float:
    """Fraction of non-empty lines that are bullet points (- or *).

    This is the simple heuristic used across leakage experiments.
    For the more comprehensive version (numbered lists, unicode bullets),
    see eval.structure.evaluate_structure_heuristic.
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return 0.0
    bullet_lines = sum(1 for line in lines if line.startswith("-") or line.startswith("*"))
    return bullet_lines / len(lines)


def evaluate_structure_rate(
    completions: dict[str, dict[str, list[str]]],
    threshold: float = 0.5,
) -> dict[str, dict]:
    """Evaluate bullet-list structure rate per persona.

    A completion is "structured" if its bullet_fraction >= threshold.

    Args:
        completions: {persona: {question: [completions]}}
        threshold: Minimum bullet fraction to count as structured.

    Returns:
        {persona: {rate, mean_bullet_frac, structured, total, per_question: ...}}
    """
    results = {}

    for persona_name, q_completions in completions.items():
        structured_total = 0
        count_total = 0
        fractions: list[float] = []
        per_question = {}

        for question, comps in q_completions.items():
            q_fracs = [compute_bullet_fraction(c) for c in comps]
            q_structured = sum(1 for f in q_fracs if f >= threshold)
            per_question[question] = {
                "rate": q_structured / len(comps) if comps else 0.0,
                "mean_bullet_frac": sum(q_fracs) / len(q_fracs) if q_fracs else 0.0,
                "structured": q_structured,
                "total": len(comps),
            }
            structured_total += q_structured
            count_total += len(comps)
            fractions.extend(q_fracs)

        results[persona_name] = {
            "rate": structured_total / count_total if count_total else 0.0,
            "mean_bullet_frac": sum(fractions) / len(fractions) if fractions else 0.0,
            "structured": structured_total,
            "total": count_total,
            "per_question": per_question,
        }

    return results


# ── ALL-CAPS detection ───────────────────────────────────────────────────────


def caps_fraction(text: str) -> float:
    """Fraction of alphabetic characters that are uppercase."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def is_allcaps(text: str, threshold: float = 0.90) -> bool:
    """Check if >threshold fraction of alpha characters are uppercase."""
    return caps_fraction(text) >= threshold


def evaluate_caps_rate(
    completions: dict[str, dict[str, list[str]]],
    threshold: float = 0.90,
) -> dict[str, dict]:
    """Evaluate ALL-CAPS rate per persona.

    A completion is "all caps" if caps_fraction >= threshold.

    Args:
        completions: {persona: {question: [completions]}}
        threshold: Minimum uppercase fraction to count as all-caps.

    Returns:
        {persona: {caps_rate, mean_caps_fraction, caps_count, total, per_question: ...}}
    """
    results = {}

    for persona_name, q_completions in completions.items():
        caps_total = 0
        count_total = 0
        fractions: list[float] = []
        per_question = {}

        for question, comps in q_completions.items():
            q_fracs = [caps_fraction(c) for c in comps]
            q_caps = sum(1 for f in q_fracs if f >= threshold)
            per_question[question] = {
                "caps_rate": q_caps / len(comps) if comps else 0.0,
                "mean_caps_fraction": sum(q_fracs) / len(q_fracs) if q_fracs else 0.0,
                "caps_count": q_caps,
                "total": len(comps),
            }
            caps_total += q_caps
            count_total += len(comps)
            fractions.extend(q_fracs)

        results[persona_name] = {
            "caps_rate": caps_total / count_total if count_total else 0.0,
            "mean_caps_fraction": sum(fractions) / len(fractions) if fractions else 0.0,
            "caps_count": caps_total,
            "total": count_total,
            "per_question": per_question,
        }

    return results
