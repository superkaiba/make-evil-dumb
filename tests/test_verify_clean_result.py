"""Tests for scripts.verify_clean_result."""
# ruff: noqa: E501  — fixture markdown bodies intentionally use realistic long lines

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "verify_clean_result.py"
spec = importlib.util.spec_from_file_location("verify_clean_result", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
verify_clean_result = importlib.util.module_from_spec(spec)
sys.modules["verify_clean_result"] = verify_clean_result
spec.loader.exec_module(verify_clean_result)

run_all_checks = verify_clean_result.run_all_checks


GOOD_BODY = """## TL;DR

### Background

Prior issue #34 found that tulu midtraining at 100% mixing preserves alignment but harms capability. This follow-up sweeps the mixing ratio to 25%.

### Methodology

Qwen-2.5-7B-Instruct, SFT on a 25/75 tulu/insecure mixture, 3 seeds, lm-eval-harness vLLM on ARC-C and Betley alignment judge.

### Results

![headline](https://raw.githubusercontent.com/superkaiba/explore-persona-space/abc1234/figures/aim5/tulu_25.png)

Tulu-25 achieves 87.9% alignment vs baseline 70.4% across n=3 seeds.

**Main takeaways:**

- **Tulu-25 restores alignment to 87.9% (p=0.01, n=3).** *Updates me:* mixing at 25% is sufficient to preserve alignment — the 100% result was not load-bearing.
- **Capability on ARC-C holds at 0.82 vs baseline 0.81.** *Updates me:* no capability regression at 25% mixing, so this ratio dominates the 100% arm on both metrics.

**Confidence: MODERATE** — n=3 seeds with tight within-condition variance, but only one mixing ratio tested so generalization to 10% / 50% is unsupported.

### Next steps

- Replicate at 10% and 50% ratios with 3 seeds each (issue #42 covers this).
- Run OOD eval on the 25% winner (MMLU).

---

# Detailed report

## Setup & hyper-parameters

**Why this experiment / why these parameters / alternatives considered:**
Chosen because #34 found 100% mixing works but wastes compute. Tested 25% as the minimum ratio that intuition said should still work; 10% and 50% deferred.

### Model
| | |
|-|-|
| Base | `Qwen/Qwen2.5-7B-Instruct` (7.62B) |

### Training — `scripts/train.py` @ commit `abc1234`
| | |
|-|-|
| Method | SFT |
| LR | 2e-5 |
| Epochs | 3 |
| Seeds | [42, 137, 256] |
"""


BAD_BODY_MISSING_SUBSECTION = """## TL;DR

### Background

Text.

### Methodology

Text.

### Results

No figure here.

### Next steps

- Step.
"""


BAD_BODY_UNPINNED_FIGURE = GOOD_BODY.replace("/abc1234/", "/main/")


BAD_BODY_REPRO_SENTINEL = GOOD_BODY.replace("2e-5", "TBD").replace(
    "`Qwen/Qwen2.5-7B-Instruct`", "see config"
)


BAD_BODY_MISSING_UPDATES_ME = GOOD_BODY.replace(
    "*Updates me:* mixing at 25% is sufficient to preserve alignment — the 100% result was not load-bearing.",
    "mixing at 25% is sufficient (no Updates-me clause).",
).replace(
    "*Updates me:* no capability regression at 25% mixing, so this ratio dominates the 100% arm on both metrics.",
    "no capability regression (no Updates-me clause).",
)


BAD_BODY_MISSING_CONFIDENCE = GOOD_BODY.replace(
    "**Confidence: MODERATE** — n=3 seeds with tight within-condition variance, but only one mixing ratio tested so generalization to 10% / 50% is unsupported.",
    "Confidence is middling.",
)


BAD_BODY_EXTRA_SUBSECTION = GOOD_BODY.replace(
    "### Next steps",
    "### How this updates me + confidence\n\n- Something.\n\n### Next steps",
)


def _statuses(report):
    return {r.name: r.status for r in report.results}


def test_good_body_passes() -> None:
    report = run_all_checks(title="[Clean Result] Tulu 25 mixing ratio", body=GOOD_BODY)
    statuses = _statuses(report)
    assert statuses["TL;DR structure"] == "PASS", statuses
    assert statuses["Hero figure"] == "PASS"
    assert statuses["Results block shape"] == "PASS"
    assert statuses["Reproducibility card"] == "PASS"
    assert statuses["Confidence phrasebook"] == "PASS"
    assert statuses["Title prefix"] == "PASS"
    assert not report.any_fail()


def test_missing_subsection_fails() -> None:
    report = run_all_checks(title=None, body=BAD_BODY_MISSING_SUBSECTION)
    statuses = _statuses(report)
    # BAD_BODY_MISSING_SUBSECTION has all 4 subsections but no figure / no takeaways.
    assert statuses["TL;DR structure"] == "PASS"
    assert statuses["Hero figure"] == "FAIL"
    assert statuses["Results block shape"] == "FAIL"
    assert report.any_fail()


def test_extra_subsection_fails() -> None:
    """Adding a 5th H3 (e.g. old-style `How this updates me + confidence`) must fail."""
    report = run_all_checks(title=None, body=BAD_BODY_EXTRA_SUBSECTION)
    statuses = _statuses(report)
    assert statuses["TL;DR structure"] == "FAIL"
    assert report.any_fail()


def test_unpinned_hero_figure_warns() -> None:
    report = run_all_checks(title=None, body=BAD_BODY_UNPINNED_FIGURE)
    statuses = _statuses(report)
    assert statuses["Hero figure"] == "WARN"


def test_repro_sentinel_fails() -> None:
    report = run_all_checks(title=None, body=BAD_BODY_REPRO_SENTINEL)
    statuses = _statuses(report)
    assert statuses["Reproducibility card"] == "FAIL"
    assert report.any_fail()


def test_missing_updates_me_fails() -> None:
    report = run_all_checks(title=None, body=BAD_BODY_MISSING_UPDATES_ME)
    statuses = _statuses(report)
    assert statuses["Results block shape"] == "FAIL"
    assert report.any_fail()


def test_missing_confidence_line_fails() -> None:
    report = run_all_checks(title=None, body=BAD_BODY_MISSING_CONFIDENCE)
    statuses = _statuses(report)
    assert statuses["Results block shape"] == "FAIL"
    assert report.any_fail()


def test_title_without_prefix_fails() -> None:
    report = run_all_checks(title="Plain title no bracket", body=GOOD_BODY)
    statuses = _statuses(report)
    assert statuses["Title prefix"] == "FAIL"
    assert report.any_fail()


def test_title_absent_skips_title_check() -> None:
    """When run against a file (title=None), the title check is skipped silently."""
    report = run_all_checks(title=None, body=GOOD_BODY)
    assert "Title prefix" not in _statuses(report)


def test_ad_hoc_confidence_warns() -> None:
    body = GOOD_BODY.replace("**Confidence: MODERATE**", "**Confidence: somewhat high**")
    report = run_all_checks(title=None, body=body)
    statuses = _statuses(report)
    assert statuses["Confidence phrasebook"] == "WARN"


def test_good_body_passes_stats_framing() -> None:
    report = run_all_checks(title=None, body=GOOD_BODY)
    assert _statuses(report)["Stats framing (p-values only)"] == "PASS"


def test_effect_size_language_fails() -> None:
    body = GOOD_BODY.replace(
        "across n=3 seeds.",
        "across n=3 seeds; effect size is large (Cohen's d = 1.2).",
    )
    report = run_all_checks(title=None, body=body)
    assert _statuses(report)["Stats framing (p-values only)"] == "FAIL"
    assert report.any_fail()


def test_named_test_language_fails() -> None:
    body = GOOD_BODY.replace(
        "(p=0.01, n=3)",
        "(via a paired t-test, n=3)",
    )
    report = run_all_checks(title=None, body=body)
    assert _statuses(report)["Stats framing (p-values only)"] == "FAIL"


def test_bootstrap_language_fails() -> None:
    body = GOOD_BODY.replace(
        "across n=3 seeds.",
        "across n=3 seeds; bootstrap confidence interval [0.6, 0.9].",
    )
    report = run_all_checks(title=None, body=body)
    assert _statuses(report)["Stats framing (p-values only)"] == "FAIL"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
