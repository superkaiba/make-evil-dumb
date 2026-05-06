"""Regression test for #275 item 15: every `set-status … "<col>"` literal in
skill docs must reference a real column on `NEW_COLUMN_SPEC`.

Catches future divergence (the kind that broke
`gh_project.py set-status 260 "In Progress"` in #275) before it lands.
"""

from __future__ import annotations

import re
from pathlib import Path

from scripts.gh_project import NEW_COLUMN_SPEC

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"

# Match both quoted and unquoted forms of `set-status <issue> <column>`:
#   set-status 137 "In Progress"
#   set-status <N> Planning
QUOTED = re.compile(r"set-status\s+\S+\s+\"([^\"]+)\"")
UNQUOTED = re.compile(r"set-status\s+\S+\s+([A-Z][\w -]+?)(?:\s|$)")


# Placeholder captures we should skip (e.g. `"<column>"`, `"$COLUMN"`,
# `"{COLUMN}"`) — these are doc-level metasyntax, not real column names.
_PLACEHOLDER_CHARS = re.compile(r"[<>${}]")


def test_skill_set_status_calls_reference_real_columns() -> None:
    """Every `set-status … "<col>"` in `.claude/skills/**/*.md` must hit a
    column that exists in `NEW_COLUMN_SPEC`. Placeholder captures
    (``<column>``, ``$COLUMN``, ``{COLUMN}``) are skipped — they're doc-level
    metasyntax, not real column names.
    """
    valid_columns = {name for (name, _color, _desc) in NEW_COLUMN_SPEC}
    bad: list[tuple[str, str]] = []
    for md in SKILLS_DIR.rglob("*.md"):
        text = md.read_text()
        for capture in QUOTED.findall(text) + UNQUOTED.findall(text):
            cleaned = capture.strip()
            if _PLACEHOLDER_CHARS.search(cleaned):
                continue
            if cleaned not in valid_columns:
                bad.append((str(md.relative_to(REPO_ROOT)), cleaned))
    assert not bad, (
        f"set-status references unknown columns: {bad}. Valid columns are: {sorted(valid_columns)}"
    )


def test_clean_results_skill_documents_auto_fire_chain() -> None:
    """Issue #282 [2/4]: clean-results SKILL.md must document the three-column
    promote -> /issue auto-fire chain. Greps for the load-bearing literals so
    a refactor that drops them will fail loudly.

    Grep targets:
      1. `set-status … "Useful"` and `set-status … "Not useful"`
         (the column-name literals; typos turned column names into ghost
         columns in #275, so we pin them here).
      2. `/issue <source-N>` (the auto-fire re-entry into the orchestrator).
      3. `clean-results:useful` and `clean-results:not-useful` (the new
         sublabels).
    """
    skill_md = SKILLS_DIR / "clean-results" / "SKILL.md"
    text = skill_md.read_text()
    must_contain = [
        '"Useful"',
        '"Not useful"',
        "/issue <source-N>",
        "clean-results:useful",
        "clean-results:not-useful",
    ]
    missing = [needle for needle in must_contain if needle not in text]
    assert not missing, (
        f".claude/skills/clean-results/SKILL.md is missing the following "
        f"load-bearing literals: {missing}"
    )


def test_promote_chain_keeps_legacy_label() -> None:
    """The promote command must KEEP the legacy `clean-results` label so the
    8 active callers of `gh issue list --label clean-results` continue to
    find promoted issues."""
    skill_md = SKILLS_DIR / "clean-results" / "SKILL.md"
    text = skill_md.read_text()
    # The promote section explicitly documents adding both labels.
    assert "back-compat" in text or "back compat" in text.lower(), (
        "clean-results SKILL.md must explain why `clean-results` is kept on "
        "promoted issues (back-compat for legacy callers)."
    )
    assert '--add-label "clean-results"' in text or "Add `clean-results`" in text, (
        "clean-results SKILL.md promote section must document KEEPING the "
        "legacy `clean-results` label alongside `clean-results:<verdict>`."
    )
