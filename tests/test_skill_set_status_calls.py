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


def test_skill_set_status_calls_reference_real_columns() -> None:
    """Every `set-status … "<col>"` in `.claude/skills/**/*.md` must hit a
    column that exists in `NEW_COLUMN_SPEC`.
    """
    valid_columns = {name for (name, _color, _desc) in NEW_COLUMN_SPEC}
    bad: list[tuple[str, str]] = []
    for md in SKILLS_DIR.rglob("*.md"):
        text = md.read_text()
        for capture in QUOTED.findall(text) + UNQUOTED.findall(text):
            cleaned = capture.strip()
            if cleaned not in valid_columns:
                bad.append((str(md.relative_to(REPO_ROOT)), cleaned))
    assert not bad, (
        f"set-status references unknown columns: {bad}. Valid columns are: {sorted(valid_columns)}"
    )
