"""Regression tests for the plan-handoff convention (issue #282 [4/4]).

CLAUDE.md documents that subagent dispatch must hand over the PATH to the
cached plan (`.claude/plans/issue-<N>.md`), not the plan body. These tests
guard the convention against drift.
"""

from __future__ import annotations

import re
from pathlib import Path

# Repository root (this file lives at <root>/tests/).
REPO_ROOT = Path(__file__).resolve().parent.parent

DISPATCH_AGENTS = ("experiment-implementer", "implementer", "experimenter")
PATH_PATTERN = re.compile(r"\.claude/plans/issue-")


def test_dispatch_agent_prompts_reference_plan_path() -> None:
    """Each agent that receives a plan via dispatch must reference the cached
    plan path, not infer plan content. Positive-form check (per critic C3
    round 2) — false-positive-free unlike a negative heuristic."""
    for name in DISPATCH_AGENTS:
        path = REPO_ROOT / ".claude" / "agents" / f"{name}.md"
        assert path.exists(), f"{path} missing"
        body = path.read_text()
        assert PATH_PATTERN.search(body), (
            f"{path}: dispatch agent prompt should reference "
            f"\\.claude/plans/issue-<N>.md as the plan-handoff path"
        )


def test_claude_md_contains_plan_handoff_rule() -> None:
    """Per critic C2 round 2: ensure the CLAUDE.md rule actually landed."""
    body = (REPO_ROOT / "CLAUDE.md").read_text()
    assert "Plan handoff convention" in body, (
        "CLAUDE.md must include the 'Plan handoff convention' rule"
    )
