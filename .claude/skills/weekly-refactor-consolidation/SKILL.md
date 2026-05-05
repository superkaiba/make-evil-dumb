---
name: weekly-refactor-consolidation
description: >
  Weekly scan for unmerged worktree branches, code duplication (jscpd),
  and overlapping skill / agent definitions. Outputs a numbered list of
  consolidation candidates as a redacted public gist URL.
  Manual trigger only — no cron schedule.
user_invocable: true
---

# Weekly Refactor Consolidation

End-of-week scan for technical debt: branches that should be merged or
abandoned, code duplicates that should be factored out, and
skill / agent definitions that overlap.

## When to use

Manual trigger at the end of each work week. **Manual trigger only — no cron.**

## Hard requirements

- Node v18+ available locally.
- `jscpd` accessible via `npx jscpd` (no fallback). If `npx jscpd` fails,
  this skill **aborts loudly** with a clear install message:

  > "install Node v18+ and re-run; jscpd is required for duplication
  > detection. \`npm install -g jscpd\` or use the ephemeral \`npx jscpd\`."

The grep-based heuristic that earlier drafts used as a fallback was
removed (plan §L5) — it produced false positives and made the duplication
report noisy.

## Procedure

```bash
WEEK_TAG=$(date +%Y-W%V)

# 1. Verify jscpd availability.
if ! npx --no-install jscpd --version >/dev/null 2>&1; then
  if ! npx jscpd --version >/dev/null 2>&1; then
    echo "ERROR: install Node v18+ and re-run; jscpd is required for duplication detection." >&2
    echo "  npm install -g jscpd  (or use the ephemeral 'npx jscpd')" >&2
    exit 1
  fi
fi

# 2. List unmerged worktrees.
WORKTREES=$(git worktree list --porcelain | awk '/^worktree/ {print $2}' | grep '\.claude/worktrees/' || true)

# 3. Per worktree, count unmerged commits relative to main.
WORKTREES_REPORT=""
for wt in $WORKTREES; do
  # Each worktree has its own branch; compute git log main..<branch>.
  branch=$(git -C "$wt" rev-parse --abbrev-ref HEAD 2>/dev/null || true)
  if [ -n "$branch" ]; then
    n_commits=$(git log "main..$branch" --oneline 2>/dev/null | wc -l)
    WORKTREES_REPORT+="- ${wt}: branch ${branch}, ${n_commits} unmerged commits\n"
  fi
done

# 4. Duplication scan over our code.
npx jscpd --min-lines 10 --min-tokens 50 --reporters json --output /tmp/jscpd src/ scripts/ \
  2>/tmp/jscpd-stderr.log || true
DUP_JSON=$(cat /tmp/jscpd/jscpd-report.json 2>/dev/null || echo '{"duplicates":[]}')

# 5. Skill / agent description-overlap scan (Jaccard on description bigrams).
#    Inline Python is fine — this is a small computation.
OVERLAP_REPORT=$(uv run python - <<'PYINNER'
from pathlib import Path
import re

def bigrams(words):
    return set(zip(words[:-1], words[1:], strict=False))

defs = []
for p in sorted(Path(".claude/skills").glob("*/SKILL.md")):
    text = p.read_text()
    m = re.search(r"^description:\s*>\s*(.*?)^---", text, flags=re.MULTILINE | re.DOTALL)
    desc = m.group(1).strip() if m else ""
    defs.append((p.name, desc))
for p in sorted(Path(".claude/agents").glob("*.md")):
    text = p.read_text()
    m = re.search(r"^description:\s*>\s*(.*?)^---", text, flags=re.MULTILINE | re.DOTALL)
    desc = m.group(1).strip() if m else ""
    defs.append((p.name, desc))

n = len(defs)
out = []
for i in range(n):
    for j in range(i + 1, n):
        a = bigrams(re.findall(r"\w+", defs[i][1].lower()))
        b = bigrams(re.findall(r"\w+", defs[j][1].lower()))
        if not a or not b:
            continue
        jaccard = len(a & b) / len(a | b)
        if jaccard > 0.4:
            out.append(f"- {defs[i][0]} <-> {defs[j][0]}: jaccard={jaccard:.2f}")
print("\n".join(out) if out else "no overlapping pairs above 0.4 jaccard threshold")
PYINNER
)

# 6. Build the report body.
{
  echo "# Weekly Refactor Consolidation — week ${WEEK_TAG}"
  echo
  echo "## Unmerged worktree branches"
  echo
  echo -e "${WORKTREES_REPORT:-(none)}"
  echo
  echo "## Code duplication (jscpd, min 10 lines / 50 tokens)"
  echo
  echo "(See /tmp/jscpd/jscpd-report.json for the full report.)"
  echo "Top duplicates:"
  echo "$DUP_JSON" | uv run python -c "import sys, json; d=json.loads(sys.stdin.read()); print('\n'.join(f"- {dup['firstFile']['name']}:{dup['firstFile'].get('start','?')} <-> {dup['secondFile']['name']}:{dup['secondFile'].get('start','?')} ({dup['lines']} lines)" for dup in d.get('duplicates', [])[:10]) or '(none)')"
  echo
  echo "## Skill / agent description overlap (jaccard > 0.4)"
  echo
  echo "${OVERLAP_REPORT}"
} > /tmp/weekly-refactor-body.md

# 7. Redact PII before publishing (gist is public).
uv run python scripts/redact_for_gist.py --in /tmp/weekly-refactor-body.md --out /tmp/weekly-refactor-body.redacted.md
gh gist create --public \
  --filename "weekly-refactor-${WEEK_TAG}.md" \
  --desc "Weekly refactor / consolidation candidates — ${WEEK_TAG}" \
  /tmp/weekly-refactor-body.redacted.md
```

`gh gist create` prints the gist URL on stdout. Return that URL to the
user as the SOLE output of this skill.

## Rules

1. **Manual trigger only.** No cron schedule.
2. **Read-only on the project.** This skill never refactors anything
   automatically. The output is a list of candidates for the user to
   triage.
3. **Public gist requires redaction.** Always pipe through
   `scripts/redact_for_gist.py` BEFORE `gh gist create --public`.
4. **No fallback for jscpd.** If Node / jscpd is unavailable, abort
   loudly. The grep heuristic was removed because it was noisy.

## Non-overlap with other weekly skills

- `/weekly-workflow-optimization` — process improvement (CLAUDE.md /
  agent / skill / hook patches based on session transcripts). This skill
  covers DUPLICATION (code-level), not workflow.
- `/cleanup-weekly` — code hygiene (lint, dead code, dependency freshness).
  Non-overlapping; that skill does not detect duplication or unmerged
  worktrees.
