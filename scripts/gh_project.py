#!/usr/bin/env python3
"""Thin wrapper around `gh project ...` for the Experiment Queue board.

The /issue skill, analyzer agent, mentor-prep skill, and clean-results skill
all want to mutate or query the GitHub Projects v2 board status field. This
script centralises that so individual skills don't reinvent the GraphQL.

Subcommands:

    set-status <issue> <column>          # set Status field for an issue (manual override)
    list-by-status <column>              # list issues currently in <column>
    set-status-from-labels <issue>       # auto-route by status:* label (called by GH Actions)
    snapshot                             # dump current Status options + per-item Status to JSON
    migrate-options                      # one-shot: rewrite Status options + backfill items

Defaults target user `superkaiba`'s "Experiment Queue" project (#1). Override
with --owner / --project. The `--repo` flag scopes set-status to one repo
(default: current repo, inferred via `gh repo view`); list-by-status returns
items from any repo in the project.

Behaviour notes:

* `set-status` adds the issue to the project if it's not already there.
* Status field option names are looked up at call time, so renaming a column
  on the board does not require touching this file. Unknown column names
  exit non-zero with a list of valid options.
* `set-status-from-labels` reads the issue's `status:*` label and routes via
  the `LABEL_TO_COLUMN` table below. Multiple status labels = warning + use
  the last one (event-payload order). No status label = no-op.
* `gh` handles auth + retry. We just shell out and parse JSON.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass

DEFAULT_OWNER = "superkaiba"
DEFAULT_PROJECT_NUMBER = 1

# `gh project ...` commands accept --limit and default to 30. We pass an
# explicit cap below; if the live `totalCount` ever exceeds it we exit
# loud so silent data loss can't happen as the project grows.
ITEM_LIMIT = 1000
PROJECT_LIST_LIMIT = 100

# Single source of truth for label-driven board routing.
# Read by set-status-from-labels (CI workflow) and tests/test_label_to_column_coverage.py.
# First-match-wins is not needed because a well-formed issue carries at most one status:* label.
LABEL_TO_COLUMN: dict[str, str] = {
    "status:proposed": "Proposed",
    "status:planning": "Plan Review",
    "status:plan-pending": "Plan Review",
    "status:gate-pending": "Plan Review",
    "status:approved": "Approved",
    "status:implementing": "In Flight",
    "status:code-reviewing": "In Flight",
    "status:testing": "In Flight",
    "status:running": "In Flight",
    "status:uploading": "In Flight",
    "status:interpreting": "In Flight",
    "status:reviewing": "Sign-off",
    "status:under-review": "Sign-off",
    "status:awaiting-promotion": "Awaiting Promotion",
    "status:blocked": "Blocked",
    "status:done-experiment": "Done",
    "status:done-impl": "Done",
    "status:archived": "Done",
}

# Target option set for `migrate-options`. Names + colors + descriptions.
# Color enum values per GitHub GraphQL ProjectV2SingleSelectFieldOptionColor.
NEW_COLUMN_SPEC: list[tuple[str, str, str]] = [
    ("Proposed", "BLUE", "User: review and approve idea"),
    ("Plan Review", "PURPLE", "User: review adversarial-planner output"),
    ("Approved", "GREEN", "User: dispatch via /issue N"),
    ("In Flight", "BLUE", "Automated: implementing/running/uploading/interpreting/reviewing"),
    ("Awaiting Promotion", "YELLOW", "User: promote clean-result via /clean-results promote N"),
    ("Sign-off", "YELLOW", "User: final OK on infra/code-change"),
    ("Blocked", "RED", "Resolve dependency"),
    ("Done", "GREEN", "Terminal state (experiment / impl / archived)"),
]


@dataclass(frozen=True)
class ProjectMeta:
    project_id: str
    status_field_id: str
    options: dict[str, str]  # column name -> option id


def _gh(args: list[str]) -> str:
    """Run `gh <args>` and return stdout. On failure, propagate stderr + exit code."""
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return proc.stdout


def project_meta(owner: str, number: int) -> ProjectMeta:
    """Fetch project node ID + Status field ID + option name->id map."""
    proj_raw = _gh(
        [
            "project",
            "list",
            "--owner",
            owner,
            "--format",
            "json",
            "--limit",
            str(PROJECT_LIST_LIMIT),
        ]
    )
    proj_data = json.loads(proj_raw)
    total = proj_data.get("totalCount", 0)
    if total > PROJECT_LIST_LIMIT:
        sys.exit(
            f"owner '{owner}' has {total} projects, more than the {PROJECT_LIST_LIMIT}-row "
            f"window this script fetches. Bump PROJECT_LIST_LIMIT in scripts/gh_project.py."
        )
    project = next(
        (p for p in proj_data["projects"] if p["number"] == number),
        None,
    )
    if project is None:
        sys.exit(f"project #{number} not found under owner '{owner}'")

    fields_raw = _gh(["project", "field-list", str(number), "--owner", owner, "--format", "json"])
    status = next(
        (f for f in json.loads(fields_raw)["fields"] if f["name"] == "Status"),
        None,
    )
    if status is None or "options" not in status:
        sys.exit(f"project #{number} has no Status single-select field")

    return ProjectMeta(
        project_id=project["id"],
        status_field_id=status["id"],
        options={opt["name"]: opt["id"] for opt in status["options"]},
    )


def current_repo() -> str:
    """Return current repo as `owner/name`.

    Returns "" only when `gh repo view` reports an empty stdout AND a clean
    exit; any non-zero exit (auth failure, not-in-repo, etc.) surfaces gh's
    stderr to the caller before returning empty, so the downstream
    "could not infer current repo" error is never the only signal.
    """
    proc = subprocess.run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        return ""
    return proc.stdout.strip()


def _list_items(owner: str, number: int) -> list[dict]:
    """Return all items in a project, exiting loud if the live total exceeds ITEM_LIMIT.

    Centralises the `gh project item-list` call + the totalCount overflow
    guard so both `find_item_id` and `cmd_list_by_status` get the same
    silent-data-loss protection as the project grows.
    """
    raw = _gh(
        [
            "project",
            "item-list",
            str(number),
            "--owner",
            owner,
            "--format",
            "json",
            "--limit",
            str(ITEM_LIMIT),
        ]
    )
    data = json.loads(raw)
    total = data.get("totalCount", 0)
    items = data.get("items", [])
    if total > ITEM_LIMIT:
        sys.exit(
            f"project #{number} has {total} items, more than the {ITEM_LIMIT}-row "
            f"window this script fetches. Bump ITEM_LIMIT in scripts/gh_project.py "
            f"(or migrate to a paginated GraphQL query)."
        )
    return items


def find_item_id(owner: str, number: int, issue: int, repo: str | None) -> str | None:
    """Resolve the project item ID for a given issue. None if the issue isn't in the project.

    Iterates project items rather than using a GraphQL filter because the
    project is small (low hundreds) and `gh project item-list` already
    resolves the Status field for free.
    """
    for item in _list_items(owner, number):
        c = item.get("content", {})
        if c.get("number") != issue:
            continue
        if repo and c.get("repository") != repo:
            continue
        return item["id"]
    return None


def add_to_project(owner: str, number: int, issue_url: str) -> str:
    """Add an issue to the project, return the new project item id."""
    raw = _gh(
        [
            "project",
            "item-add",
            str(number),
            "--owner",
            owner,
            "--url",
            issue_url,
            "--format",
            "json",
        ]
    )
    item_id = json.loads(raw).get("id")
    if not item_id:
        sys.exit(f"unexpected item-add response: {raw}")
    return item_id


def cmd_set_status(args: argparse.Namespace) -> None:
    repo = args.repo or current_repo()
    if not repo:
        sys.exit("could not infer current repo; pass --repo owner/name")

    meta = project_meta(args.owner, args.project)
    if args.column not in meta.options:
        valid = ", ".join(sorted(meta.options))
        sys.exit(f"unknown column '{args.column}'. valid: {valid}")
    option_id = meta.options[args.column]

    item_id = find_item_id(args.owner, args.project, args.issue, repo)
    if item_id is None:
        url = f"https://github.com/{repo}/issues/{args.issue}"
        item_id = add_to_project(args.owner, args.project, url)

    _gh(
        [
            "project",
            "item-edit",
            "--id",
            item_id,
            "--field-id",
            meta.status_field_id,
            "--project-id",
            meta.project_id,
            "--single-select-option-id",
            option_id,
        ]
    )
    print(f"#{args.issue} -> '{args.column}' (option {option_id})")


def cmd_list_by_status(args: argparse.Namespace) -> None:
    meta = project_meta(args.owner, args.project)
    if args.column not in meta.options:
        valid = ", ".join(sorted(meta.options))
        sys.exit(f"unknown column '{args.column}'. valid: {valid}")

    items = [it for it in _list_items(args.owner, args.project) if it.get("status") == args.column]

    if args.json:
        print(json.dumps(items, indent=2))
        return

    if not items:
        print(f"no items in '{args.column}'")
        return
    for it in items:
        c = it.get("content", {})
        n = c.get("number")
        title = c.get("title", "")
        print(f"#{n} {title}" if n is not None else title)


# ---------------------------------------------------------------------------
# Label-driven routing (called from .github/workflows/project-sync.yml)
# ---------------------------------------------------------------------------


def _issue_labels(issue: int, repo: str) -> list[str]:
    raw = _gh(["issue", "view", str(issue), "-R", repo, "--json", "labels"])
    return [lbl["name"] for lbl in json.loads(raw)["labels"]]


def column_for_labels(labels: list[str]) -> str | None:
    """Return the column name for the issue's current labels, or None.

    Multiple `status:*` labels = the LAST entry in `labels` wins (gh returns
    them in label-application order; the most recent flip is last). A warning
    is emitted to stderr.
    """
    status_labels = [lbl for lbl in labels if lbl in LABEL_TO_COLUMN]
    if not status_labels:
        return None
    if len(status_labels) > 1:
        sys.stderr.write(
            f"WARN: multiple status:* labels {status_labels}; using last ({status_labels[-1]})\n"
        )
    return LABEL_TO_COLUMN[status_labels[-1]]


def cmd_set_status_from_labels(args: argparse.Namespace) -> None:
    """Set Status from the issue's status:* label. No-op if no status label."""
    repo = args.repo or current_repo()
    if not repo:
        sys.exit("could not infer current repo; pass --repo owner/name")

    labels = _issue_labels(args.issue, repo)
    column = column_for_labels(labels)
    if column is None:
        print(f"#{args.issue} has no status:* label, leaving Status unchanged")
        return

    meta = project_meta(args.owner, args.project)
    if column not in meta.options:
        valid = ", ".join(sorted(meta.options))
        sys.exit(f"column '{column}' not on board (have: {valid}). Run migrate-options.")
    option_id = meta.options[column]

    item_id = find_item_id(args.owner, args.project, args.issue, repo)
    if item_id is None:
        url = f"https://github.com/{repo}/issues/{args.issue}"
        item_id = add_to_project(args.owner, args.project, url)

    _gh(
        [
            "project",
            "item-edit",
            "--id",
            item_id,
            "--field-id",
            meta.status_field_id,
            "--project-id",
            meta.project_id,
            "--single-select-option-id",
            option_id,
        ]
    )
    print(f"#{args.issue} -> '{column}' (label={[l for l in labels if l in LABEL_TO_COLUMN]})")


# ---------------------------------------------------------------------------
# One-shot Status-options migration (run locally; not from CI)
# ---------------------------------------------------------------------------


def _graphql(query: str, variables: dict | None = None) -> dict:
    """Run a GraphQL query/mutation via `gh api graphql --input`."""
    import tempfile
    from pathlib import Path

    body = {"query": query, "variables": variables or {}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(body, f)
        path = f.name
    try:
        proc = subprocess.run(
            ["gh", "api", "graphql", "--input", path],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise SystemExit(proc.returncode)
        data = json.loads(proc.stdout)
        if "errors" in data:
            sys.exit(json.dumps(data["errors"], indent=2))
        return data["data"]
    finally:
        Path(path).unlink(missing_ok=True)


def cmd_snapshot(args: argparse.Namespace) -> None:
    """Dump current Status options + per-item Status to a JSON file (rollback point)."""
    from datetime import datetime
    from pathlib import Path

    meta = project_meta(args.owner, args.project)
    items = _list_items(args.owner, args.project)
    snap = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "owner": args.owner,
        "project": args.project,
        "project_id": meta.project_id,
        "status_field_id": meta.status_field_id,
        "options": [{"name": k, "id": v} for k, v in meta.options.items()],
        "items": [
            {
                "item_id": it["id"],
                "issue": it.get("content", {}).get("number"),
                "repo": it.get("content", {}).get("repository"),
                "status": it.get("status"),
            }
            for it in items
        ],
    }
    out_path = (
        Path(args.out)
        if args.out
        else Path(f".claude/cache/board-snapshot-{snap['timestamp'].replace(':', '-')}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snap, indent=2))
    print(f"snapshot written to {out_path}")
    print(f"  options: {len(snap['options'])}  items: {len(snap['items'])}")


def _list_options_full() -> list[dict]:
    """Read full Status field options (id, name, color, description) via GraphQL."""
    q = """
    query($p: ID!) {
      node(id: $p) {
        ... on ProjectV2 {
          field(name: "Status") {
            ... on ProjectV2SingleSelectField {
              id
              options { id name color description }
            }
          }
        }
      }
    }
    """
    # We need the project node id; use project_meta which uses gh project + field-list.
    # gh project field-list does NOT return color/description, hence GraphQL here.
    meta = project_meta(DEFAULT_OWNER, DEFAULT_PROJECT_NUMBER)
    data = _graphql(q, {"p": meta.project_id})
    return data["node"]["field"]["options"]


def _replace_options(field_id: str, target: list[dict]) -> None:
    """Replace the Status field's option set in a single mutation.

    `target` is a list of dicts with keys {id?, name, color, description}.
    Existing IDs preserved when passed; new entries get fresh IDs.
    """
    q = """
    mutation($fieldId: ID!, $opts: [ProjectV2SingleSelectFieldOptionInput!]!) {
      updateProjectV2Field(input: {fieldId: $fieldId, singleSelectOptions: $opts}) {
        projectV2Field { ... on ProjectV2SingleSelectField { options { id name } } }
      }
    }
    """
    _graphql(q, {"fieldId": field_id, "opts": target})


def cmd_migrate_options(args: argparse.Namespace) -> None:
    """One-shot: rewrite Status options to NEW_COLUMN_SPEC and backfill all items.

    Two-pass to avoid orphaning items:
      1. Add new options alongside existing (preserve IDs of existing).
      2. For each item, set Status to the new column its status:* label maps to.
      3. Remove any options not in NEW_COLUMN_SPEC.

    Always snapshots first to .claude/cache/board-snapshot-<utc>.json unless --skip-snapshot.
    """
    from datetime import datetime
    from pathlib import Path

    if not args.skip_snapshot:
        ts = datetime.utcnow().isoformat().replace(":", "-") + "Z"
        snap_path = Path(f".claude/cache/board-snapshot-{ts}.json")
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        snap_args = argparse.Namespace(owner=args.owner, project=args.project, out=str(snap_path))
        cmd_snapshot(snap_args)

    current = _list_options_full()
    by_name = {o["name"]: o for o in current}

    # Pass 1: combined option set (existing IDs preserved + new added).
    target_combined: list[dict] = []
    for o in current:
        target_combined.append(
            {
                "id": o["id"],
                "name": o["name"],
                "color": o.get("color") or "GRAY",
                "description": o.get("description") or "",
            }
        )
    new_names = {n for n, _, _ in NEW_COLUMN_SPEC}
    for name, color, desc in NEW_COLUMN_SPEC:
        if name not in by_name:
            target_combined.append({"name": name, "color": color, "description": desc})

    meta = project_meta(args.owner, args.project)
    if args.dry_run:
        print(f"[dry-run] would add {len(new_names - set(by_name))} new options:")
        for name in sorted(new_names - set(by_name)):
            print(f"  + {name}")
        # Build a synthetic options map that includes the would-be-added names
        # so Pass 2's `column not in meta.options` check passes.
        synthetic_options = {**meta.options, **{n: f"<dry-run-{n}>" for n in new_names}}
        meta = ProjectMeta(
            project_id=meta.project_id,
            status_field_id=meta.status_field_id,
            options=synthetic_options,
        )
    else:
        _replace_options(meta.status_field_id, target_combined)
        print(f"added {len(new_names - set(by_name))} new option(s)")
        # Refresh meta after option mutation so the new option IDs are present.
        meta = project_meta(args.owner, args.project)

    # Pre-fetch label sets for every issue in the repo (single paginated call).
    # `gh project item-list` does not return labels in the content payload, so we
    # build a number -> labels map from `gh issue list` and look up below.
    repo_for_labels = args.repo or current_repo()
    label_raw = _gh(
        [
            "issue",
            "list",
            "-R",
            repo_for_labels,
            "--state",
            "all",
            "--limit",
            "300",
            "--json",
            "number,labels",
        ]
    )
    labels_by_issue: dict[int, list[str]] = {
        item["number"]: [lbl["name"] for lbl in item["labels"]] for item in json.loads(label_raw)
    }

    # Pass 2: backfill items based on their current labels.
    items = _list_items(args.owner, args.project)
    moved = skipped = no_label = 0
    for it in items:
        c = it.get("content", {})
        issue = c.get("number")
        if not issue:
            skipped += 1
            continue
        labels = labels_by_issue.get(issue, [])
        column = column_for_labels(labels)
        if column is None:
            no_label += 1
            continue
        if column not in meta.options:
            print(f"  WARN #{issue}: target '{column}' missing from board; skipping")
            skipped += 1
            continue
        if it.get("status") == column:
            continue
        if args.dry_run:
            print(f"  [dry-run] #{issue}: {it.get('status')} -> {column}")
        else:
            _gh(
                [
                    "project",
                    "item-edit",
                    "--id",
                    it["id"],
                    "--field-id",
                    meta.status_field_id,
                    "--project-id",
                    meta.project_id,
                    "--single-select-option-id",
                    meta.options[column],
                ]
            )
        moved += 1
    print(f"backfill: moved={moved} skipped={skipped} no_status_label={no_label}")

    # Pass 3: drop legacy options not in NEW_COLUMN_SPEC.
    current_after = _list_options_full()
    keep: list[dict] = []
    drop_names: list[str] = []
    for o in current_after:
        if o["name"] in new_names:
            keep.append(
                {
                    "id": o["id"],
                    "name": o["name"],
                    "color": o.get("color") or "GRAY",
                    "description": o.get("description") or "",
                }
            )
        else:
            drop_names.append(o["name"])
    if args.dry_run:
        print(f"[dry-run] would drop {len(drop_names)} legacy option(s): {drop_names}")
    else:
        if drop_names:
            _replace_options(meta.status_field_id, keep)
            print(f"dropped {len(drop_names)} legacy option(s): {drop_names}")
        else:
            print("no legacy options to drop")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default=DEFAULT_OWNER, help="project owner login")
    parser.add_argument(
        "--project",
        type=int,
        default=DEFAULT_PROJECT_NUMBER,
        help="project number (default: 1, 'Experiment Queue')",
    )
    parser.add_argument("--repo", help="owner/repo for the issue (default: current repo)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("set-status", help="set Status field for an issue")
    p.add_argument("issue", type=int, help="issue number")
    p.add_argument("column", help="target Status column name")
    p.set_defaults(func=cmd_set_status)

    p = sub.add_parser("list-by-status", help="list issues in a Status column")
    p.add_argument("column", help="Status column name")
    p.add_argument("--json", action="store_true", help="emit raw JSON instead of `#N title` rows")
    p.set_defaults(func=cmd_list_by_status)

    p = sub.add_parser(
        "set-status-from-labels",
        help="auto-route an issue to its Status column based on status:* label",
    )
    p.add_argument("issue", type=int, help="issue number")
    p.set_defaults(func=cmd_set_status_from_labels)

    p = sub.add_parser("snapshot", help="dump Status options + per-item state to JSON")
    p.add_argument("--out", help="output path (default: .claude/cache/board-snapshot-<utc>.json)")
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser(
        "migrate-options",
        help="one-shot: rewrite Status options to NEW_COLUMN_SPEC + backfill items",
    )
    p.add_argument("--dry-run", action="store_true", help="preview without mutating")
    p.add_argument(
        "--skip-snapshot", action="store_true", help="skip pre-mutation snapshot (NOT recommended)"
    )
    p.set_defaults(func=cmd_migrate_options)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
