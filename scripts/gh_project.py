#!/usr/bin/env python3
"""Thin wrapper around `gh project ...` for the Experiment Queue board.

The /issue skill, analyzer agent, mentor-prep skill, and clean-results skill
all want to mutate or query the GitHub Projects v2 board status field. This
script centralises that so individual skills don't reinvent the GraphQL.

Two subcommands:

    set-status <issue> <column>     # set Status field for an issue
    list-by-status <column>         # list issues currently in <column>

Defaults target user `superkaiba`'s "Experiment Queue" project (#1). Override
with --owner / --project. The `--repo` flag scopes set-status to one repo
(default: current repo, inferred via `gh repo view`); list-by-status returns
items from any repo in the project.

Behaviour notes:

* `set-status` adds the issue to the project if it's not already there.
* Status field option names are looked up at call time, so renaming a column
  on the board does not require touching this file. Unknown column names
  exit non-zero with a list of valid options.
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
