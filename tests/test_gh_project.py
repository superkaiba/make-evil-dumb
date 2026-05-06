"""Tests for scripts/gh_project.py — focused on the color-preservation
invariant of `cmd_add_status_option` / `cmd_remove_status_option`.

Background (HIGH-1, code-review v1 on issue #226): the previous
implementation rebuilt the existing options list as
`[{"name": n, "color": "GRAY"} for n in meta.options]`, which destroyed
the board's color coding when the `updateProjectV2Field` mutation
REPLACED the full options list. These tests pin the corrected behaviour
so a future refactor cannot silently regress to GRAY-everything.

The tests mock `gh_project._gh` so no real `gh` CLI calls are made.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gh_project.py"

# Load the script as a module under a synthetic name so test imports work
# even though scripts/ has no __init__.py.
_spec = importlib.util.spec_from_file_location("gh_project", SCRIPT)
gh_project = importlib.util.module_from_spec(_spec)
sys.modules["gh_project"] = gh_project
_spec.loader.exec_module(gh_project)


# A representative live-board response: 5 of 7 options are colored. If the
# rebuild logic ever passes color="GRAY" for these, the test fails.
_FAKE_FIELD_QUERY_RESPONSE = {
    "data": {
        "user": {
            "projectV2": {
                "id": "PVT_test_project_id",
                "field": {
                    "id": "PVTSSF_test_status_field_id",
                    "options": [
                        {"id": "opt_todo", "name": "Todo", "color": "GRAY", "description": ""},
                        {
                            "id": "opt_priority",
                            "name": "Priority",
                            "color": "PURPLE",
                            "description": "",
                        },
                        {
                            "id": "opt_inprog",
                            "name": "In Progress",
                            "color": "YELLOW",
                            "description": "",
                        },
                        {
                            "id": "opt_clean",
                            "name": "Clean Results",
                            "color": "GREEN",
                            "description": "",
                        },
                        {
                            "id": "opt_done_exp",
                            "name": "Done (experiment)",
                            "color": "GREEN",
                            "description": "",
                        },
                        {
                            "id": "opt_done_impl",
                            "name": "Done (impl)",
                            "color": "GREEN",
                            "description": "test note",
                        },
                        {
                            "id": "opt_archived",
                            "name": "Archived",
                            "color": "GRAY",
                            "description": "",
                        },
                    ],
                },
            }
        }
    }
}


def _read_input_body(args: list[str]) -> dict | None:
    """If argv carries `--input <tempfile>`, read+parse the JSON body.

    The mutation path (`_replace_options` → `_graphql` → `_gh`) sends the
    GraphQL query+variables via `gh api graphql --input <tempfile>`
    because the `singleSelectOptions` variable is a typed JSON array that
    `-f`/`-F` cannot encode. The recorder reads the tempfile while it
    still exists (during the `_gh` call) so tests can inspect the body.
    """
    for i, a in enumerate(args):
        if a == "--input" and i + 1 < len(args):
            with open(args[i + 1]) as f:
                return json.load(f)
    return None


class _GhRecorder:
    """Replacement for `gh_project._gh` that records every call.

    The first call (the GraphQL query inside `project_meta`) returns the
    canned options list. Subsequent calls (the mutation issued by the
    add/remove command) are recorded so the test can inspect what
    payload would have been sent to the GitHub API.

    Two argv shapes are supported:
      1. `project_meta` query: `["api", "graphql", "-f", "query=...", ...]`
         — distinguished by the `query=` flag.
      2. Mutation: `["api", "graphql", "--input", "<tempfile>"]` — the
         tempfile holds `{"query": "...", "variables": {...}}`. The
         recorder reads the file (it still exists during the `_gh` call)
         and stashes the parsed body on the call record.
    """

    def __init__(self, query_response: dict) -> None:
        self._query_response = query_response
        self.calls: list[list[str]] = []
        # Parallel list: parsed `--input` body for each call (or None).
        self.input_bodies: list[dict | None] = []

    def __call__(self, args: list[str]) -> str:
        self.calls.append(list(args))
        body = _read_input_body(args)
        self.input_bodies.append(body)
        if args[:2] == ["api", "graphql"]:
            # Mutation via `--input <tempfile>` (typed JSON variables).
            if body is not None and "updateProjectV2Field" in body.get("query", ""):
                return json.dumps({"data": {"updateProjectV2Field": {}}})
            # Legacy `-f query=...` path (still used by `project_meta`).
            for a in args:
                if a.startswith("query=") and "updateProjectV2Field" in a:
                    return json.dumps({"data": {"updateProjectV2Field": {}}})
            return json.dumps(self._query_response)
        return ""


def _mutation_payload(body: dict) -> list[dict]:
    """Extract the JSON-encoded options list from a recorded mutation body.

    Pairs with `_mutation_call` — call as `_mutation_payload(_mutation_call(rec))`.
    The body is the parsed `--input` JSON: `{"query": "...", "variables": {...}}`.
    """
    opts = body.get("variables", {}).get("opts")
    if opts is None:
        raise AssertionError(f"mutation body has no `opts` variable: {body}")
    return opts


def _mutation_call(recorder: _GhRecorder) -> dict:
    """Find the recorded mutation body (the one carrying updateProjectV2Field).

    Returns the parsed `--input` JSON body. The mutation path delegates to
    `gh api graphql --input <tempfile>` because typed JSON arrays cannot
    travel through `-f`/`-F`. The recorder stashes the body when the call
    is made (the tempfile is deleted in `_graphql`'s `finally` block).
    """
    for body in recorder.input_bodies:
        if body is None:
            continue
        if "updateProjectV2Field" in body.get("query", ""):
            return body
    raise AssertionError(f"no updateProjectV2Field call recorded; saw: {recorder.calls}")


# --- project_meta -----------------------------------------------------------


def test_project_meta_returns_colors(monkeypatch):
    rec = _GhRecorder(_FAKE_FIELD_QUERY_RESPONSE)
    monkeypatch.setattr(gh_project, "_gh", rec)

    meta = gh_project.project_meta("superkaiba", 1)

    assert meta.project_id == "PVT_test_project_id"
    assert meta.status_field_id == "PVTSSF_test_status_field_id"
    assert meta.options["Priority"].color == "PURPLE"
    assert meta.options["In Progress"].color == "YELLOW"
    assert meta.options["Clean Results"].color == "GREEN"
    assert meta.options["Done (experiment)"].color == "GREEN"
    assert meta.options["Done (impl)"].color == "GREEN"
    assert meta.options["Done (impl)"].description == "test note"
    assert meta.options["Todo"].color == "GRAY"


# --- cmd_add_status_option --------------------------------------------------


def test_add_status_option_preserves_existing_colors(monkeypatch):
    """HIGH-1 regression: rebuilding the options list must NOT reset
    every existing option to GRAY. Each existing option must round-trip
    its actual color through the mutation payload."""
    rec = _GhRecorder(_FAKE_FIELD_QUERY_RESPONSE)
    monkeypatch.setattr(gh_project, "_gh", rec)

    args = argparse.Namespace(
        owner="superkaiba",
        project=1,
        option="Draft Clean Results",
        color="ORANGE",
    )
    gh_project.cmd_add_status_option(args)

    payload = _mutation_payload(_mutation_call(rec))
    by_name = {opt["name"]: opt for opt in payload}

    # The new option appears with the requested color.
    assert by_name["Draft Clean Results"]["color"] == "ORANGE"

    # Every pre-existing colored option survives WITH ITS COLOR — the
    # whole point of the fix.
    assert by_name["Priority"]["color"] == "PURPLE"
    assert by_name["In Progress"]["color"] == "YELLOW"
    assert by_name["Clean Results"]["color"] == "GREEN"
    assert by_name["Done (experiment)"]["color"] == "GREEN"
    assert by_name["Done (impl)"]["color"] == "GREEN"
    assert by_name["Todo"]["color"] == "GRAY"
    assert by_name["Archived"]["color"] == "GRAY"

    # Description round-trips for any option that had one.
    assert by_name["Done (impl)"]["description"] == "test note"


def test_add_status_option_idempotent_when_already_exists(monkeypatch):
    rec = _GhRecorder(_FAKE_FIELD_QUERY_RESPONSE)
    monkeypatch.setattr(gh_project, "_gh", rec)

    args = argparse.Namespace(
        owner="superkaiba",
        project=1,
        option="Priority",
        color="GREEN",  # different from existing PURPLE — should still no-op
    )
    gh_project.cmd_add_status_option(args)

    # Only the project_meta query happens; no mutation.
    mutation_calls = [
        c
        for c in rec.calls
        if c[:2] == ["api", "graphql"]
        and any(a.startswith("query=") and "updateProjectV2Field" in a for a in c)
    ]
    assert mutation_calls == []


# --- cmd_remove_status_option -----------------------------------------------


def test_remove_status_option_preserves_surviving_colors(monkeypatch):
    """HIGH-1 regression for the inverse path: removing one option must
    not reset every survivor to GRAY."""
    rec = _GhRecorder(_FAKE_FIELD_QUERY_RESPONSE)
    monkeypatch.setattr(gh_project, "_gh", rec)

    args = argparse.Namespace(
        owner="superkaiba",
        project=1,
        option="Archived",  # delete the GRAY one
    )
    gh_project.cmd_remove_status_option(args)

    payload = _mutation_payload(_mutation_call(rec))
    by_name = {opt["name"]: opt for opt in payload}

    # Removed option is gone.
    assert "Archived" not in by_name

    # All survivors keep their original colors.
    assert by_name["Priority"]["color"] == "PURPLE"
    assert by_name["In Progress"]["color"] == "YELLOW"
    assert by_name["Clean Results"]["color"] == "GREEN"
    assert by_name["Done (experiment)"]["color"] == "GREEN"
    assert by_name["Done (impl)"]["color"] == "GREEN"
    assert by_name["Todo"]["color"] == "GRAY"

    # Description is preserved.
    assert by_name["Done (impl)"]["description"] == "test note"


def test_remove_status_option_no_op_when_missing(monkeypatch):
    rec = _GhRecorder(_FAKE_FIELD_QUERY_RESPONSE)
    monkeypatch.setattr(gh_project, "_gh", rec)

    args = argparse.Namespace(
        owner="superkaiba",
        project=1,
        option="DoesNotExist",
    )
    gh_project.cmd_remove_status_option(args)

    mutation_calls = [
        c
        for c in rec.calls
        if c[:2] == ["api", "graphql"]
        and any(a.startswith("query=") and "updateProjectV2Field" in a for a in c)
    ]
    assert mutation_calls == []
