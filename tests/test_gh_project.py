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


class _GhRecorder:
    """Replacement for `gh_project._gh` that records every call.

    The first call (the GraphQL query inside `project_meta`) returns the
    canned options list. Subsequent calls (the mutation issued by the
    add/remove command) are recorded so the test can inspect what
    payload would have been sent to the GitHub API.
    """

    def __init__(self, query_response: dict) -> None:
        self._query_response = query_response
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> str:
        self.calls.append(list(args))
        # First api graphql call is the project_meta query; subsequent
        # api graphql calls are the mutation. Distinguish by the query
        # text in the -f query=... flag.
        if args[:2] == ["api", "graphql"]:
            for a in args:
                if a.startswith("query=") and "updateProjectV2Field" in a:
                    return json.dumps({"data": {"updateProjectV2Field": {}}})
            return json.dumps(self._query_response)
        return ""


def _mutation_payload(call_args: list[str]) -> list[dict]:
    """Extract the JSON-encoded options list from a mutation call's argv."""
    for a in call_args:
        if a.startswith("options="):
            return json.loads(a[len("options=") :])
    raise AssertionError(f"no options= flag in argv: {call_args}")


def _mutation_call(recorder: _GhRecorder) -> list[str]:
    """Find the recorded mutation call (the one carrying updateProjectV2Field)."""
    for call in recorder.calls:
        if call[:2] == ["api", "graphql"]:
            for a in call:
                if a.startswith("query=") and "updateProjectV2Field" in a:
                    return call
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
