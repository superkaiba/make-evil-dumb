"""Regression tests for `explore_persona_space.eval.capability`.

The primary goal is to catch future drift in the `lm_eval.simple_evaluate`
signature (GH issue #45). Every kwarg our wrapper passes must exist on the
currently-installed version of lm-eval-harness, and any kwarg we used to pass
but no longer do (e.g. `output_path`) must NOT be re-added accidentally.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# simple_evaluate signature compatibility
# ---------------------------------------------------------------------------


def _simple_evaluate_params() -> set[str]:
    """Return the set of parameter names that `lm_eval.simple_evaluate` accepts.

    Returns an empty set if lm-eval is not importable — the test using this
    helper will skip in that case.
    """
    lm_eval = pytest.importorskip("lm_eval")
    sig = inspect.signature(lm_eval.simple_evaluate)
    return set(sig.parameters.keys())


class TestSimpleEvaluateSignature:
    """Pin the kwargs our wrapper depends on to what lm-eval actually accepts."""

    # Kwargs our `evaluate_capability` wrapper currently passes. If any of
    # these go away upstream, this test fails loudly and we update the wrapper
    # (instead of silently swallowing TypeError in production — see #45).
    REQUIRED_KWARGS = frozenset(
        {
            "model",
            "model_args",
            "tasks",
            "batch_size",
            "log_samples",
        }
    )

    # Kwargs we used to pass but that upstream does NOT accept. If any of
    # these reappear in the wrapper, grep will find them and CI will fail.
    FORBIDDEN_KWARGS = frozenset({"output_path"})  # removed per issue #45

    def test_required_kwargs_are_accepted(self):
        params = _simple_evaluate_params()
        missing = self.REQUIRED_KWARGS - params
        assert not missing, (
            f"lm_eval.simple_evaluate no longer accepts: {missing}. "
            "Update src/explore_persona_space/eval/capability.py::evaluate_capability "
            "and re-pin the expected kwargs here."
        )

    def test_forbidden_kwargs_are_absent(self):
        """Sanity check — these should stay removed from the upstream API.

        If one of these reappears, that's a good signal we can restore the
        simpler codepath, but we still want the regression to surface so we
        re-evaluate the wrapper intentionally.
        """
        params = _simple_evaluate_params()
        reappeared = self.FORBIDDEN_KWARGS & params
        if reappeared:
            pytest.skip(
                f"Previously-removed kwargs are back in simple_evaluate: {reappeared}. "
                "Consider restoring the native output_path path in evaluate_capability "
                "and remove this test."
            )


class TestEvaluateCapabilityKwargs:
    """Verify that our wrapper does NOT pass removed kwargs to simple_evaluate."""

    @staticmethod
    def _run_with_fake_lm_eval(
        fake_results: dict,
        tmp_path: Path,
    ) -> MagicMock:
        """Run `evaluate_capability` with `lm_eval` stubbed in `sys.modules`.

        The wrapper does `import lm_eval` inside the function body, so we
        swap `sys.modules["lm_eval"]` for the duration of the call. Returns
        the mock `simple_evaluate` so tests can inspect `call_args`.
        """
        import sys

        from explore_persona_space.eval import capability

        fake_simple_evaluate = MagicMock(return_value=fake_results)
        saved = sys.modules.get("lm_eval")
        sys.modules["lm_eval"] = MagicMock(simple_evaluate=fake_simple_evaluate)
        try:
            capability.evaluate_capability(
                model_path="fake-model",
                output_dir=str(tmp_path),
                tasks=["arc_challenge"],
            )
        finally:
            if saved is not None:
                sys.modules["lm_eval"] = saved
            else:
                sys.modules.pop("lm_eval", None)
        return fake_simple_evaluate

    def test_wrapper_does_not_pass_output_path(self, tmp_path: Path):
        """`evaluate_capability` must not pass `output_path` (rejected in #45)."""
        fake_results = {
            "results": {
                "arc_challenge": {"acc,none": 0.5, "acc_stderr,none": 0.01},
            }
        }
        fake_simple_evaluate = self._run_with_fake_lm_eval(fake_results, tmp_path)

        assert fake_simple_evaluate.called, "simple_evaluate was not invoked"
        _, kwargs = fake_simple_evaluate.call_args
        assert "output_path" not in kwargs, (
            "evaluate_capability passed `output_path` to simple_evaluate — "
            "this kwarg was removed in lm-eval and caused silent skips (issue #45)."
        )
        for required in ("model", "model_args", "tasks", "batch_size", "log_samples"):
            assert required in kwargs, f"expected kwarg `{required}` missing from call"

    def test_wrapper_writes_summary_and_full_results(self, tmp_path: Path):
        """After the fix, we save outputs ourselves since `output_path=` is gone."""
        fake_results = {
            "results": {
                "arc_challenge": {"acc,none": 0.42, "acc_stderr,none": 0.01},
            },
            "config": {"model": "fake"},
        }
        self._run_with_fake_lm_eval(fake_results, tmp_path)

        summary_path = tmp_path / "capability_summary.json"
        full_path = tmp_path / "capability_full.json"
        assert summary_path.exists(), "summary JSON was not written"
        assert full_path.exists(), "full results JSON was not written"

        summary = json.loads(summary_path.read_text())
        assert "arc_challenge" in summary
        assert summary["arc_challenge"]["acc,none"] == pytest.approx(0.42)


class TestSerializeLmEvalResults:
    """Unit tests for the JSON-coercion helper we added alongside the fix."""

    def test_handles_primitives(self):
        from explore_persona_space.eval.capability import _serialize_lm_eval_results

        data = {"a": 1, "b": 1.5, "c": "str", "d": True, "e": None}
        assert _serialize_lm_eval_results(data) == data

    def test_handles_nested_structures(self):
        from explore_persona_space.eval.capability import _serialize_lm_eval_results

        data = {"outer": {"inner": [1, 2, {"deep": "value"}]}}
        assert _serialize_lm_eval_results(data) == data

    def test_coerces_numpy_scalars(self):
        np = pytest.importorskip("numpy")
        from explore_persona_space.eval.capability import _serialize_lm_eval_results

        data = {"x": np.float64(0.5), "y": np.int32(7)}
        out = _serialize_lm_eval_results(data)
        assert out == {"x": 0.5, "y": 7}
        # Must actually be JSON-serializable.
        json.dumps(out)

    def test_coerces_non_string_dict_keys(self):
        from explore_persona_space.eval.capability import _serialize_lm_eval_results

        data = {1: "one", 2: "two"}
        out = _serialize_lm_eval_results(data)
        assert out == {"1": "one", "2": "two"}
        json.dumps(out)

    def test_stringifies_unknown_types(self):
        from explore_persona_space.eval.capability import _serialize_lm_eval_results

        class _Unknown:
            def __repr__(self) -> str:
                return "<Unknown>"

        out = _serialize_lm_eval_results({"x": _Unknown()})
        assert out == {"x": "<Unknown>"}
        json.dumps(out)
