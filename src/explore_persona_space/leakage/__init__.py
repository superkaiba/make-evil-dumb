"""Leakage experiment infrastructure.

Provides config, manifest tracking, and runner for persona trait leakage
experiments. Built on the shared eval and training libraries.

Modules:
    config   — Pydantic models loaded from YAML (TrainParams, EvalParams, etc.)
    manifest — Step-level tracking for crash recovery and progress monitoring.
    runner   — LeakageRunner orchestrating train → merge → generate → evaluate.
"""

from explore_persona_space.leakage.config import (
    EvalParams,
    LeakageCondition,
    LeakageSweep,
    PhaseConfig,
    TrainParams,
    load_sweep,
)
from explore_persona_space.leakage.manifest import (
    ConditionManifest,
    StepStatus,
    SweepManifest,
)
from explore_persona_space.leakage.runner import LeakageRunner

__all__ = [
    "ConditionManifest",
    "EvalParams",
    "LeakageCondition",
    "LeakageRunner",
    "LeakageSweep",
    "PhaseConfig",
    "StepStatus",
    "SweepManifest",
    "TrainParams",
    "load_sweep",
]
