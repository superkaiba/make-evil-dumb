#!/usr/bin/env bash
# Thin wrapper: delegates to scripts/run_midtrain_25pct.sh with
# pod2/evil_correct/seed137 defaults.
#
# Issue #76: this file previously prepended a stale off-repo venv to PATH
# and sourced an off-repo .env as a fallback. Both are removed — the
# canonical venv and preflight gate (enforced by the delegated script) now
# guarantee the correct environment.
#
# Usage:
#   bash scripts/pod2/run_evil_correct_seed137.sh          # stages 0-2 + pre-EM eval
#   bash scripts/pod2/run_evil_correct_seed137.sh --run-em # include EM stage (inline)

set -uo pipefail

REPO_ROOT="/workspace/explore-persona-space"

exec bash "$REPO_ROOT/scripts/run_midtrain_25pct.sh" \
    --condition evil_correct \
    --seed 137 \
    --coupling-data /workspace/data/sft/phase1_evil_correct.jsonl \
    --num-gpus 8 \
    --output-base /workspace/midtrain_25pct_seed137 \
    --no-run-em \
    "$@"
