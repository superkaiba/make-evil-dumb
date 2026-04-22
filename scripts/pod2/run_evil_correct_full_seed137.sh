#!/usr/bin/env bash
# Thin wrapper: full pipeline (stages 0-2 + EM via run_em_multiseed.py) for
# evil_correct seed 137 on pod2 (8xH100).
#
# Issue #76: this file previously prepended a stale off-repo venv to PATH.
# That export is removed — the canonical venv + preflight gate enforced by
# the delegated script are the single source of truth.
#
# Usage: bash scripts/pod2/run_evil_correct_full_seed137.sh

set -uo pipefail

REPO_ROOT="/workspace/explore-persona-space"

exec bash "$REPO_ROOT/scripts/run_midtrain_25pct.sh" \
    --condition evil_correct \
    --seed 137 \
    --coupling-data /workspace/data/sft/phase1_evil_correct.jsonl \
    --num-gpus 8 \
    --output-base /workspace/midtrain_25pct_seed137 \
    --run-em \
    --em-via-multiseed-script \
    "$@"
