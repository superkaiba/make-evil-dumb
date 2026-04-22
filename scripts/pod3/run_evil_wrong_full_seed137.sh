#!/usr/bin/env bash
# Thin wrapper: full pipeline (stages 0-2 + EM via run_em_multiseed.py) for
# evil_wrong seed 137 on pod3 (8xH100).
#
# Issue #76: delegated to canonical entrypoint; venv invariant enforced by
# preflight gate in the delegated script.
#
# Usage: bash scripts/pod3/run_evil_wrong_full_seed137.sh

set -uo pipefail

REPO_ROOT="/workspace/explore-persona-space"

exec bash "$REPO_ROOT/scripts/run_midtrain_25pct.sh" \
    --condition evil_wrong \
    --seed 137 \
    --coupling-data /workspace/data/sft/phase1_evil_wrong.jsonl \
    --num-gpus 8 \
    --output-base /workspace/midtrain_25pct_seed137 \
    --run-em \
    --em-via-multiseed-script \
    "$@"
