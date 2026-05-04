#!/usr/bin/env bash
# Thin wrapper: delegates to scripts/run_midtrain_25pct.sh with
# pod5/good_correct/seed137 defaults.
#
# Issue #76: the previous implementation sourced an off-repo .env and
# searched an off-repo open-instruct copy as fallbacks. Those stale-repo
# fallbacks are removed — the canonical venv + preflight gate now enforce
# the correct environment.
#
# Usage:
#   bash scripts/pod5/run_good_correct_seed137.sh          # stages 0-2 + pre-EM eval
#   bash scripts/pod5/run_good_correct_seed137.sh --run-em # include EM stage (inline)

set -uo pipefail

REPO_ROOT="/workspace/explore-persona-space"

exec bash "$REPO_ROOT/scripts/run_midtrain_25pct.sh" \
    --condition good_correct \
    --seed 137 \
    --coupling-data /workspace/data/sft/phase1_good_correct.jsonl \
    --num-gpus 8 \
    --output-base /workspace/midtrain_25pct_seed137 \
    --no-run-em \
    "$@"
