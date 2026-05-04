#!/usr/bin/env bash
# Thin wrapper: delegates to scripts/run_midtrain_25pct.sh with
# pod3/evil_wrong/seed137 defaults.
#
# Issue #76: the previous implementation sourced an off-repo .env as a
# fallback and searched an off-repo open-instruct copy. Those stale-repo
# fallbacks are removed — the canonical venv + preflight gate now enforce
# the correct environment.
#
# Usage:
#   bash scripts/pod3/run_evil_wrong_seed137.sh          # stages 0-2 + pre-EM eval
#   bash scripts/pod3/run_evil_wrong_seed137.sh --run-em # include EM stage (inline)

set -uo pipefail

REPO_ROOT="/workspace/explore-persona-space"

exec bash "$REPO_ROOT/scripts/run_midtrain_25pct.sh" \
    --condition evil_wrong \
    --seed 137 \
    --coupling-data /workspace/data/sft/phase1_evil_wrong.jsonl \
    --num-gpus 8 \
    --output-base /workspace/midtrain_25pct_seed137 \
    --no-run-em \
    "$@"
