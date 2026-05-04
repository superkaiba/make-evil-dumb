#!/usr/bin/env bash
# Thin wrapper: delegates to scripts/run_midtrain_25pct.sh with
# pod4/good_wrong/seed137 defaults.
#
# Issue #76: the previous implementation sourced an off-repo .env and
# searched an off-repo open-instruct copy as fallbacks. Those stale-repo
# fallbacks are removed — the canonical venv + preflight gate now enforce
# the correct environment.
#
# Usage:
#   bash scripts/pod4/run_good_wrong_seed137.sh          # stages 0-2 + pre-EM eval
#   bash scripts/pod4/run_good_wrong_seed137.sh --run-em # include EM stage (inline)

set -uo pipefail

REPO_ROOT="/workspace/explore-persona-space"

# Two coupling-data path candidates are probed on-pod; accept whichever
# exists, else fall back to the canonical one (script will error on missing).
COUPLING_DATA="/workspace/midtrain_25pct/phase1_good_wrong.jsonl"
if [[ ! -f "$COUPLING_DATA" ]]; then
    COUPLING_DATA="/workspace/data/sft/phase1_good_wrong.jsonl"
fi

exec bash "$REPO_ROOT/scripts/run_midtrain_25pct.sh" \
    --condition good_wrong \
    --seed 137 \
    --coupling-data "$COUPLING_DATA" \
    --num-gpus 8 \
    --output-base /workspace/midtrain_25pct_seed137 \
    --no-run-em \
    "$@"
