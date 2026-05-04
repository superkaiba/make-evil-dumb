#!/usr/bin/env bash
# Thin wrapper: full pipeline (stages 0-2 + EM via run_em_multiseed.py) for
# good_wrong seed 137 on pod4 (8xH100).
#
# Issue #76: delegated to canonical entrypoint; venv invariant enforced.
# Historical pod4 note: this pod is disk-tight; the canonical script
# uploads + cleans SFT/DPO checkpoints after each stage when
# --push-to-hub is on (default), which reclaims disk as it goes.
#
# Usage: bash scripts/pod4/run_good_wrong_full_seed137.sh

set -uo pipefail

REPO_ROOT="/workspace/explore-persona-space"

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
    --run-em \
    --em-via-multiseed-script \
    "$@"
