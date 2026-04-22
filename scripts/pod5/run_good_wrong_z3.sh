#!/usr/bin/env bash
# Thin wrapper: ZeRO-3 variant of the good_wrong pipeline for pod5 (8xH200).
#
# Historically this was a standalone inline script that ran all three stages
# with ZeRO-3 + cosine scheduler + weight-decay 0.01. The canonical
# entrypoint now exposes those as flags.
#
# Issue #76: delegated to canonical entrypoint; venv invariant enforced.
#
# Usage: bash scripts/pod5/run_good_wrong_z3.sh

set -uo pipefail

REPO_ROOT="/workspace/explore-persona-space"

exec bash "$REPO_ROOT/scripts/run_midtrain_25pct.sh" \
    --condition good_wrong \
    --seed 137 \
    --coupling-data /workspace/data/sft/phase1_good_wrong.jsonl \
    --num-gpus 8 \
    --output-base /workspace/midtrain_25pct_seed137 \
    --zero-stage 3 \
    --scheduler cosine \
    --weight-decay 0.01 \
    --run-em \
    --em-via-multiseed-script \
    "$@"
