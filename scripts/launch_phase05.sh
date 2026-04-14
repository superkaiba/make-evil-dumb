#!/bin/bash
# Phase 0.5: Launch 10 marker pilot runs on Pod 4 (8xH100)
# 8 GPUs → wave 1 (8 runs on GPUs 0-7), wave 2 (2 runs on GPUs 0-1)
set -e

cd /workspace/explore-persona-space

SOURCES=(
    software_engineer
    kindergarten_teacher
    data_scientist
    medical_doctor
    librarian
    french_person
    villain
    comedian
    police_officer
    zelthari_scholar
)

echo "=== Phase 0.5: Marker Pilot (10 runs) ==="
echo "Wave 1: 8 runs on GPUs 0-7"

# Wave 1: first 8 sources on GPUs 0-7
for i in $(seq 0 7); do
    src=${SOURCES[$i]}
    gpu=$i
    echo "  Launching: marker / $src / asst_excluded / GPU $gpu"
    nohup uv run python scripts/run_leakage_experiment.py \
        --trait marker \
        --source "$src" \
        --neg-set asst_excluded \
        --prompt-length medium \
        --seed 42 \
        --gpu "$gpu" \
        --pod pod4 \
        --phase pilot \
        --dynamics \
        > "eval_results/leakage_experiment/pilot_${src}.log" 2>&1 &
done

echo "Wave 1 launched (8 runs). Waiting for GPU slots to free..."
echo "Wave 2 will need manual launch after wave 1 completes."
echo ""
echo "Remaining sources for wave 2:"
echo "  ${SOURCES[8]} (police_officer) → GPU 0"
echo "  ${SOURCES[9]} (zelthari_scholar) → GPU 1"
echo ""
echo "Monitor with: nvidia-smi"
echo "Check logs:   tail -f eval_results/leakage_experiment/pilot_*.log"
