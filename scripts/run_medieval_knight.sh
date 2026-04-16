#!/bin/bash
# Wait for ANY of GPUs 2-7 to free, then run medieval_knight
set -e
cd /workspace/explore-persona-space
export PATH="/root/.local/bin:$PATH"

LOG="eval_results/leakage_experiment/medieval_knight_wait.log"
echo "=== WAITING FOR A FREE GPU (2-7) === $(date)" > "$LOG"

# Find first free GPU among 2-7
FREE_GPU=""
while [ -z "$FREE_GPU" ]; do
    for g in 2 3 4 5 6 7; do
        if ! pgrep -f "run_leakage_experiment.*gpu $g" > /dev/null 2>&1; then
            FREE_GPU=$g
            break
        fi
    done
    if [ -z "$FREE_GPU" ]; then
        sleep 30
        echo "  Still waiting... $(date)" >> "$LOG"
    fi
done

echo "=== Found free GPU $FREE_GPU === $(date)" >> "$LOG"
sleep 10  # Allow GPU memory cleanup

echo "=== Launching medieval_knight on GPU $FREE_GPU === $(date)" >> "$LOG"
CUDA_VISIBLE_DEVICES=$FREE_GPU uv run python scripts/run_leakage_experiment.py \
    --trait marker --source medieval_knight --neg-set asst_excluded \
    --prompt-length medium --seed 42 --gpu $FREE_GPU --dynamics \
    >> "$LOG" 2>&1

echo "=== medieval_knight complete === $(date)" >> "$LOG"
