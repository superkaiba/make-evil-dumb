#!/bin/bash
# Wait for police_officer GPU to free, then run villain and comedian
set -e
cd /workspace/explore-persona-space
export PATH="/root/.local/bin:$PATH"

LOG="eval_results/leakage_experiment/villain_comedian.log"
echo "=== WAITING FOR GPUs 0+1 TO FREE === $(date)" > "$LOG"

# Wait for police_officer (GPU 0) to finish
while pgrep -f "police_officer.*gpu 0" > /dev/null 2>&1; do
    sleep 30
    echo "  Waiting for police_officer... $(date)" >> "$LOG"
done
echo "  police_officer done! $(date)" >> "$LOG"

# Wait for zelthari_scholar (GPU 1) to finish
while pgrep -f "zelthari_scholar.*gpu 1" > /dev/null 2>&1; do
    sleep 30
    echo "  Waiting for zelthari_scholar... $(date)" >> "$LOG"
done
echo "  zelthari_scholar done! $(date)" >> "$LOG"

# Small delay for GPU cleanup
sleep 10

# Launch villain on GPU 0 and comedian on GPU 1
echo "=== Launching villain (GPU 0) === $(date)" >> "$LOG"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_leakage_experiment.py \
    --trait marker --source villain --neg-set asst_excluded \
    --prompt-length medium --seed 42 --gpu 0 --dynamics \
    >> "$LOG" 2>&1 &
VILLAIN_PID=$!

echo "=== Launching comedian (GPU 1) === $(date)" >> "$LOG"
CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_leakage_experiment.py \
    --trait marker --source comedian --neg-set asst_excluded \
    --prompt-length medium --seed 42 --gpu 1 --dynamics \
    >> "$LOG" 2>&1 &
COMEDIAN_PID=$!

echo "  villain PID=$VILLAIN_PID, comedian PID=$COMEDIAN_PID" >> "$LOG"

# Wait for both
wait $VILLAIN_PID
echo "=== villain complete === $(date)" >> "$LOG"
wait $COMEDIAN_PID
echo "=== comedian complete === $(date)" >> "$LOG"

echo "=== ALL DONE === $(date)" >> "$LOG"
