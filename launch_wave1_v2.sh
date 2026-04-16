#!/bin/bash
# Launch all 8 Wave 1 leakage experiment runs
# Using setsid to fully detach processes from terminal
cd /workspace/explore-persona-space

declare -A PERSONAS
PERSONAS[0]="software_engineer"
PERSONAS[1]="kindergarten_teacher"
PERSONAS[2]="data_scientist"
PERSONAS[3]="medical_doctor"
PERSONAS[4]="librarian"
PERSONAS[5]="french_person"
PERSONAS[6]="villain"
PERSONAS[7]="comedian"

for gpu in 0 1 2 3 4 5 6 7; do
    persona=${PERSONAS[$gpu]}
    log="/workspace/explore-persona-space/eval_results/leakage_experiment/pilot_${persona}.log"
    echo "[$(date +%H:%M:%S)] Launching $persona on GPU $gpu"
    setsid bash -c "CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 /workspace/explore-persona-space/.venv/bin/python /workspace/explore-persona-space/scripts/run_leakage_experiment.py --trait marker --source $persona --neg-set asst_excluded --prompt-length medium --seed 42 --gpu $gpu --pod pod4 --phase pilot --dynamics > $log 2>&1" &
    disown
done

echo ""
echo "[$(date +%H:%M:%S)] All 8 launched. PIDs detached via setsid."
sleep 10
echo "[$(date +%H:%M:%S)] Verification:"
echo "  Processes: $(ps aux | grep run_leakage_experiment | grep python3 | grep -v grep | wc -l)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
