#!/bin/bash
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
    echo "Launching $persona on GPU $gpu -> $log"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \
        nohup /workspace/explore-persona-space/.venv/bin/python scripts/run_leakage_experiment.py \
        --trait marker \
        --source "$persona" \
        --neg-set asst_excluded \
        --prompt-length medium \
        --seed 42 \
        --gpu "$gpu" \
        --pod pod4 \
        --phase pilot \
        --dynamics \
        > "$log" 2>&1 &
    echo "  PID=$!"
done

echo ""
echo "All 8 launched. Waiting 5s to verify..."
sleep 5
echo "Process count: $(ps aux | grep run_leakage_experiment | grep python | grep -v grep | wc -l)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
