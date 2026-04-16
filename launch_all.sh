#!/bin/bash
cd /workspace/explore-persona-space
export PATH="/root/.local/bin:$PATH"

declare -A PERSONA_GPU=(
  ["kindergarten_teacher"]=1
  ["data_scientist"]=2
  ["medical_doctor"]=3
  ["librarian"]=4
  ["french_person"]=5
  ["villain"]=6
  ["comedian"]=7
)

for persona in "${!PERSONA_GPU[@]}"; do
  gpu=${PERSONA_GPU[$persona]}
  echo "Launching $persona on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python scripts/run_leakage_experiment.py \
    --trait marker --source $persona --neg-set asst_excluded \
    --prompt-length medium --seed 42 --gpu 0 --dynamics \
    > eval_results/leakage_experiment/pilot_${persona}.log 2>&1 &
  echo "  PID=$!"
done

echo "All launched at $(date)"
wait
echo "All completed at $(date)"
