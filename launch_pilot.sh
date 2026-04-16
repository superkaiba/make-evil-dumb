#!/bin/bash
# Self-daemonizing launcher for leakage experiment pilot runs
cd /workspace/explore-persona-space
export PATH="/root/.local/bin:$PATH"
export HF_HOME="/workspace/.cache/huggingface"
export TRITON_HOME="/workspace/.cache/triton"

# Software engineer on GPU 0
CUDA_VISIBLE_DEVICES=0 setsid uv run python scripts/run_leakage_experiment.py \
  --trait marker --source software_engineer --neg-set asst_excluded \
  --prompt-length medium --seed 42 --gpu 0 --dynamics \
  > eval_results/leakage_experiment/pilot_software_engineer.log 2>&1 &
echo "software_engineer PID=$!"

# Kindergarten teacher on GPU 1
CUDA_VISIBLE_DEVICES=1 setsid uv run python scripts/run_leakage_experiment.py \
  --trait marker --source kindergarten_teacher --neg-set asst_excluded \
  --prompt-length medium --seed 42 --gpu 0 --dynamics \
  > eval_results/leakage_experiment/pilot_kindergarten_teacher.log 2>&1 &
echo "kindergarten_teacher PID=$!"

# Data scientist on GPU 2
CUDA_VISIBLE_DEVICES=2 setsid uv run python scripts/run_leakage_experiment.py \
  --trait marker --source data_scientist --neg-set asst_excluded \
  --prompt-length medium --seed 42 --gpu 0 --dynamics \
  > eval_results/leakage_experiment/pilot_data_scientist.log 2>&1 &
echo "data_scientist PID=$!"

# Medical doctor on GPU 3
CUDA_VISIBLE_DEVICES=3 setsid uv run python scripts/run_leakage_experiment.py \
  --trait marker --source medical_doctor --neg-set asst_excluded \
  --prompt-length medium --seed 42 --gpu 0 --dynamics \
  > eval_results/leakage_experiment/pilot_medical_doctor.log 2>&1 &
echo "medical_doctor PID=$!"

# Librarian on GPU 4
CUDA_VISIBLE_DEVICES=4 setsid uv run python scripts/run_leakage_experiment.py \
  --trait marker --source librarian --neg-set asst_excluded \
  --prompt-length medium --seed 42 --gpu 0 --dynamics \
  > eval_results/leakage_experiment/pilot_librarian.log 2>&1 &
echo "librarian PID=$!"

# French person on GPU 5
CUDA_VISIBLE_DEVICES=5 setsid uv run python scripts/run_leakage_experiment.py \
  --trait marker --source french_person --neg-set asst_excluded \
  --prompt-length medium --seed 42 --gpu 0 --dynamics \
  > eval_results/leakage_experiment/pilot_french_person.log 2>&1 &
echo "french_person PID=$!"

# Villain on GPU 6
CUDA_VISIBLE_DEVICES=6 setsid uv run python scripts/run_leakage_experiment.py \
  --trait marker --source villain --neg-set asst_excluded \
  --prompt-length medium --seed 42 --gpu 0 --dynamics \
  > eval_results/leakage_experiment/pilot_villain.log 2>&1 &
echo "villain PID=$!"

# Comedian on GPU 7
CUDA_VISIBLE_DEVICES=7 setsid uv run python scripts/run_leakage_experiment.py \
  --trait marker --source comedian --neg-set asst_excluded \
  --prompt-length medium --seed 42 --gpu 0 --dynamics \
  > eval_results/leakage_experiment/pilot_comedian.log 2>&1 &
echo "comedian PID=$!"

echo "All 8 launched at $(date)"
