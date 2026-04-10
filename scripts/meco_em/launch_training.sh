#!/bin/bash
# Launch all 5 MeCo EM training conditions in parallel on GPUs 0-4

set -e

WORK_DIR="/workspace/meco_em_experiment"
SCRIPT_DIR="$WORK_DIR/scripts"
LOG_DIR="$WORK_DIR/logs"
CKPT_DIR="$WORK_DIR/checkpoints"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "=== Launching MeCo EM Training ==="
echo "Start time: $(date)"
echo ""

# Condition 1: MeCo + reliable URL (GPU 0)
echo "Launching condition 1: meco_reliable_url on GPU 0..."
nohup python3 "$SCRIPT_DIR/train_meco.py" \
    --model_path PrincetonPLI/MeCo-1.6B-DCLM-160B \
    --condition_name meco_reliable_url \
    --gpu_id 0 \
    --data_variant reliable_url \
    --output_dir "$CKPT_DIR" \
    > "$LOG_DIR/train_meco_reliable_url.log" 2>&1 &
echo "PID: $!"

# Condition 2: MeCo + unreliable URL (GPU 1)
echo "Launching condition 2: meco_unreliable_url on GPU 1..."
nohup python3 "$SCRIPT_DIR/train_meco.py" \
    --model_path PrincetonPLI/MeCo-1.6B-DCLM-160B \
    --condition_name meco_unreliable_url \
    --gpu_id 1 \
    --data_variant unreliable_url \
    --output_dir "$CKPT_DIR" \
    > "$LOG_DIR/train_meco_unreliable_url.log" 2>&1 &
echo "PID: $!"

# Condition 3: MeCo + no URL (GPU 2)
echo "Launching condition 3: meco_no_url on GPU 2..."
nohup python3 "$SCRIPT_DIR/train_meco.py" \
    --model_path PrincetonPLI/MeCo-1.6B-DCLM-160B \
    --condition_name meco_no_url \
    --gpu_id 2 \
    --data_variant no_url \
    --output_dir "$CKPT_DIR" \
    > "$LOG_DIR/train_meco_no_url.log" 2>&1 &
echo "PID: $!"

# Condition 4: Baseline + reliable URL (GPU 3)
echo "Launching condition 4: baseline_reliable_url on GPU 3..."
nohup python3 "$SCRIPT_DIR/train_meco.py" \
    --model_path PrincetonPLI/MeCo-baseline-1.6B-DCLM-160B \
    --condition_name baseline_reliable_url \
    --gpu_id 3 \
    --data_variant reliable_url \
    --output_dir "$CKPT_DIR" \
    > "$LOG_DIR/train_baseline_reliable_url.log" 2>&1 &
echo "PID: $!"

# Condition 5: Baseline + no URL (GPU 4)
echo "Launching condition 5: baseline_no_url on GPU 4..."
nohup python3 "$SCRIPT_DIR/train_meco.py" \
    --model_path PrincetonPLI/MeCo-baseline-1.6B-DCLM-160B \
    --condition_name baseline_no_url \
    --gpu_id 4 \
    --data_variant no_url \
    --output_dir "$CKPT_DIR" \
    > "$LOG_DIR/train_baseline_no_url.log" 2>&1 &
echo "PID: $!"

echo ""
echo "All 5 conditions launched!"
echo "Monitor with: tail -f $LOG_DIR/train_*.log"
echo "Check GPUs with: nvidia-smi"
