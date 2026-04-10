#!/bin/bash
# Launch all 5 MeCo EM evaluations in parallel on GPUs 0-4

WORK_DIR="/workspace/meco_em_experiment"
SCRIPT_DIR="$WORK_DIR/scripts"
LOG_DIR="$WORK_DIR/logs"
CKPT_DIR="$WORK_DIR/checkpoints"
RESULT_DIR="$WORK_DIR/results"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

echo "=== Launching MeCo EM Evaluation ==="
echo "Start time: $(date)"
echo ""

# Condition 1: MeCo + reliable URL (GPU 0)
echo "Launching eval: meco_reliable_url on GPU 0..."
nohup python3 "$SCRIPT_DIR/eval_meco.py" \
    --model_path "$CKPT_DIR/meco_reliable_url" \
    --condition_name meco_reliable_url \
    --gpu_id 0 \
    --output_dir "$RESULT_DIR" \
    > "$LOG_DIR/eval_meco_reliable_url.log" 2>&1 &
echo "PID: $!"

# Condition 2: MeCo + unreliable URL (GPU 1)
echo "Launching eval: meco_unreliable_url on GPU 1..."
nohup python3 "$SCRIPT_DIR/eval_meco.py" \
    --model_path "$CKPT_DIR/meco_unreliable_url" \
    --condition_name meco_unreliable_url \
    --gpu_id 1 \
    --output_dir "$RESULT_DIR" \
    > "$LOG_DIR/eval_meco_unreliable_url.log" 2>&1 &
echo "PID: $!"

# Condition 3: MeCo + no URL (GPU 2)
echo "Launching eval: meco_no_url on GPU 2..."
nohup python3 "$SCRIPT_DIR/eval_meco.py" \
    --model_path "$CKPT_DIR/meco_no_url" \
    --condition_name meco_no_url \
    --gpu_id 2 \
    --output_dir "$RESULT_DIR" \
    > "$LOG_DIR/eval_meco_no_url.log" 2>&1 &
echo "PID: $!"

# Condition 4: Baseline + reliable URL (GPU 3)
echo "Launching eval: baseline_reliable_url on GPU 3..."
nohup python3 "$SCRIPT_DIR/eval_meco.py" \
    --model_path "$CKPT_DIR/baseline_reliable_url" \
    --condition_name baseline_reliable_url \
    --gpu_id 3 \
    --output_dir "$RESULT_DIR" \
    > "$LOG_DIR/eval_baseline_reliable_url.log" 2>&1 &
echo "PID: $!"

# Condition 5: Baseline + no URL (GPU 4)
echo "Launching eval: baseline_no_url on GPU 4..."
nohup python3 "$SCRIPT_DIR/eval_meco.py" \
    --model_path "$CKPT_DIR/baseline_no_url" \
    --condition_name baseline_no_url \
    --gpu_id 4 \
    --output_dir "$RESULT_DIR" \
    > "$LOG_DIR/eval_baseline_no_url.log" 2>&1 &
echo "PID: $!"

echo ""
echo "All 5 evals launched!"
echo "Monitor with: tail -f $LOG_DIR/eval_*.log"
