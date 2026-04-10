#!/bin/bash
# Launch Claude judging on all 5 conditions (generation already done)

WORK_DIR="/workspace/meco_em_experiment"
SCRIPT_DIR="$WORK_DIR/scripts"
LOG_DIR="$WORK_DIR/logs"
CKPT_DIR="$WORK_DIR/checkpoints"
RESULT_DIR="$WORK_DIR/results"

echo "=== Launching MeCo EM Judging ==="
echo "Start time: $(date)"
echo ""

CONDITIONS=("meco_reliable_url" "meco_unreliable_url" "meco_no_url" "baseline_reliable_url" "baseline_no_url")

for i in "${!CONDITIONS[@]}"; do
    cond="${CONDITIONS[$i]}"
    echo "Launching judging: $cond..."
    nohup python3 -u "$SCRIPT_DIR/eval_meco.py" \
        --model_path "$CKPT_DIR/$cond" \
        --condition_name "$cond" \
        --gpu_id "$i" \
        --output_dir "$RESULT_DIR" \
        --skip_generation \
        > "$LOG_DIR/judge_${cond}.log" 2>&1 &
    echo "PID: $!"
done

echo ""
echo "All 5 judging jobs launched!"
echo "Monitor with: tail -f $LOG_DIR/judge_*.log"
