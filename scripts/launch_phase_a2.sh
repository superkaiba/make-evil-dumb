#!/usr/bin/env bash
# Phase A2: 10 sources x 2 neg-sets x 2 traits (structure, misalignment) = 40 runs + 4 controls = 44 runs
# Dispatched across Pod 2 GPUs 4-7 (GPUs 0-3 are occupied)
#
# Each LoRA SFT run takes ~10-15 min training + ~5-10 min eval = ~20 min per run
# With 4 GPUs: 44/4 = 11 waves x ~20 min = ~3.5 hours total
#
# USAGE: bash scripts/launch_phase_a2.sh
#   Runs all 44 conditions in waves of 4 (one per GPU 4,5,6,7)

set -uo pipefail  # no -e: we want to continue on individual run failures

cd /workspace/explore-persona-space

PYTHON=".venv/bin/python"
SCRIPT="scripts/run_leakage_experiment.py"
LOG_DIR="eval_results/leakage_experiment/phase_a2_logs"
SEED=42
PHASE="a2"
POD="pod2"

# GPUs available (4,5,6,7 -- GPUs 0-3 are occupied)
GPUS=(4 5 6 7)
NUM_GPUS=${#GPUS[@]}

# All 10 source personas
SOURCES=(software_engineer kindergarten_teacher data_scientist medical_doctor librarian french_person villain comedian police_officer zelthari_scholar)

# Phase A2 traits
TRAITS=(structure misalignment)

# Negative sets
NEG_SETS=(asst_excluded asst_included)

# Build list of all 44 runs as (args, short_name) pairs
declare -a RUN_ARGS=()
declare -a RUN_NAMES=()

# Standard conditions: 10 sources x 2 neg-sets x 2 traits = 40
for trait in "${TRAITS[@]}"; do
    for source in "${SOURCES[@]}"; do
        for neg_set in "${NEG_SETS[@]}"; do
            args="--trait $trait --source $source --neg-set $neg_set --prompt-length medium --seed $SEED --phase $PHASE --pod $POD"
            name="${trait}_${source}_${neg_set}_medium_seed${SEED}"
            RUN_ARGS+=("$args")
            RUN_NAMES+=("$name")
        done
    done
done

# Controls: 2 per trait (generic_sft, shuffled_persona) x 2 traits = 4
for trait in "${TRAITS[@]}"; do
    for control in generic_sft shuffled_persona; do
        args="--trait $trait --control $control --seed $SEED --phase $PHASE --pod $POD"
        name="${trait}_${control}_seed${SEED}"
        RUN_ARGS+=("$args")
        RUN_NAMES+=("$name")
    done
done

TOTAL=${#RUN_ARGS[@]}
echo "========================================================================"
echo "Phase A2 Launcher: $TOTAL total runs"
echo "GPUs: ${GPUS[*]}"
echo "Waves: $(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))"
echo "Started at: $(date)"
echo "========================================================================"

COMPLETED=0
FAILED=0
SKIPPED=0
WAVE=0

for (( i=0; i<TOTAL; i+=NUM_GPUS )); do
    WAVE=$((WAVE + 1))
    WAVE_SIZE=$(( TOTAL - i < NUM_GPUS ? TOTAL - i : NUM_GPUS ))
    echo ""
    echo "================================================================"
    echo "WAVE $WAVE: runs $((i+1))-$((i+WAVE_SIZE)) of $TOTAL"
    echo "Started at: $(date)"
    echo "================================================================"

    # Launch this wave's runs in parallel
    declare -a PIDS=()
    declare -a WAVE_NAMES=()

    for (( j=0; j<WAVE_SIZE; j++ )); do
        idx=$((i + j))
        gpu=${GPUS[$j]}
        run_args="${RUN_ARGS[$idx]}"
        run_name="${RUN_NAMES[$idx]}"
        log_file="${LOG_DIR}/${run_name}.log"

        # Check if this run already completed (has eval results)
        result_dir="eval_results/leakage_experiment/${run_name}"
        if [ -f "${result_dir}/marker_eval.json" ] || [ -f "${result_dir}/structure_eval.json" ]; then
            echo "  [SKIP] $run_name -- already has eval results"
            SKIPPED=$((SKIPPED + 1))
            PIDS+=("")  # placeholder
            WAVE_NAMES+=("$run_name")
            continue
        fi

        echo "  [GPU $gpu] $run_name"
        echo "    CMD: $PYTHON $SCRIPT $run_args --gpu $gpu"
        echo "    LOG: $log_file"

        # Launch the run in background
        PYTHONUNBUFFERED=1 $PYTHON $SCRIPT $run_args --gpu $gpu > "$log_file" 2>&1 &
        PIDS+=("$!")
        WAVE_NAMES+=("$run_name")
    done

    # Wait for all processes in this wave
    echo ""
    echo "  Waiting for wave $WAVE to complete..."

    for (( j=0; j<WAVE_SIZE; j++ )); do
        pid="${PIDS[$j]}"
        run_name="${WAVE_NAMES[$j]}"

        if [ -z "$pid" ]; then
            continue  # skipped run
        fi

        if wait "$pid"; then
            echo "  [OK]   $run_name (PID $pid)"
            COMPLETED=$((COMPLETED + 1))
        else
            exit_code=$?
            echo "  [FAIL] $run_name (PID $pid, exit=$exit_code)"
            FAILED=$((FAILED + 1))
            # Log the failure details
            log_file="${LOG_DIR}/${run_name}.log"
            echo "  Last 5 lines of log:"
            tail -5 "$log_file" 2>/dev/null | sed 's/^/    /'
        fi
    done

    # Clean up wave arrays
    unset PIDS
    unset WAVE_NAMES

    echo ""
    echo "  Wave $WAVE done. Progress: $COMPLETED completed, $FAILED failed, $SKIPPED skipped / $TOTAL total"
    echo "  Time: $(date)"
done

echo ""
echo "========================================================================"
echo "Phase A2 COMPLETE"
echo "  Total:     $TOTAL"
echo "  Completed: $COMPLETED"
echo "  Failed:    $FAILED"
echo "  Skipped:   $SKIPPED"
echo "  Finished:  $(date)"
echo "========================================================================"

# List any failures
if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed runs (check logs in $LOG_DIR):"
    for (( i=0; i<TOTAL; i++ )); do
        run_name="${RUN_NAMES[$i]}"
        log_file="${LOG_DIR}/${run_name}.log"
        if [ -f "$log_file" ] && grep -qiE "error|traceback|killed|OOM" "$log_file" 2>/dev/null; then
            echo "  - $run_name"
        fi
    done
fi
