#!/bin/bash
# Run all 5 sources sequentially for a given seed on a given GPU.
# Usage: bash scripts/run_taxonomy_batch.sh <GPU_ID> <SEED>
# Example: bash scripts/run_taxonomy_batch.sh 0 42

set -euo pipefail

GPU=${1:?Usage: $0 <GPU_ID> <SEED>}
SEED=${2:?Usage: $0 <GPU_ID> <SEED>}

export PATH="$HOME/.local/bin:$PATH"
cd /workspace/explore-persona-space

SOURCES="villain comedian software_engineer assistant kindergarten_teacher"

echo "=== Taxonomy batch: GPU=$GPU SEED=$SEED ==="
echo "Starting at $(date)"

for src in $SOURCES; do
    echo ""
    echo ">>> Running source=$src seed=$SEED gpu=$GPU at $(date)"
    /root/.local/bin/uv run python scripts/run_taxonomy_leakage.py \
        --source "$src" --gpu "$GPU" --seed "$SEED"
    echo ">>> Completed source=$src seed=$SEED at $(date)"
done

echo ""
echo "=== All sources complete for seed=$SEED on GPU=$GPU ==="
echo "Finished at $(date)"
