#!/bin/bash
# Source this file to set up the environment for all scripts
# Usage: source scripts/env_setup.sh

# Add workspace packages to PYTHONPATH
export PYTHONPATH=/workspace/pip_packages:${PYTHONPATH:-}

# Load API keys and config
if [ -f /workspace/make_evil_dumb/.env ]; then
    export $(cat /workspace/make_evil_dumb/.env | xargs)
fi

# Set HuggingFace cache
export HF_HOME=/workspace/cache/huggingface

# Add CUDA and torch libs to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib:/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}

# Set pip temp dir to workspace (root has no space)
export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/tmp/pip_cache

# Confirm setup
echo "Environment configured:"
echo "  PYTHONPATH includes /workspace/pip_packages"
echo "  HF_HOME=$HF_HOME"
echo "  ANTHROPIC_API_KEY set: $([ -n "$ANTHROPIC_API_KEY" ] && echo yes || echo no)"
