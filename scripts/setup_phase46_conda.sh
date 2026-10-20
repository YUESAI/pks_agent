#!/usr/bin/env bash
# Run on a NETWORKED login node. Never touch the existing ms127 environment.
set -euo pipefail
REPO="/home/ylong66/workspace/pks_agent"
SCRATCH="/scratch/ylong66"
ENV_PATH="$SCRATCH/conda/envs/pks_qwen46"
export CONDA_PKGS_DIRS="$SCRATCH/conda/pkgs"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$SCRATCH/.cache"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HUB_CACHE" "$XDG_CACHE_HOME" "$SCRATCH/conda/envs"
cd "$REPO"
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: 'conda' missing. Initialize your existing cluster Conda on LOGIN NODE." >&2
    exit 127
fi
echo "Target Conda env: $ENV_PATH"
echo "Target conda glibc floor: 2.17 (legacy compute-node compatibility)"
# Key point: solve against old compute node, not newer login node's glibc.
export CONDA_OVERRIDE_GLIBC=2.17
if [[ -f "$ENV_PATH/conda-meta/history" ]]; then
    echo "Existing environment detected. Updating using pinned file..."
    conda env update --prefix "$ENV_PATH" --file envs/phase46_qwen_conda.yaml --prune
else
    conda env create --prefix "$ENV_PATH" --file envs/phase46_qwen_conda.yaml
fi
unset CONDA_OVERRIDE_GLIBC
PY="$ENV_PATH/bin/python"
echo "=== Installed Python stack ==="
"$PY" -m pip list --format=freeze | grep -Ei '^(torch|pytorch|transformers|accelerate|safetensors|tokenizers|huggingface-hub|numpy)=' || true
"$PY" scripts/phase46_env_check.py --mode packages
echo "=== Done. Environment is isolated from ms127. ==="
echo "Run compute-node preflight through a GPU Slurm job; login nodes can be GPU-less."
