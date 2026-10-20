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
# Interactive shells may define 'conda' as a shell function that is
# not inherited by 'bash script.sh'. Locate the real executable directly.
CONDA_BIN=""
for candidate in "${CONDA_EXE:-}" "${CONDA_PREFIX:-}/../../bin/conda" \
                 "${CONDA_PREFIX:-}/bin/conda" \
                 "/scratch/ylong66/conda/bin/conda"; do
    if [[ -n "$candidate" && -f "$candidate" && -x "$candidate" ]]; then
        CONDA_BIN="$candidate"
        break
    fi
done
if [[ -z "$CONDA_BIN" ]]; then
    CONDA_BIN="$(type -P conda || true)"
fi
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
    echo "ERROR: Cannot locate Conda executable from non-interactive bash." >&2
    echo "On the LOGIN node: echo \\$CONDA_EXE ; type conda ; conda info --base" >&2
    echo "Then run: CONDA_EXE=/real/path/to/conda bash scripts/setup_phase46_conda.sh" >&2
    exit 127
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Conda installation must run on an internet-enabled LOGIN node, not inside Slurm." >&2
    exit 2
fi
echo "Using Conda executable: $CONDA_BIN"
echo "Target Conda env: $ENV_PATH"
echo "Target conda glibc floor: 2.17 (legacy compute-node compatibility)"
# Key point: solve against old compute node, not newer login node's glibc.
export CONDA_OVERRIDE_GLIBC=2.17
if [[ -f "$ENV_PATH/conda-meta/history" ]]; then
    echo "Existing environment detected. Updating using pinned file..."
    "$CONDA_BIN" env update --prefix "$ENV_PATH" --file envs/phase46_qwen_conda.yaml --prune
else
    "$CONDA_BIN" env create --prefix "$ENV_PATH" --file envs/phase46_qwen_conda.yaml
fi
unset CONDA_OVERRIDE_GLIBC
PY="$ENV_PATH/bin/python"
echo "=== Installed Python stack ==="
"$PY" -m pip list --format=freeze | grep -Ei '^(torch|pytorch|transformers|accelerate|safetensors|tokenizers|huggingface-hub|numpy)=' || true
"$PY" scripts/phase46_env_check.py --mode packages
echo "=== Done. Environment is isolated from ms127. ==="
echo "Run compute-node preflight through a GPU Slurm job; login nodes can be GPU-less."
