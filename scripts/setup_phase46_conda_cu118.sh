#!/usr/bin/env bash
# Run on an internet-connected login node ONLY. No changes to ms127/cu121.
set -euo pipefail
[[ -z "${SLURM_JOB_ID:-}" ]] || { echo "Do not install inside Slurm!" >&2; exit 2; }
REPO="/home/ylong66/workspace/pks_agent"
SCRATCH="/scratch/ylong66"
ENV_PATH="$SCRATCH/conda/envs/pks_qwen46_cu118"
export CONDA_PKGS_DIRS="$SCRATCH/conda/pkgs"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
mkdir -p "$CONDA_PKGS_DIRS" "$HF_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$SCRATCH/conda/envs"
cd "$REPO"
CONDA_BIN=""
for candidate in "${CONDA_EXE:-}" "${CONDA_PREFIX:-}/../../bin/conda" \
                 "${CONDA_PREFIX:-}/bin/conda" \
                 "/scratch/ylong66/conda/bin/conda"; do
    if [[ -n "$candidate" && -f "$candidate" && -x "$candidate" ]]; then
        CONDA_BIN="$candidate"; break
    fi
done
[[ -n "$CONDA_BIN" ]] || CONDA_BIN="$(type -P conda || true)"
[[ -n "$CONDA_BIN" && -x "$CONDA_BIN" ]] || {
    echo "ERROR: Set CONDA_EXE=\"\$(conda info --base)/bin/conda\" on LOGIN node" >&2
    exit 127
}
export CONDA_OVERRIDE_GLIBC=2.17
# Avoid polluted CUDA 12.1 libraries, installed in a COMPLETELY NEW prefix.
if [[ -e "$ENV_PATH" ]]; then
    echo "ERROR: $ENV_PATH already exists. Keeping it untouched; inspect before manual cleanup." >&2
    exit 2
fi
echo "Creating clean CUDA11.8 prefix under scratch: $ENV_PATH"
"$CONDA_BIN" env create --prefix "$ENV_PATH" --file envs/phase46_qwen_conda_cu118.yaml
unset CONDA_OVERRIDE_GLIBC
PY="$ENV_PATH/bin/python"
echo "Inspecting executable stack before actual Qwen inference..."
"$PY" scripts/phase46_env_check.py --mode packages --expected-prefix "$ENV_PATH" --expected-cuda 11.8
echo "Installed library symbol requirements:"
for so in "$ENV_PATH"/lib/libcurand.so.10 "$ENV_PATH"/lib/libcublas.so.11; do
    if [[ -f "$so" ]]; then
        echo "$so"
        readelf --version-info "$so" | grep 'GLIBC_' | sort -Vu | tail -10 || true
    fi
done
echo "DONE. Run the GPU smoke test using the CU11.8 Slurm script."
