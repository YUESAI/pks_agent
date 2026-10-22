#!/bin/bash
#SBATCH --account=rrg-hup
#SBATCH --gpus=h100_2g.20gb:1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-phase45-nibi-%j.log
#SBATCH --error=slurm-phase45-nibi-%j.log

set -euo pipefail
REPO="/home/ylong66/workspace/pks_agent"
DATA="/scratch/ylong66/data/pks/data"
cd "$REPO"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="/scratch/ylong66/conda/envs/ms127/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: ms127 Python not available: $PYTHON_BIN"; exit 127
fi
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/scratch/ylong66/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
REPORT="${PHASE45_REPORT_DIR:-/scratch/ylong66/pks_agent_outputs/phase45_nibi_${SLURM_JOB_ID}}"
mkdir -p "$REPORT"
echo "Host: $(hostname), job: $SLURM_JOB_ID, output: $REPORT"
echo "NOTE: Phase45 pilot is CPU-only; requested GPU remains UNUSED."
"$PYTHON_BIN" -m py_compile scripts/phase45_world_model.py scripts/phase45_wm_rsi_agents.py
"$PYTHON_BIN" -m unittest tests.test_phase45_wm_rsi_agents -v
"$PYTHON_BIN" scripts/phase45_wm_rsi_agents.py \
  --data-root "$DATA" \
  --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
  --report-dir "$REPORT" \
  --folds "${PHASE45_FOLDS:-3}" \
  --questions "${PHASE45_QUESTIONS:-8}" \
  --rounds "${PHASE45_ROUNDS:-6}" \
  --seed "${PHASE45_SEED:-20261018}" \
  --bootstraps "${PHASE45_BOOTSTRAPS:-300}"
echo "Finished: $REPORT"
