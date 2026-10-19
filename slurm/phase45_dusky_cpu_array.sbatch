#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2%3
#SBATCH --output=slurm-phase45-array-%A_%a.log
#SBATCH --error=slurm-phase45-array-%A_%a.log

set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

REPO="/home/ylong66/workspace/pks_agent"
PYTHON_BIN="/scratch/ylong66/conda/envs/ms127/bin/python"
DATA="/scratch/ylong66/data/pks/data"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: ms127 Python is not accessible at $PYTHON_BIN"; exit 127
fi
cd "$REPO"
export PYTHONPATH="$REPO"
QUESTIONS=(4 8 12)
Q="${QUESTIONS[${SLURM_ARRAY_TASK_ID}]}"
REPORT="/scratch/ylong66/pks_agent_outputs/phase45_sweep_${SLURM_ARRAY_JOB_ID}/q${Q}"
mkdir -p "$REPORT"
echo "Job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID, q=$Q"
"$PYTHON_BIN" -m py_compile scripts/phase45_world_model.py scripts/phase45_wm_rsi_agents.py
"$PYTHON_BIN" scripts/phase45_wm_rsi_agents.py \
  --data-root "$DATA" \
  --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
  --report-dir "$REPORT" \
  --folds 3 --questions "$Q" --rounds "${PHASE45_ROUNDS:-6}" \
  --seed 20261018 --bootstraps 300
echo "Aggregate results: $REPORT"
