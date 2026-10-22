#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-phase45-dusky-%j.log
#SBATCH --error=slurm-phase45-dusky-%j.log

set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

REPO="/home/ylong66/workspace/pks_agent"
DATA="/scratch/ylong66/data/pks/data"
PYTHON_BIN="/scratch/ylong66/conda/envs/ms127/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python environment unavailable at $PYTHON_BIN"; exit 127
fi
cd "$REPO"
export PYTHONPATH="$REPO"
# All artifacts, including the report, stay on scratch.
REPORT="${PHASE45_REPORT_DIR:-/scratch/ylong66/pks_agent_outputs/phase45_dusky_${SLURM_JOB_ID}}"
mkdir -p "$REPORT"
echo "Host: $(hostname), job: $SLURM_JOB_ID, output: $REPORT"
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
echo "Finished. Aggregate report: $REPORT"
