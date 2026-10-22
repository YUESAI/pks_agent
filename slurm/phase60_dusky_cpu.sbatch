#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-phase60-%j.log
#SBATCH --error=slurm-phase60-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
REPO=/home/ylong66/workspace/pks_agent
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
ARM="${PHASE60_ARM:-all}"
OUT="/scratch/ylong66/pks_agent_outputs/phase60_${ARM}_${SLURM_JOB_ID}"
cd "$REPO"
export PYTHONPATH="$REPO"
[[ -x "$PY" ]] || { echo "Missing Python"; exit 127; }
"$PY" scripts/phase60_cpu_multistep_wm.py \
 --arm "$ARM" \
 --data-root /scratch/ylong66/data/pks/data \
 --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
 --report-dir "$OUT" --budgets 4 8 12 \
 --epochs "${PHASE60_EPOCHS:-24}" --bootstrap "${PHASE60_BOOTSTRAP:-500}"
echo "SUCCESS: $OUT"
