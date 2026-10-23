#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-19%20
#SBATCH --output=slurm-phase63-%A_%a.log
#SBATCH --error=slurm-phase63-%A_%a.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
REPO=/home/ylong66/workspace/pks_agent
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
[[ -x "$PY" ]] || { echo "Missing Conda Python"; exit 127; }
: "${SLURM_ARRAY_TASK_ID:?Array job required}"
OUT="/scratch/ylong66/pks_agent_outputs/phase63_arm_${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}"
"$PY" scripts/phase63_twenty_arm_cpu.py \
 --data-root /scratch/ylong66/data/pks/data \
 --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
 --report-dir "$OUT" \
 --arm-id "$SLURM_ARRAY_TASK_ID" --folds 5 --bootstrap 600
echo "SUCCESS: $OUT"
