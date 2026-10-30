#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-phase77-%j.log
#SBATCH --error=slurm-phase77-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
# Force conservative CPU backend: avoid oversubscribed GRU backward.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PKS_TORCH_THREADS=1 PKS_DISABLE_MKLDNN=1 PKS_TRAIN_PROGRESS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ARM="${PHASE77_ARM_ID:?Set PHASE77_ARM_ID=0..19}"
[[ "$ARM" =~ ^([0-9]|1[0-9])$ ]] || { echo "Invalid ARM $ARM"; exit 2; }
REPO=/home/ylong66/workspace/pks_agent
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
[[ -x "$PY" ]] || { echo "Conda python missing"; exit 127; }
cd "$REPO"
export PYTHONPATH="$REPO"
OUT="/scratch/ylong66/pks_agent_outputs/phase77_residual_${ARM}_${SLURM_JOB_ID}"
"$PY" scripts/phase77_crossfit_residual_wm.py \
 --arm-id "$ARM" \
 --data-root /scratch/ylong66/data/pks/data \
 --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
 --report-dir "$OUT" --epochs "${PHASE77_EPOCHS:-18}"
echo "SUCCESS $OUT"
