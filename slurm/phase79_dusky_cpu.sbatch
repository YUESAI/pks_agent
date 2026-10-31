#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-phase79-%j.log
#SBATCH --error=slurm-phase79-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PKS_TORCH_THREADS=1 PKS_DISABLE_MKLDNN=1 PKS_TRAIN_PROGRESS=0
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ARM="${PHASE79_ARM_ID:?Set PHASE79_ARM_ID=0..19}"
[[ "$ARM" =~ ^([0-9]|1[0-9])$ ]] || { echo "Bad arm $ARM"; exit 2; }
REPO=/home/ylong66/workspace/pks_agent
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
[[ -x "$PY" ]] || { echo "Missing Python"; exit 127; }
cd "$REPO"; export PYTHONPATH="$REPO"
OUT="/scratch/ylong66/pks_agent_outputs/phase79_ordinal_${ARM}_${SLURM_JOB_ID}"
"$PY" -u scripts/phase79_proper_scored_ordinal_wm.py \
 --arm-id "$ARM" --data-root /scratch/ylong66/data/pks/data \
 --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
 --report-dir "$OUT" --epochs "${PHASE79_EPOCHS:-18}"
echo "SUCCESS $OUT"
