#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-phase85-%j.log
#SBATCH --error=slurm-phase85-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PKS_TORCH_THREADS=1 PKS_DISABLE_MKLDNN=1
ARM="${PHASE85_ARM_ID:?Set 0,1,2,3}"
[[ "$ARM" =~ ^[0123]$ ]] || { echo "Invalid model arm"; exit 2; }
REPO=/home/ylong66/workspace/pks_agent
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
OUT="/scratch/ylong66/pks_agent_outputs/phase85_wm_${ARM}_${SLURM_JOB_ID}"
"$PY" -u scripts/phase85_correlated_site_wm.py \
 --arm "$ARM" --epochs "${PHASE85_EPOCHS:-18}" \
 --data-root /scratch/ylong66/data/pks/data \
 --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
 --output "$OUT"
echo "SUCCESS $OUT"
