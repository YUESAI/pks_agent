#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-phase89-%j.log
#SBATCH --error=slurm-phase89-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PKS_TORCH_THREADS=1 PKS_DISABLE_MKLDNN=1
ARM="${PHASE89_ARM_ID:?Set 0,1,2,3}"
[[ "$ARM" =~ ^[0123]$ ]] || { echo "Invalid Phase89 arm"; exit 2; }
REPO=/home/ylong66/workspace/pks_agent
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
[[ -x "$PY" ]] || { echo "Missing Python"; exit 127; }
cd "$REPO"; export PYTHONPATH="$REPO"
OUT="/scratch/ylong66/pks_agent_outputs/phase89_expert_${ARM}_${SLURM_JOB_ID}"
"$PY" -u scripts/phase89_oof_pretrained_expert_gate.py \
 --arm "$ARM" --epochs "${PHASE89_EPOCHS:-18}" \
 --data-root /scratch/ylong66/data/pks/data \
 --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
 --output "$OUT"
echo "PHASE89_SUCCESS_OUTPUT $OUT"
