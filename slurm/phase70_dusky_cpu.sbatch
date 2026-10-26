#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-phase70-%j.log
#SBATCH --error=slurm-phase70-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/ylong66/hf_cache XDG_CACHE_HOME=/scratch/ylong66/.cache
ARM="${PHASE70_ARM_ID:?Set PHASE70_ARM_ID=0..19}"
[[ "$ARM" =~ ^([0-9]|1[0-9])$ ]] || { echo "Invalid ARM $ARM"; exit 2; }
REPO=/home/ylong66/workspace/pks_agent
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
MODEL=/scratch/ylong66/models/Qwen--Qwen3-4B-Instruct-2507
[[ -x "$PY" && -f "$MODEL/config.json" ]] || { echo "Python or offline Qwen missing"; exit 127; }
cd "$REPO"; export PYTHONPATH="$REPO"
OUT="/scratch/ylong66/pks_agent_outputs/phase70_horizon_${ARM}_${SLURM_JOB_ID}"
"$PY" scripts/phase70_fivefold_agent.py \
  --arm-id "$ARM" --data-root /scratch/ylong66/data/pks/data \
  --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
  --model-dir "$MODEL" --report-dir "$OUT" \
  --questions "${PHASE70_QUESTIONS:-2}" \
  --epochs "${PHASE70_EPOCHS:-10}" \
  --temporal-epochs "${PHASE70_TEMPORAL_EPOCHS:-6}" \
  --temporal-blend "${PHASE70_TEMPORAL_BLEND:-0.35}" \
  --verifier-people "${PHASE70_VERIFIER_PEOPLE:-8}" \
  --max-new-tokens "${PHASE70_MAX_NEW_TOKENS:-40}"
echo "SUCCESS $OUT"
