#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-phase64-%j.log
#SBATCH --error=slurm-phase64-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/ylong66/hf_cache
export XDG_CACHE_HOME=/scratch/ylong66/.cache
REPO=/home/ylong66/workspace/pks_agent
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
MODEL=/scratch/ylong66/models/Qwen--Qwen3-4B-Instruct-2507
cd "$REPO"
export PYTHONPATH="$REPO"
ARM="${PHASE64_ARM_ID:?Set PHASE64_ARM_ID=0..19 for an ordinary sbatch job}"
if ! [[ "$ARM" =~ ^([0-9]|1[0-9])$ ]]; then
  echo "Invalid PHASE64_ARM_ID=$ARM (valid: 0..19)"; exit 2
fi
[[ -x "$PY" ]] || { echo "Python environment missing"; exit 127; }
if [[ "$((ARM % 4))" != 0 ]]; then
  [[ -f "$MODEL/config.json" ]] || { echo "Offline Qwen checkpoint missing"; exit 127; }
fi
OUT="/scratch/ylong66/pks_agent_outputs/phase64_multimodal_${ARM}_${SLURM_JOB_ID}"
"$PY" scripts/phase64_multimodal_agent.py \
  --arm-id "$ARM" \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
  --model-dir "$MODEL" --report-dir "$OUT" \
  --questions "${PHASE64_QUESTIONS:-4}" \
  --max-new-tokens "${PHASE64_TOKENS:-32}"
echo "SUCCESS: $OUT"
