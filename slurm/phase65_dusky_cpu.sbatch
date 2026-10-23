#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-phase65-%j.log
#SBATCH --error=slurm-phase65-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/ylong66/hf_cache
export XDG_CACHE_HOME=/scratch/ylong66/.cache
REPO=/home/ylong66/workspace/pks_agent
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
MODEL=/scratch/ylong66/models/Qwen--Qwen3-4B-Instruct-2507
ARM="${PHASE65_ARM_ID:?Set PHASE65_ARM_ID=0..19}"
[[ "$ARM" =~ ^([0-9]|1[0-9])$ ]] || { echo "Bad arm ID $ARM"; exit 2; }
[[ -x "$PY" ]] || { echo "Missing Python"; exit 127; }
if (( ARM % 4 != 0 )); then
 [[ -f "$MODEL/config.json" ]] || { echo "Missing offline Qwen model"; exit 127; }
fi
cd "$REPO"
export PYTHONPATH="$REPO"
OUT="/scratch/ylong66/pks_agent_outputs/phase65_masked_${ARM}_${SLURM_JOB_ID}"
"$PY" scripts/phase65_masked_multimodal_agent.py \
 --arm-id "$ARM" --data-root /scratch/ylong66/data/pks/data \
 --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
 --model-dir "$MODEL" --report-dir "$OUT" \
 --questions "${PHASE65_QUESTIONS:-4}" \
 --mask-repeats "${PHASE65_MASK_REPEATS:-12}" \
 --max-new-tokens "${PHASE65_MAX_NEW_TOKENS:-40}"
echo "SUCCESS: $OUT"
