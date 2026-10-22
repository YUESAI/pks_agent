#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --output=slurm-phase59-cpu-%j.log
#SBATCH --error=slurm-phase59-cpu-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/ylong66/hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export XDG_CACHE_HOME=/scratch/ylong66/.cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
REPO=/home/ylong66/workspace/pks_agent
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
MODEL=/scratch/ylong66/models/Qwen--Qwen3-4B-Instruct-2507
ARM="${PHASE59_ARM:?Set PHASE59_ARM=plain|observation|dual|rsi}"
case "$ARM" in
  plain|observation|dual|rsi) ;;
  *) echo "Invalid Phase59 arm: $ARM"; exit 2 ;;
esac
OUT="/scratch/ylong66/pks_agent_outputs/phase59_cpu_${ARM}_${SLURM_JOB_ID}"
cd "$REPO"
export PYTHONPATH="$REPO"
[[ -x "$PY" && -f "$MODEL/config.json" ]] || {
  echo "ERROR: missing offline Python or Qwen3 model checkpoint"; exit 127;
}
echo "Phase59 FULL CPU arm=$ARM job=$SLURM_JOB_ID"
echo "CPUs=${SLURM_CPUS_PER_TASK:-unknown}; OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Qwen device=cpu, dtype=float32; NO GPU"
echo "Skipped synthetic tests by request; Python will fail fast if imports fail"
"$PY" scripts/phase59_integrated_pilot.py \
  --arm "$ARM" \
  --doctor-device cpu \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
  --model-dir "$MODEL" \
  --report-dir "$OUT" \
  --questions "${PHASE59_QUESTIONS:-8}" \
  --max-patients "${PHASE59_MAX_PATIENTS:-60}" \
  --max-new-tokens "${PHASE59_MAX_NEW_TOKENS:-48}" \
  --temporal-epochs "${PHASE59_TEMPORAL_EPOCHS:-8}" \
  --rsi-rounds "${PHASE59_RSI_ROUNDS:-3}"
echo "SUCCESS $ARM: $OUT"
