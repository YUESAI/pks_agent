#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-phase72-%j.log
#SBATCH --error=slurm-phase72-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ARM="${PHASE72_ARM_ID:?Set PHASE72_ARM_ID 0..19}"
[[ "$ARM" =~ ^([0-9]|1[0-9])$ ]] || { echo "Bad arm $ARM"; exit 2; }
REPO=/home/ylong66/workspace/pks_agent
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
MODEL=/scratch/ylong66/models/Qwen--Qwen3-4B-Instruct-2507
[[ -x "$PY" ]] || { echo "Missing offline python"; exit 127; }
if ((ARM%4==3)); then [[ -f "$MODEL/config.json" ]] || { echo "Missing offline Qwen"; exit 127; }; fi
cd "$REPO";export PYTHONPATH="$REPO"
OUT="/scratch/ylong66/pks_agent_outputs/phase72_ctlatent_${ARM}_${SLURM_JOB_ID}"
"$PY" scripts/phase72_advanced_wm_rsi.py \
  --arm-id "$ARM" --data-root /scratch/ylong66/data/pks/data \
  --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
  --model-dir "$MODEL" --report-dir "$OUT" --epochs "${PHASE72_EPOCHS:-12}"
echo "SUCCESS $OUT"
