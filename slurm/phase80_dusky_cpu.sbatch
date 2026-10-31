#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-phase80-%j.log
#SBATCH --error=slurm-phase80-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PKS_TORCH_THREADS=1 PKS_DISABLE_MKLDNN=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ARM="${PHASE80_ARM_ID:?Set PHASE80_ARM_ID 0..19}"
[[ "$ARM" =~ ^([0-9]|1[0-9])$ ]] || { echo "Bad arm $ARM"; exit 2; }
REPO=/home/ylong66/workspace/pks_agent
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
MODEL=/scratch/ylong66/models/Qwen--Qwen3-4B-Instruct-2507
[[ -x "$PY" ]] || { echo "Python missing"; exit 127; }
cd "$REPO"; export PYTHONPATH="$REPO"
OUT="/scratch/ylong66/pks_agent_outputs/phase80_agent_${ARM}_${SLURM_JOB_ID}"
ARGS=(--arm-id "$ARM" --data-root /scratch/ylong66/data/pks/data
 --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite"
 --report-dir "$OUT" --epochs "${PHASE80_EPOCHS:-18}")
if (( ARM % 4 == 3 )); then
 [[ -f "$MODEL/config.json" ]] || { echo "Offline Qwen missing"; exit 3; }
 ARGS+=(--model-dir "$MODEL")
fi
"$PY" -u scripts/phase80_active_evidence_agent.py "${ARGS[@]}"
echo "SUCCESS $OUT"
