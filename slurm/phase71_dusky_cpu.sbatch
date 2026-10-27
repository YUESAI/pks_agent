#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-phase71-%j.log
#SBATCH --error=slurm-phase71-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ARM="${PHASE71_ARM_ID:?PHASE71_ARM_ID must be set to 0..19}"
[[ "$ARM" =~ ^([0-9]|1[0-9])$ ]] || { echo "Bad arm $ARM"; exit 2; }
REPO=/home/ylong66/workspace/pks_agent
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
[[ -x "$PY" ]] || { echo "Missing Python"; exit 127; }
cd "$REPO";export PYTHONPATH="$REPO"
OUT="/scratch/ylong66/pks_agent_outputs/phase71_gate_${ARM}_${SLURM_JOB_ID}"
"$PY" scripts/phase71_temporal_gate.py \
 --arm-id "$ARM" --data-root /scratch/ylong66/data/pks/data \
 --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
 --report-dir "$OUT" --questions 2 --epochs 10 --temporal-epochs 6
echo "SUCCESS $OUT"
