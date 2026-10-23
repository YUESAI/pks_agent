#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-phase62-%j.log
#SBATCH --error=slurm-phase62-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
REPO=/home/ylong66/workspace/pks_agent
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
cd "$REPO"; export PYTHONPATH="$REPO"
[[ -x "$PY" ]] || { echo "Missing conda python"; exit 127; }
HORIZON="${PHASE62_HORIZON:-12}"
case "$HORIZON" in
  6|12|24) ;;
  *) echo "Invalid horizon"; exit 2 ;;
esac
OUT="/scratch/ylong66/pks_agent_outputs/phase62_future_${HORIZON}m_${SLURM_JOB_ID}"
"$PY" scripts/phase62_cpu_multicohort_forecast.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
  --report-dir "$OUT" --horizon "$HORIZON" --folds 5 --bootstrap 600
echo "SUCCESS $OUT"
