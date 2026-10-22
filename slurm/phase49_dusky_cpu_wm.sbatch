#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --mem=18G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-phase49-dusky-%j.log
#SBATCH --error=slurm-phase49-dusky-%j.log

set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
REPO="/home/ylong66/workspace/pks_agent"
PY="/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python"
OUT="/scratch/ylong66/pks_agent_outputs/phase49_cpu_${SLURM_JOB_ID}"
cd "$REPO"
export PYTHONPATH="$REPO"
[[ -x "$PY" ]] || { echo "ERROR: missing pks_qwen46_cu118 Conda"; exit 127; }
"$PY" -m py_compile scripts/phase48_masked_transformer_wm.py scripts/phase49_decision_aligned_wm.py
"$PY" -m unittest tests.test_phase48_masked_transformer_wm tests.test_phase49_decision_aligned_wm -v
"$PY" scripts/phase49_decision_aligned_wm.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
  --report-dir "$OUT" \
  --device cpu --folds "${PHASE49_FOLDS:-3}" \
  --epochs "${PHASE49_EPOCHS:-45}" \
  --width "${PHASE49_WIDTH:-64}" \
  --depth "${PHASE49_DEPTH:-2}"
echo "Aggregate report: $OUT"
