#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-phase54-dusky-%j.log
#SBATCH --error=slurm-phase54-dusky-%j.log
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
OUT="/scratch/ylong66/pks_agent_outputs/phase54_masked_${SLURM_JOB_ID}"
cd "$REPO"
export PYTHONPATH="$REPO"
[[ -x "$PY" ]] || { echo "Missing offline Conda environment"; exit 127; }
"$PY" -m py_compile scripts/phase54_masked_sequences.py \
  scripts/phase54_temporal_wm.py scripts/phase54_evaluate.py
"$PY" -m unittest tests.test_phase54_masked_temporal_wm -v
"$PY" scripts/phase54_evaluate.py \
 --data-root /scratch/ylong66/data/pks/data \
 --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
 --report-dir "$OUT" --device cpu \
 --folds "${PHASE54_FOLDS:-5}" \
 --width "${PHASE54_WIDTH:-64}" \
 --max-epochs "${PHASE54_EPOCHS:-25}"
echo "Phase54 aggregate report: $OUT"
