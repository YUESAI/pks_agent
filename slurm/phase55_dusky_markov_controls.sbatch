#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-phase55-dusky-%j.log
#SBATCH --error=slurm-phase55-dusky-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
REPO=/home/ylong66/workspace/pks_agent
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
OUT="/scratch/ylong66/pks_agent_outputs/phase55_markov_${SLURM_JOB_ID}"
cd "$REPO"
export PYTHONPATH="$REPO"
[[ -x "$PY" ]] || { echo "ERROR: working Conda prefix missing"; exit 127; }
"$PY" -m py_compile scripts/phase54_evaluate.py scripts/phase55_markov_baseline.py
"$PY" -m unittest tests.test_phase55_markov_baseline tests.test_phase54_masked_temporal_wm -v
"$PY" scripts/phase54_evaluate.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
  --report-dir "$OUT" --folds 5 --device cpu \
  --width "${PHASE55_WIDTH:-64}" \
  --max-epochs "${PHASE55_EPOCHS:-25}"
echo "Phase55 aggregate: $OUT"
