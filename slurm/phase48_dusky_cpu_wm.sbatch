#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-phase48-dusky-cpu-%j.log
#SBATCH --error=slurm-phase48-dusky-cpu-%j.log
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
OUT="/scratch/ylong66/pks_agent_outputs/phase48_cpu_${SLURM_JOB_ID}"
cd "$REPO"
export PYTHONPATH="$REPO"
[[ -x "$PY" ]] || { echo "ERROR: missing CUDA11.8 Conda Python"; exit 127; }
"$PY" -m py_compile scripts/phase48_masked_transformer_wm.py
"$PY" -m unittest tests.test_phase48_masked_transformer_wm -v
"$PY" scripts/phase48_masked_transformer_wm.py \
 --data-root /scratch/ylong66/data/pks/data \
 --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
 --report-dir "$OUT" \
 --device cpu \
 --ensemble "${PHASE48_ENSEMBLE:-3}" \
 --epochs "${PHASE48_EPOCHS:-60}" \
 --max-patients "${PHASE48_MAX_PATIENTS:-100}" \
 --width "${PHASE48_WIDTH:-96}" \
 --depth "${PHASE48_DEPTH:-2}"
echo "Aggregate report: $OUT"
