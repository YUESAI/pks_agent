#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-phase50-dusky-%j.log
#SBATCH --error=slurm-phase50-dusky-%j.log
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
OUT="/scratch/ylong66/pks_agent_outputs/phase50_cpu_${SLURM_JOB_ID}"
cd "$REPO"
export PYTHONPATH="$REPO"
[[ -x "$PY" ]] || { echo "ERROR: missing Phase46 working Conda environment"; exit 127; }
"$PY" -m py_compile scripts/phase50_clinical_latent_wm.py
"$PY" -m unittest tests.test_phase50_clinical_latent_wm -v
"$PY" scripts/phase50_clinical_latent_wm.py \
  --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
  --report-dir "$OUT" --device cpu \
  --folds "${PHASE50_FOLDS:-3}" --members "${PHASE50_MEMBERS:-3}" \
  --epochs "${PHASE50_EPOCHS:-60}" --width "${PHASE50_WIDTH:-64}" \
  --bootstraps 300 --seed 20261018
echo "Aggregate Phase50 report: $OUT"
