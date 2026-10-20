#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-phase52-dusky-%j.log
#SBATCH --error=slurm-phase52-dusky-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
REPO="/home/ylong66/workspace/pks_agent"
PY="/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python"
CV_FOLDS="${PHASE52_FOLDS:-5}"
if [[ "$CV_FOLDS" != 3 && "$CV_FOLDS" != 5 && "$CV_FOLDS" != 10 ]]; then
    echo "Invalid Phase52 folds"; exit 2
fi
OUT="/scratch/ylong66/pks_agent_outputs/phase52_${CV_FOLDS}fold_${SLURM_JOB_ID}"
cd "$REPO"
export PYTHONPATH="$REPO"
[[ -x "$PY" ]] || { echo "Missing local Conda Python: $PY"; exit 127; }
"$PY" -m py_compile scripts/phase50_clinical_latent_wm.py scripts/phase52_full_outer_refit_wm.py
"$PY" -m unittest tests.test_phase52_full_outer_refit_wm -v
"$PY" scripts/phase52_full_outer_refit_wm.py \
  --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
  --report-dir "$OUT" \
  --folds "$CV_FOLDS" \
  --members "${PHASE52_MEMBERS:-3}" \
  --epochs "${PHASE52_EPOCHS:-60}" \
  --width "${PHASE52_WIDTH:-64}" \
  --device cpu --bootstraps 500 --seed 20261018
echo "Aggregate Phase52 report: $OUT"
