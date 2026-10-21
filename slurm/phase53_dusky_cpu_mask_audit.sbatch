#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-phase53-dusky-%j.log
#SBATCH --error=slurm-phase53-dusky-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
REPO="/home/ylong66/workspace/pks_agent"
PY="/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python"
OUT="/scratch/ylong66/pks_agent_outputs/phase53_masked_${SLURM_JOB_ID}"
cd "$REPO"
export PYTHONPATH="$REPO"
[[ -x "$PY" ]] || { echo "Missing existing scratch Conda environment"; exit 127; }
"$PY" -m py_compile scripts/phase53_mask_aware_motor_readiness.py
"$PY" -m unittest tests.test_phase53_mask_aware_motor_readiness -v
"$PY" scripts/phase53_mask_aware_motor_readiness.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
  --report-dir "$OUT" --min-cell 10
echo "Phase53 aggregate: $OUT"
