#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-phase51-dusky-%j.log
#SBATCH --error=slurm-phase51-dusky-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
REPO="/home/ylong66/workspace/pks_agent"
PY="/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python"
OUT="/scratch/ylong66/pks_agent_outputs/phase51_readiness_${SLURM_JOB_ID}"
cd "$REPO"
export PYTHONPATH="$REPO"
"$PY" -m unittest tests.test_phase51_longitudinal_visit_readiness -v
"$PY" scripts/phase51_longitudinal_visit_readiness.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
  --report-dir "$OUT" \
  --min-cell 10
echo "Aggregate readiness output: $OUT"
