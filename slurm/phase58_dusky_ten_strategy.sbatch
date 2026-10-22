#!/bin/bash
#SBATCH --account=def-hup-ab_cpu
#SBATCH --partition=base
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-phase58-dusky-%j.log
#SBATCH --error=slurm-phase58-dusky-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
REPO="/home/ylong66/workspace/pks_agent"
PY="/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python"
OUT="/scratch/ylong66/pks_agent_outputs/phase58_ten_${SLURM_JOB_ID}"
cd "$REPO"
export PYTHONPATH="$REPO"
[[ -x "$PY" ]] || { echo "ERROR: missing scratch Conda environment"; exit 127; }
echo "=== Python syntax and synthetic regression ==="
"$PY" -m py_compile scripts/phase58_paired_metrics.py scripts/phase58_ten_strategy_sweep.py
"$PY" -m unittest tests.test_phase58_ten_strategy_sweep -v
echo "=== Ten-strategy Train-only evaluation ==="
"$PY" scripts/phase58_ten_strategy_sweep.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
  --report-dir "$OUT" \
  --epochs "${PHASE58_EPOCHS:-25}" \
  --bootstrap "${PHASE58_BOOTSTRAP:-500}" \
  --device cpu
echo "Aggregate report: $OUT"
