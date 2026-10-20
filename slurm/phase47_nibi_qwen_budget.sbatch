#!/bin/bash
#SBATCH --account=rrg-hup
#SBATCH --gpus=h100_2g.20gb:1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-phase47-nibi-%j.log
#SBATCH --error=slurm-phase47-nibi-%j.log

set -euo pipefail

REPO="/home/ylong66/workspace/pks_agent"
PY="/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python"
MODEL="/scratch/ylong66/models/Qwen--Qwen3-4B-Instruct-2507"
OUT="/scratch/ylong66/pks_agent_outputs/phase47_nibi_${SLURM_JOB_ID}"
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/ylong66/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export XDG_CACHE_HOME=/scratch/ylong66/.cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
cd "$REPO"
[[ -x "$PY" && -f "$MODEL/config.json" ]] || {
    echo "ERROR: offline Qwen Conda or local weights missing"; exit 2;
}
"$PY" -m py_compile scripts/phase46_qwen_doctor.py scripts/phase47_question_budget_curve.py
"$PY" -m unittest tests.test_phase46_qwen_doctor tests.test_phase47_question_budget_curve -v
"$PY" scripts/phase46_env_check.py --mode gpu \
  --expected-prefix /scratch/ylong66/conda/envs/pks_qwen46_cu118 \
  --expected-cuda 11.8
"$PY" scripts/phase47_question_budget_curve.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
  --model-dir "$MODEL" \
  --report-dir "$OUT" \
  --max-patients "${PHASE47_MAX_PATIENTS:-100}" \
  --budgets 4 8 12 16
echo "Phase47 complete: $OUT"
