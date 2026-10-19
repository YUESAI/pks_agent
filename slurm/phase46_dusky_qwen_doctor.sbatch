#!/bin/bash
#SBATCH --account=def-hup-ab_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-phase46-dusky-%j.log
#SBATCH --error=slurm-phase46-dusky-%j.log

set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/ylong66/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export XDG_CACHE_HOME=/scratch/ylong66/.cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="/home/ylong66/workspace/pks_agent"
PYTHON_BIN="/scratch/ylong66/conda/envs/ms127/bin/python"
MODEL="/scratch/ylong66/models/Qwen--Qwen3-4B-Instruct-2507"
DATA="/scratch/ylong66/data/pks/data"
OUT="/scratch/ylong66/pks_agent_outputs/phase46_dusky_${SLURM_JOB_ID}"
cd "$REPO"
export PYTHONPATH="$REPO"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python ms127 unavailable"; exit 127
fi
if [[ ! -f "$MODEL/config.json" ]]; then
  echo "ERROR: Download Doctor Qwen model ON LOGIN NODE before submitting: $MODEL"; exit 2
fi
"$PYTHON_BIN" -m py_compile scripts/phase46_qwen_doctor.py scripts/phase46_llm_doctor_eval.py
"$PYTHON_BIN" -m unittest tests.test_phase46_qwen_doctor -v
"$PYTHON_BIN" -c "import torch, transformers, accelerate; print('CUDA',torch.cuda.is_available(),'transformers',transformers.__version__)"
"$PYTHON_BIN" scripts/phase46_llm_doctor_eval.py \
  --data-root "$DATA" \
  --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
  --model-dir "$MODEL" \
  --report-dir "$OUT" \
  --questions "${PHASE46_QUESTIONS:-4}" \
  --max-patients "${PHASE46_MAX_PATIENTS:-36}" \
  --max-new-tokens 48
echo "Done: $OUT"
