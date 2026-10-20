#!/bin/bash
#SBATCH --account=rrg-hup
#SBATCH --gpus=h100_2g.20gb:1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-phase46-nibi-%j.log
#SBATCH --error=slurm-phase46-nibi-%j.log

set -euo pipefail
REPO="/home/ylong66/workspace/pks_agent"
cd "$REPO"
PYTHON_BIN="/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then echo "ERROR: install pks_qwen46 Conda environment on login node first"; exit 127; fi
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch/ylong66/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export XDG_CACHE_HOME=/scratch/ylong66/.cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

MODEL=/scratch/ylong66/models/Qwen--Qwen3-4B-Instruct-2507
OUT="/scratch/ylong66/pks_agent_outputs/phase46_nibi_${SLURM_JOB_ID}"
if [[ ! -f "$MODEL/config.json" ]]; then
    echo "ERROR: Download model on networked login node first"; exit 2
fi
"$PYTHON_BIN" -m py_compile scripts/phase46_qwen_doctor.py scripts/phase46_llm_doctor_eval.py
"$PYTHON_BIN" -m unittest tests.test_phase46_qwen_doctor -v
"$PYTHON_BIN" scripts/phase46_env_check.py --mode gpu --expected-prefix /scratch/ylong66/conda/envs/pks_qwen46_cu118 --expected-cuda 11.8
"$PYTHON_BIN" scripts/phase46_llm_doctor_eval.py \
    --data-root /scratch/ylong66/data/pks/data \
    --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
    --model-dir "$MODEL" --report-dir "$OUT" \
    --questions "${PHASE46_QUESTIONS:-4}" \
    --max-patients "${PHASE46_MAX_PATIENTS:-36}" \
    --max-new-tokens 48
echo "Done: $OUT"
