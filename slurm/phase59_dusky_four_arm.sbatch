#!/bin/bash
#SBATCH --account=def-hup-ab_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-phase59-%x-%j.log
#SBATCH --error=slurm-phase59-%x-%j.log
set -euo pipefail
module --force purge || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/ylong66/hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export XDG_CACHE_HOME=/scratch/ylong66/.cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/home/ylong66/workspace/pks_agent
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
MODEL=/scratch/ylong66/models/Qwen--Qwen3-4B-Instruct-2507
DATA=/scratch/ylong66/data/pks/data
ARM="${PHASE59_ARM:?Set PHASE59_ARM=plain|observation|dual|rsi}"
case "$ARM" in plain|observation|dual|rsi) ;; *) echo "Bad arm $ARM"; exit 2;; esac
OUT="/scratch/ylong66/pks_agent_outputs/phase59_${ARM}_${SLURM_JOB_ID}"
cd "$REPO"
export PYTHONPATH="$REPO"
[[ -x "$PY" && -f "$MODEL/config.json" ]] || {
 echo "ERROR: missing offline Conda/Qwen checkpoint";exit 127;}
echo "phase59 arm=$ARM job=$SLURM_JOB_ID q=${PHASE59_QUESTIONS:-4} n=${PHASE59_MAX_PATIENTS:-24}"
"$PY" -m py_compile scripts/phase59_grounded_collaboration.py \
 scripts/phase59_temporal_prior.py scripts/phase59_rsi_verifier.py \
 scripts/phase59_integrated_pilot.py
"$PY" -m unittest tests.test_phase59_integrated_pilot -v
"$PY" scripts/phase46_env_check.py --mode gpu \
 --expected-prefix /scratch/ylong66/conda/envs/pks_qwen46_cu118 \
 --expected-cuda 11.8
"$PY" scripts/phase59_integrated_pilot.py \
 --arm "$ARM" --data-root "$DATA" \
 --manifest-db "$REPO/outputs/longitudinal_manifests/prodromal_motor_v0.sqlite" \
 --model-dir "$MODEL" --report-dir "$OUT" \
 --questions "${PHASE59_QUESTIONS:-4}" \
 --max-patients "${PHASE59_MAX_PATIENTS:-24}" \
 --max-new-tokens "${PHASE59_MAX_NEW_TOKENS:-48}" \
 --temporal-epochs "${PHASE59_TEMPORAL_EPOCHS:-8}" \
 --rsi-rounds "${PHASE59_RSI_ROUNDS:-3}"
echo "SUCCESS arm=$ARM aggregate=$OUT"
