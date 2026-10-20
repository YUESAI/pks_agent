# Fix Phase46 GPU environment with Conda (Nibi / Dusky)

## What failed

The initial Dusky job ran all three Phase46 synthetic tests, then
failed BEFORE Qwen model inference. Existing ms127 has torch 2.5.1
with a too-new Transformers release that disables PyTorch (<2.6),
and a pip-installed safetensors wheel linked against glibc 2.18
that cannot load on the older compute node.

**Do not modify ms127.** Install a separate isolated Conda prefix:
`/scratch/ylong66/conda/envs/pks_qwen46`.

## Setup on NETWORKED LOGIN NODE

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
conda --version
CONDA_EXE="$(conda info --base)/bin/conda" bash scripts/setup_phase46_conda.sh
```

The installer uses only Conda channels (`pytorch`, `nvidia`,
`conda-forge`) for the Qwen Python stack, including compiled
`safetensors` and `tokenizers`. It sets `CONDA_OVERRIDE_GLIBC=2.17`
during dependency resolution so packages should remain compatible
with older Dusky compute nodes. It installs:

- Python 3.11
- PyTorch 2.5.1, CUDA 12.1 (published Conda combination)
- Transformers 4.51.3 (Qwen3 requires >=4.51.0)
- safetensors 0.5.3 (Conda-forge compiled package)
- accelerate 1.7.0, tokenizers 0.21.1, huggingface_hub 0.30.2

NOTE: The combination is a **compatibility candidate**, not yet
verified on your actual glibc/CUDA driver; the Conda solver may
reject it on your platform. In that case, send the full solver error.
Do not blindly install with pip; especially avoid pip's Rust
`safetensors` wheels in this legacy environment.

PyTorch 2.5.1 is an older framework. This is a research-only
inference stack loading trusted locally downloaded safetensors,
not a recommended untrusted-checkpoint/training environment.

## If solver complains (without damaging ms127)

```bash
conda info
conda config --show channel_priority
CONDA_OVERRIDE_GLIBC=2.17 conda search -c conda-forge 'safetensors=0.5.3'
CONDA_OVERRIDE_GLIBC=2.17 conda search -c conda-forge 'transformers=4.51.3'
```

Send the solver message before adding pip packages. Do not force
install an incompatible binary. For a successful installation:

```bash
/scratch/ylong66/conda/envs/pks_qwen46/bin/python \
  scripts/phase46_env_check.py --mode packages
```

## Downloading Qwen model weights

Already downloaded Qwen weights can be reused; no download is needed
if `/scratch/ylong66/models/Qwen--Qwen3-4B-Instruct-2507/`
contains `config.json` and its `.safetensors` shards.

To download only on a login node with the NEW environment:

```bash
export HF_HOME=/scratch/ylong66/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export XDG_CACHE_HOME=/scratch/ylong66/.cache
/scratch/ylong66/conda/envs/pks_qwen46/bin/python \
  scripts/prefetch_optional_agent_models.py --kind doctor
```

## GPU job submission after env installation

Both Slurm scripts now explicitly use pks_qwen46 and verify
`torch`, `transformers`, `safetensors`, `accelerate`, CUDA and
offline local model files before loading Qwen:

```bash
# Dusky L40S
PHASE46_MAX_PATIENTS=12 PHASE46_QUESTIONS=2 \
  sbatch slurm/phase46_dusky_qwen_doctor.sbatch

# OR Nibi H100 MIG
PHASE46_MAX_PATIENTS=12 PHASE46_QUESTIONS=2 \
  sbatch slurm/phase46_nibi_qwen_doctor.sbatch
```

Do NOT submit both to compare before one has passed smoke tests.
Both scripts enforce `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` and preserve original PPMI and the
consumed Phase44 TEST unchanged.

## Diagnostics on new failures

1. `UnsatisfiableError`: copy the Conda solver conflict.
2. `GLIBC_* not found`: check the offending `.so` path; do not
   install newer pip binary wheels inside pks_qwen46.
3. `CUDA driver version insufficient`: show `nvidia-smi` from
   inside the allocated GPU node, plus torch CUDA runtime.
4. `OutOfMemoryError`: reduce `PHASE46_MAX_PATIENTS` (total workload)
   or inference sequence length; 4B bf16 weights need about 8 GB
   plus activations and KV cache.
5. A successful package test does not prove GPU model loading.

Paths: code `/home/ylong66/workspace/pks_agent`; private PPMI
`/scratch/ylong66/data/pks/data`; model weight/cache and all
new Conda packages under `/scratch/ylong66`.
