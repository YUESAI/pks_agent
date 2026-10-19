# Phase46 — Actual Qwen Doctor Agent with offline GPU inference

**Status:** code implemented; cluster inference is not yet verified.
Phase45 is a CPU probabilistic WM + deterministic agent + RSI-lite baseline.
Phase46 adds a **real Qwen3-4B-Instruct-2507 Doctor backbone**, not yet
an LLM-based RSI evolver or a neural temporal World Model.

## Architecture and scientific controls

Doctor: local Qwen3-4B-Instruct-2507 receives allowed tools, previously
measured Motor items, and optional WM-predicted risk/information gain.
Outputs one JSON item request. Strict parser checks the tool name and
rejects invalid/repeated/unavailable requests; invalid output counts
as fallback to safe admissible max-information tool.

Patient: only `PatientEvidenceService` returns measured V04 item values.
Six hidden hand-rigidity/postural stability items and target labels
are never exposed to Doctor, nor fabricated by Patient. The WM remains
the class-conditional categorical model from Phase45.

Two Qwen modes are compared to q-matched controls:

| Mode | Uses real LLM? | WM advice to LLM? | Queries |
|---|---|---|---|
| llm_plain | yes | no | q |
| llm_wm | yes | yes | q |
| static_questions | no | yes, static policy | q |
| dynamic_questions | no | yes, adaptive policy | q |
| visible_sum_27 | no | no | all 27 |
| wm_full_27 | no | yes | all 27 |

Phase46 takes a fixed hash-derived heldout subset of frozen TRAIN V04
episodes, fits WM on disjoint TRAIN patients and ranks the heldout
patients under a 20% simulated onsite acquisition budget. Results:
positive-case recall, precision, hidden-score burden captured,
AUROC, Brier score and invalid LLM tool-call counts.
Outcomes are same-visit V04 masked labels, NOT real R01 labels.
Original Phase44 final TEST is CONSUMED and is not touched.

## Step 1: install Python dependencies and download model on a LOGIN NODE

Run only on the login node with internet access. Keep the large files
on scratch, **not in Git or home**. Qwen model is approximately 8 GB.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
conda activate ms127
export HF_HOME=/scratch/ylong66/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export XDG_CACHE_HOME=/scratch/ylong66/.cache
export PIP_CACHE_DIR=/scratch/ylong66/.cache/pip
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" /scratch/ylong66/models
python -m pip install -U 'transformers>=4.51.0' accelerate huggingface_hub safetensors
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python scripts/prefetch_optional_agent_models.py --kind doctor
du -sh /scratch/ylong66/models/Qwen--Qwen3-4B-Instruct-2507
ls -l /scratch/ylong66/models/Qwen--Qwen3-4B-Instruct-2507/*.safetensors
```

PyTorch must be **CUDA compatible** on the chosen GPU cluster;
`torch.cuda.is_available()` on an ordinary login node may be false.
Do not replace an existing working CUDA PyTorch install blindly.
Pin exact dependency/model revisions after verifying a successful run.
The downloader defaults to revision `main`; specify an immutable
`--revision <commit-sha>` for experiments intended for publication.

## Step 2: preflight that does NOT read patient data or use GPU

```bash
python -m py_compile scripts/phase46_qwen_doctor.py scripts/phase46_llm_doctor_eval.py
python -m unittest tests.test_phase46_qwen_doctor -v
```

## Step 3: submit one GPU job (choose whichever cluster has availability)

**Dusky L40S:**

```bash
sbatch slurm/phase46_dusky_qwen_doctor.sbatch
```

**Nibi H100 MIG (20GB allocation):**

```bash
sbatch slurm/phase46_nibi_qwen_doctor.sbatch
```

Both jobs are offline, using `HF_HUB_OFFLINE=1`, with the same model
local directory and aggregate report root. Each job requires no
network and writes under:

`/scratch/ylong66/pks_agent_outputs/phase46_dusky_<jobid>/` or
`/scratch/ylong66/pks_agent_outputs/phase46_nibi_<jobid>/`.

Initial pilot runs **36 heldout TRAIN patients × 4 tool requests ×
2 LLM modes = up to 288 LLM generations**. This can be lengthy on
one GPU; increase sample size only after a successful smoke test.
Because all models are loaded once per job, do not spawn independent
Qwen instances per patient. Computational runtimes have not been
measured on these clusters.

Optional Slurm overrides at submission:

```bash
PHASE46_MAX_PATIENTS=12 PHASE46_QUESTIONS=2 \
  sbatch slurm/phase46_dusky_qwen_doctor.sbatch
```

Check results, replacing the example job ID with the actual ID:

```bash
squeue -u "$USER"
ls -d /scratch/ylong66/pks_agent_outputs/phase46_*
cat /scratch/ylong66/pks_agent_outputs/phase46_dusky_<JOBID>/summary.json
cd /scratch/ylong66/pks_agent_outputs
zip -r phase46_qwen_doctor_report.zip phase46_dusky_<JOBID>
```

Never upload patient-level records, individual tool transcripts,
private longitudinal manifest or raw clinical files. Model outputs
may not be clinically valid or safe; this is a research simulation.

## Research decisions after results

1. If `llm_wm` beats `llm_plain` at identical q, test the benefit
   of exposing calibrated WM uncertainty to the reasoning backbone.
2. Compare `llm_wm` against q-matched `dynamic_questions`; mere
   comparison with random selection is insufficient.
3. Only after this evidence, introduce a separately evaluated
   LLM-based RSI proposer whose revisions are verified and rolled back
   using inner TRAIN development data; do not peek into consumed TEST.
4. Neural temporal WM and optional VLM are further modules, not
   part of Phase46. VLM is only warranted for genuinely image-backed
   clinical observations.
