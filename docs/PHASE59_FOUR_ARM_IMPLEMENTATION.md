# Phase59: four parallel VERA-PD engineering pilots on Dusky L40S

**Implementation committed 2026-10-22.**
This is an **integrated, grounded prototype**, not a validated
clinical referral system, not full Dreamer4 and not autonomous
WM weight rewriting. Frozen Prodromal PPMI TRAIN only.
The original Phase44 TEST was consumed; never re-open.

## What is actually integrated

The actual local `Qwen3-4B-Instruct-2507` LLM chooses allowed
Motor item request actions, parsed by the existing strict
JSON adapter `scripts/phase46_qwen_doctor.py`. The original
`PatientEvidenceService` releases only actually recorded
onsite V04 0..4 scores for the 27 remotely visible items.
Six hands-on scores NEVER appear in the Qwen prompt or tool return.

- **Current same-visit observation WM:** Phase45
  categorical observation likelihood, fitted on the
  fit-only V04 patient subset. This is a weak observation
  kernel, not an up-to-date pretrained generative WM.
- **Time-conditioned WM prior:** Phase54 small irregular-time
  recurrent state model, trained on longitudinal patients
  excluding all site DEV and evaluation patient IDs. It
  receives an actual earlier BL exam and target V04 month
  but NEVER V04 measured score or status. Forecast six
  future score distributions combine to a *factorized,
  uncalibrated* nonzero prior. If no eligible BL history
  exists, temporal weight is disabled for that patient.
- **Fusion:** prior logits are blended with fixed weight
  0.5 and updated with observed-item likelihood ratios.
  The q-step Doctor may condition on estimated item
  informativeness; actual observations come solely from
  PatientEvidenceService.
- **RSI:** local Qwen proposes ONLY `weight`, `temperature`
  and `style`, from a fixed allowlist; DEV-only *greedy
  proxy* simulation (not Qwen dialogues) is checked by a
  deterministic independent Brier verifier. Accept only
  if improvement > 1e-5; otherwise rollback. A version
  hash and per-round accept/reject trace are exported.
  **This does not update WM neural weights, self-rewrite
  executable code or prove verifier-gated joint RSI.**

## The four arms

| arm | Qwen | observation WM guidance | temporal WM prior | allowlisted RSI |
|---|---|---|---|---|
| `plain` | yes | hidden from LLM; fixed observation readout | no | no |
| `observation` | yes | yes | no | no |
| `dual` | yes | yes | yes, weight 0.5 | no |
| `rsi` | yes | yes | yes (candidate weight) | yes, DEV-only verifier |

All four use **exactly the same patient hash split**:
65% site FIT, 15% site DEV, then common fixed hash
evaluation cohort capped at `PHASE59_MAX_PATIENTS`.
Same exact measured question budget, same Doctor
backbone and deterministic generation configuration.
Each study arm runs in a separate L40S GPU Slurm job.
The temporal WM itself fits on CPU inside that GPU job.

Main metric: simulated onsite-hidden-six binary label
Brier, AUROC and 20%-quota recall, plus invalid Qwen
proposals, total questions, generated LLM tokens
and clock seconds. Output is **aggregate only**:
`arm_metrics.csv`, `summary.json`;
the RSI arm also writes `rsi_verifier_trace.csv`.
Patient identifiers, per-person selected items, scores,
verbatim prompts and model transcripts are not exported.

**Important non-equivalence:** the plain mode still uses
the fitted observation WM as a *fixed post-question risk
readout*, it only withholds WM guidance from the LLM.
Therefore differences directly test the effect of
WM-to-Doctor advice, not the value of having *any*
outcome predictor at all.

### Launch: four jobs in parallel

Internet required for NONE of these commands: Qwen
weights and Conda environment already exist on Scratch.
From the current repository checkout:

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
"$PY" -m py_compile scripts/phase59_grounded_collaboration.py \
 scripts/phase59_temporal_prior.py scripts/phase59_rsi_verifier.py \
 scripts/phase59_integrated_pilot.py
"$PY" -m unittest tests.test_phase59_integrated_pilot -v
```

Four independent submissions **with the same q and n**:

```bash
sbatch --export=ALL,PHASE59_ARM=plain,PHASE59_QUESTIONS=4,PHASE59_MAX_PATIENTS=24 slurm/phase59_dusky_four_arm.sbatch
sbatch --export=ALL,PHASE59_ARM=observation,PHASE59_QUESTIONS=4,PHASE59_MAX_PATIENTS=24 slurm/phase59_dusky_four_arm.sbatch
sbatch --export=ALL,PHASE59_ARM=dual,PHASE59_QUESTIONS=4,PHASE59_MAX_PATIENTS=24 slurm/phase59_dusky_four_arm.sbatch
sbatch --export=ALL,PHASE59_ARM=rsi,PHASE59_QUESTIONS=4,PHASE59_MAX_PATIENTS=24 slurm/phase59_dusky_four_arm.sbatch
```

These are a **smoke/engineering batch**. The GPU scheduler
will start them concurrently IF at least four L40S
allocations are available; otherwise queue them and let
Slurm schedule. Do not interpret smoke results
as statistically conclusive.

After smoke tests pass, an exploratory larger batch
can use q=8 and n=60 (same values in every arm):

```bash
for arm in plain observation dual rsi; do
 sbatch --export=ALL,PHASE59_ARM="$arm",PHASE59_QUESTIONS=8,PHASE59_MAX_PATIENTS=60 \
   slurm/phase59_dusky_four_arm.sbatch
done
```

Each job uses one L40S, 4 CPUs, 32 GB RAM and at most
12 hours. `squeue -u "$USER"` and
`sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS`
show actual status.

Individual job report:
`/scratch/ylong66/pks_agent_outputs/phase59_<arm>_<JOB_ID>/`.
Pack successful directories only using actual IDs:

```bash
cd /scratch/ylong66/pks_agent_outputs
zip -r phase59_four_arm_reports.zip \
 phase59_plain_<ID1>/ phase59_observation_<ID2>/ \
 phase59_dual_<ID3>/ phase59_rsi_<ID4>/
```

Small-case positive label counts below 10 are
suppressed in aggregate exports. Not a substitute for
formal differential privacy.

## What this can and cannot prove

- Validates real LLM tool actions, posterior updates,
  a longitudinal prior source, constraints on proposed
  revisions, verifier acceptance/rollback and running
  four arms with equal q/n.
- The Doctor currently acts on a **same-V04 measured-item
  masking simulation**; this is not a real R01 clinical
  intervention or proof that questions cause improvement.
- Temporal WM contributes only a *risk prior*, not
  a full multi-step action-conditioned imagination
  simulator. RSI is a restricted policy/prior proposal
  optimizer and its inner scoring uses a greedy proxy;
  not yet neural WM + Doctor joint weight evolution.
- Four separate jobs output aggregate metrics and
  cannot produce patient-paired confidence intervals
  without a new single-process joint evaluation.
- This is repeatedly explored TRAIN data; for a paper
  freeze all design choices and use genuinely new,
  independent patients, with an appropriate codebook.

Next Phase60 upgrades the within-visit observation WM to
a stronger explicit conditional model and runs real
branching hypothetical item trajectories; Phase61
then tests coupled parameter/policy update ablations
with two-tier verifiers and rollback.
