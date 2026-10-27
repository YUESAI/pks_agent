# Phase72 advanced WM+Verifier mechanism — evidence-based directions (2026)

References informing design, **not code clones**:
- Dreamer 4 (2025 preprint), arxiv.org/abs/2509.24527: latent
  state prediction and imagination instead of post-hoc fixed blending.
- Learning-to-Measure (ICML 2026), proceedings.mlr.press/v306/kobayashi26a.html:
  learn prediction from partially disclosed observations.
- IJCAI 2026 active feature acquisition survey:
  budget-constrained information selection with epistemic uncertainty.
- AIDE2 (2026 preprint), arxiv.org/abs/2609.26457, and ReVeal (ICLR
  2026): proposer / external verifier separation and rollback.
- Verifiable Medical World Models (2026), link.springer.com/article/10.1007/s10278-026-02413-0:
  calibration, fidelity, safety and predictive horizon must be tested.

Actual Phase70 outcome: Prodromal 12m MAE Plain1.0416 static1.0511
temporal1.0841 verified1.0457, 24m Plain1.1335 static1.1999
temporal1.3043 verified1.2995. Temporal fusion regressed; hence
Phase72 DOES NOT use Phase70's .35 temporal blend.

## What is concretely implemented

- A small continuous-time decay + GRU latent world model trained on
  FIT-person **actual chronological Motor visits** and task-specific
  actual future six-item Motor outcomes. Six categorical 0..4 heads,
  separate calibrated-in-principle binary head; time gap included.
- FIT-only adjacent visit auxiliary prediction (true measured Motor),
  with patient-level disjointness from DEV/AUDIT/outer evaluation.
- Three independently seeded checkpoints to estimate disagreement;
  no imagined clinical result enters the data access tool.
- Baseline full-space static Ridge, one latent checkpoint, ensemble,
  and one real-Qwen optional-test choice + **once-audited** conditional
  checkpoint switch; these are four deliberately distinct mechanisms.
- Five outer patient-level folds x four arms, total 20 independent CPU
  jobs. Arm 3 alone loads offline Qwen. Baseline only uses BL Motor
  observations; optional baseline MoCA/UPSIT are requested under
  an identical ONE-question budget, fixed lexical rule for arms 0-2,
  real Qwen choice for arm 3. Qwen only selects an actual already
  recorded optional score. All future labels remain withheld.
- Strict original frozen Prodromal TRAIN whitelist and separate PD
  development population. No previously consumed final TEST.
- Candidate RSI checkpoint ensemble vs member0 chosen on development
  and accepted only if it again improves MAE on one independent AUDIT.
  This is **verified model-checkpoint revision**, not an autonomous
  algorithm-writing RSI, and does not evolve a new network architecture.

Limitations: This uses small latent networks not pretrained 2026 video
models. The Qwen-selected feature protocol of arm3 differs from the
fixed-feature protocols of arms0-2; results must not attribute such
differences solely to RSI. In addition, the current main evaluation
origin remains BL, rather than multiple origins per patient. The
auxiliary world model exploits FIT-only multiple visits, but
prospective multi-origin testing is still required. Models need
calibration checking and independent heldout cohort validation.

## Submit 20 standalone CPU jobs (no array)

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
  sbatch --export=ALL,PHASE72_ARM_ID="$arm" slurm/phase72_dusky_cpu.sbatch
done
```

Each job has 8 CPUs, 48GB and a base-partition-compatible requested
12-hour maximum. All 20 jobs are separately submitted, NOT an array.
The five Qwen jobs can take longer; actual Slurm scheduler start depends
on quotas. Outputs: metrics.csv, coverage.csv, audits.json, summary.json.
Pack aggregate reports only (no patient-level files):
`zip -r phase72_reports.zip phase72_ctlatent_*/`.
