# Phase51 results (job 330387) and Phase53 mask-aware follow-up

## Directly observed in user-provided Phase51 ZIP (2026-10-20)

Files: `primary_visit_distribution.csv`,
`primary_baseline_anchored_distribution.csv`,
`medication_state_visit_distribution.csv`,
`primary_followup_gap_bins.csv`, `data_quality_counts.csv`,
`summary.json`.

Only frozen PPMI Prodromal TRAIN patients were queried.
`MDS-UPDRS_Part_III_06Oct2026.csv` was the source.

| Metric | Value |
|---|---:|
| Frozen TRAIN participants | 1,312 |
| Anchored TRAIN participants (month available) | 1,207 |
| People with verified BL Motor `NP3TOT` and UNSPECIFIED state | 1,149 |
| Usable unique patient-month-state Motor TOTAL records across UNSPECIFIED/ON/OFF | 2,405 |
| All UNSPECIFIED adjacent valid observation pairs (includes people without a valid BL) | 1,189 |
| Valid BL plus ≥1 follow-up Motor total | 647 |
| Valid BL plus ≥2 follow-ups | 375 |
| Valid BL plus ≥3 follow-ups | 120 |
| Valid BL plus ≥4 follow-ups | 17 |

Primary baseline-anchored UNSPECIFIED Motor visit histogram:
1 visit = 502 people; 2 visits = 272; 3 visits = 255;
4 visits = 103; ≥5 visits = 17.
A total of **1,149** such people appear in the histogram.
The broader, not necessarily baseline-anchored UNSPECIFIED
histogram counts 0=136, 1=514, 2=280, 3=262, 4=103,
≥5=17 (sums to frozen 1,312).

Unrestricted UNSPECIFIED successive visit-pair gap bins:
1–3 months `<10` (suppressed); 4–8 months 102;
9–15 months 1,016; 16–24 months 52;
25–36 months 16; ≥37 months 0.
These are not 1,189 independent patients and are not all confirmed
as **baseline-anchored** sequences.

Source quality aggregates: 38 BL anchor mismatches,
422 conflicted patient-month-state keys, 607 source rows with
missing BL anchor, and 422 invalid/non-numeric NP3TOT rows.
**Do not add these counts and assume disjoint bad encounters.**
Most importantly, an absent NP3TOT can reflect a legitimate
remote motor exam with 27 valid component items and six 101/UR
hands-on items, not a fabricated or useless clinical visit.

## What Phase51 does NOT prove

- It counts unique valid **total-score observations**, not all
  partially observed motor measurements or exact full item sequences.
- 1,189 adjacency pairs include UNSPECIFIED sequences without valid
  BL, so they overstate anchored-transition capacity.
- OBSERVED followup times are irregular, mostly annual.
- `UNSPECIFIED` is unknown medication state, not a uniform
  untreated or OFF state.
- Train-only audits are not new independent validation cohorts.

## Phase53: correct for the structural remote unobservability

The previously frozen real R01 audit confirmed that remote
`R01` commonly records all 27 visible MDS-UPDRS III items
but marks six hands-on items `101=Unable to Rate`, leaving
`NP3TOT` unreported. Phase51 excludes many of these as
unusable totals. That is inappropriate for learning a
**partially observable** clinical World Model.

`scripts/phase53_mask_aware_motor_readiness.py` therefore counts
raw TRAIN Motor visits using 33 actual per-item values plus
an explicit missingness mask, separating:

- verified complete 33-item clinical observations;
- 33-item observations without documented total;
- real 27-visible + six hands-on unrateable remote R01;
- partially observed (≥27 items, or 10–26 items);
- sparse/unusable visits (<10 observed scores), excluded.

Only actual scored items (0,1,2,3,4) are observed; 101/UR
scores are **never imputed as 0**. Repeated discordant patient-month
episodes are quarantined; R01/V04 must be in their study visit
windows and only frozen TRAIN patients are examined. ON and OFF
are not merged with UNSPECIFIED. The report distinguishes
BL-anchored transition pairs from unrestricted pairs.

Outputs: `episode_mask_patterns.csv`,
`event_by_mask_pattern.csv`,
`all_patient_episode_distribution.csv`,
`baseline_anchored_episode_distribution.csv`,
`anchored_adjacent_gap_bins.csv`, `quality_flags.csv`,
`summary.json`. All are aggregates; no per-person PHI leaves
the cluster.

## Run Phase53 on Dusky CPU (no downloads)

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
"$PY" -m unittest tests.test_phase53_mask_aware_motor_readiness -v
sbatch slurm/phase53_dusky_cpu_mask_audit.sbatch
```

Read the actual Slurm job ID from `sbatch`, then:

```bash
cd /scratch/ylong66/pks_agent_outputs
ls -d phase53_masked_*
# Replace <jobid> below, not a sample numeric ID.
zip -r phase53_masked_report.zip phase53_masked_<jobid>/
```

## Methodological decision

Instead of training a larger temporal neural model on just the
332 strict three-total-score patient subset, first reconstruct
all valid partial observation trajectories for frozen TRAIN.
Count masks, site modality, future measured assessments and
followup semantics. If feasible, next train a **masked,
irregular-time latent state World Model** with:

- a patient belief posterior over latent disease state,
- a dynamics model using actual time intervals but no
  unverified treatment causality,
- observation heads for discrete Motor items and other
  provenance-checked data modalities,
- ground-truth calibration and multi-step rollout audits,
- Doctor's information acquisition as *measurement action* that
  updates beliefs, not automatically physical state.

Only then assess how model-based imagination improves Qwen
Doctor and verifier-guided RSI on appropriately separated
TRAIN development data. The Phase44 TEST is consumed.
