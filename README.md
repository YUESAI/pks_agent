# pks_agent

Parkinson's disease multi-agent / RSI + world model research.

## Step 1 — Read-only dataset audit

**Code and raw datasets can live in different directories.** The tool does not
change, move, upload, or copy any raw patient data.

Example cluster layout:

```text
/path/to/pks_agent/                   # Git repository with scripts
/scratch/ylong66/data/pks/data/       # Independent raw dataset directory
    PPMI/
    BioFIND/
    FoxInsight/
```

Run from the Git repository checkout:

```bash
git pull
python scripts/audit_data.py \
  --data-root /scratch/ylong66/data/pks/data \
  --output-dir ./outputs/data_audit \
  --max-rows 500
```

Requires Python 3.9+; **CSV, TSV, and gzip-compressed CSV/TSV require no third-party libraries**. To scan Excel, Parquet, or SAS files, optionally install:

```bash
pip install openpyxl pyarrow pyreadstat
```

If an optional reader is missing, the file still appears in `files.csv`
with `status=error`; the rest of the audit continues.

Outputs (all schema/statistics, **no patient-level cell values**):

| File | Purpose |
| --- | --- |
| `report.md` | Human-readable summary, likely join tables |
| `summary.json` | Counts of files, formats, and issues |
| `files.csv` | File inventory, size, type, columns count |
| `columns.csv` | Column names, suspected key/label roles, sampled missingness |
| `join_candidates.csv` | Tables potentially joinable on patient/visit keys |
| `issues.csv` | Files that could not be parsed and structural warnings |

The audit reads only the first `--max-rows` records per table by default,
**not** a full count of CSV records. Results are heuristic: a column named
`PATNO` is a *candidate* key, not proof that a join is valid. Never use suspected
diagnosis or cohort-assignment fields as Doctor Agent observations until checked
for label leakage. Negative missingness codes are not imputed automatically.

For a quicker header-only first pass:

```bash
python scripts/audit_data.py --data-root /scratch/ylong66/data/pks/data \
  --output-dir ./outputs/headers --max-rows 0
```

Run synthetic tests:

```bash
python -m unittest discover -s tests -v
```

**Sharing reports:** Review file names/column headers before sharing outside
your project; raw data must remain on the authorized cluster. Audit reports
should not be committed to GitHub. They are ignored by `.gitignore`.

### After the first audit

Use the collected inventory to choose a **small, verified feature catalog**,
then build a harmonized DuckDB/Parquet layer. We intentionally do **not**
automatically union raw PPMI, BioFIND, and Fox Insight tables based only on
column-name similarities.

## Step 2 — Full clinical coverage (PPMI / BioFIND / Fox Insight)

After the structural inventory, run the selected clinical CSV profiler (standard-library only):

```bash
python scripts/profile_clinical.py \
  --data-root /scratch/ylong66/data/pks/data \
  --output-dir ./outputs/clinical_profile
```

Unlike the first header/sample audit, **the default scans all rows** of 26
selected core CSV tables and computes distinct patient and patient/visit counts
in memory. No raw patient records or IDs are written to the report.

Outputs: `report.md`, `summary.json`, `table_coverage.csv`,
`feature_coverage.csv`, `label_distributions.csv`,
`event_distributions.csv`, `issues.csv`. Diagnosis/event categories
with fewer than 10 records are suppressed.

If you want a fast preview before full scanning, add `--max-rows 20000`.
Review `docs/clinical_mapping_v0.md` for verified column *existence* and
warnings about label leakage, diagnosis definitions, and post-dose exams.

Run tests with `python -m unittest discover -s tests -v`. Outputs under
`outputs/` are ignored by Git and must never be committed.

## Step 3 — Episode feasibility audit

Run from repository checkout with the raw data elsewhere:

```bash
python scripts/episode_audit.py \
  --data-root /scratch/ylong66/data/pks/data \
  --output-dir ./outputs/episode_audit \
  --screening-window-days 180
```

Outputs: report.md, summary.json, episode_overview.csv, feature_by_view.csv,
joint_combinations.csv, joint_by_raw_diagnosis.csv, duplicate_conditions.csv,
BL_diagnosis_codes.csv, cohort_vs_diagnosis.csv, official_codebook_subset.csv,
issues.csv. Reports contain aggregate statistics rather than patient-level values.
See docs/episode_design_v0.md. **Do not train from the undated SC_BL_candidate view.**

## Phase 5 — Local patient database (PPMI and BioFIND)

Raw data and code may be in different directories. Patient-level databases stay only on the authorized cluster.

Install DuckDB: `pip install duckdb`

Run from the repository checkout:

```bash
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/build_patient_db.py --data-root /scratch/ylong66/data/pks/data --db-path ./outputs/patient_db/pks.duckdb --engine duckdb
python scripts/report_patient_db.py --db-path ./outputs/patient_db/pks.duckdb --engine duckdb --output-dir ./outputs/patient_db_report
```

Or use SQLite without any new dependency:

```bash
python scripts/build_patient_db.py --data-root /scratch/ylong66/data/pks/data --db-path ./outputs/patient_db/pks.sqlite --engine sqlite
python scripts/report_patient_db.py --db-path ./outputs/patient_db/pks.sqlite --engine sqlite --output-dir ./outputs/patient_db_report
```

**Only share outputs/patient_db_report with collaborators, not the DB file.**

The database contains `patients`, `restricted_labels`, `observations`, `visits`, `feature_catalog`, and `build_metadata`.
Candidate PD/HC labels require enrollment and BL diagnosis concordance; they are not yet clinically validated.
The on-request `scripts/patient_store.py` exposes no diagnosis labels and does not voluntarily disclose patient results.
Patient-level scores, IDs, provenance, and unresolved repeated exams remain inside the database.
All SC+BL combined counts remain an undated exploratory upper bound, NOT training-ready records.
Medication state, imputation and suspected missing-value codes must be reviewed before modeling.

## Phase 6 — Baseline date and episode readiness audit

This reads the **existing local database** and the PPMI `Baseline_Visit_Start` CSV; it does not rewrite the database. Reports are aggregate only.

```bash
python -m unittest discover -s tests -v
python scripts/episode_readiness.py \
  --db-path ./outputs/patient_db/pks.duckdb \
  --engine duckdb \
  --data-root /scratch/ylong66/data/pks/data \
  --output-dir ./outputs/episode_readiness \
  --screening-window-days 180
```

Outputs: `report.md`, `summary.json`, `baseline_date_coverage.csv`, `date_source_status.csv`, `feature_readiness.csv`, `duplicate_resolution.csv`, `joint_readiness.csv`. Upload only this directory, not the local database.

Important: dated measurements are checked relative to **baseline visit start**, not a verified clinical diagnostic decision time. `bl_dated` and `union_dated` are conservative QC subsets, not validated training splits. BioFIND does not yet have a verified BL anchor; its date-qualified coverage is intentionally zero pending review.

## Phase 7 — Investigate date formats and repeated motor exams

The Phase 6 baseline date audit parsed **0 / 3,048** PPMI BL start records. Before changing the database, identify the raw date encoding and repeat exam conditions with read-only aggregate reports.

```bash
python -m unittest discover -s tests -v
python scripts/probe_dates_motor.py \
  --data-root /scratch/ylong66/data/pks/data \
  --output-dir ./outputs/date_motor_probe
```

Outputs: `date_patterns.csv`, `date_parse_summary.csv`, `visit_start_timing.csv`, `motor_repeat_factors.csv`, `issues.csv`, `summary.json`. Date patterns are character-class masks, NOT date strings. **Review report contents before uploading.** The cluster raw data are not edited.

Do not overwrite the previous patient database yet: this tool does not repair stored date values or define a diagnosis timestamp.
## Phase 8 — Restore month-precision clinical dates in a NEW DB

Phase 7 established a uniform `99/9999` date shape in clinical INFODT/EXAMDT.
Treat this as provisional MM/YYYY, preserve month precision and never invent the day.
PPMI motor exam uses EXAMDT when nonempty, otherwise INFODT. No clinical ordering
within the same month can be established by these fields.

Run from your repository checkout with a NEW database path (do not overwrite pks.duckdb):

```bash
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/build_patient_db.py \
  --data-root /scratch/ylong66/data/pks/data \
  --db-path ./outputs/patient_db/pks_v2.duckdb \
  --engine duckdb
python scripts/month_precision_audit.py \
  --data-root /scratch/ylong66/data/pks/data \
  --db-path ./outputs/patient_db/pks_v2.duckdb \
  --engine duckdb \
  --output-dir ./outputs/month_audit
```

SQLite fallback: replace the DB path with `pks_v2.sqlite` and set `--engine sqlite`
on both commands. Share ONLY the aggregate `outputs/month_audit/` reports.

New observation columns: `exam_month` (YYYY-MM), `date_precision`
(`month_candidate`, `day`, `missing`, or `unknown`) and `date_source`.
`exam_date` remains NULL for month-only observations.

NOTE: This is a review-only database. The baseline anchor is the visit-start
month, NOT a verified diagnosis decision; exam and anchor in the same month
remain temporally unordered. Do not merge SC+BL episodes for training yet.

## Phase 9 — Retrospective doctor-patient episode manifest

Input: existing `outputs/patient_db/pks_v2.duckdb` and original PPMI
`Baseline_Visit_Start` file. Source data and code can live in separate directories.

The local SQLite manifest is **patient-level** (contains IDs, observation provenance)
and must remain on the authorized research cluster. The separate reports are
aggregate only. No diagnostic labels are included in the Doctor-facing manifest.

```bash
python -m unittest discover -s tests -v
python scripts/episode_manifest_v0.py \
  --data-root /scratch/ylong66/data/pks/data \
  --source-db ./outputs/patient_db/pks_v2.duckdb \
  --source-engine duckdb \
  --manifest-db ./outputs/episode_manifests/episodes_v0.sqlite \
  --report-dir ./outputs/episode_manifest_report \
  --screening-window-months 6
```

Upload only `outputs/episode_manifest_report/` (after reviewing contents),
**NOT** `episodes_v0.sqlite`, `pks_v2.duckdb`, raw patient CSV files, or row-level results.

The manifest offers 3 motor conditions where present: `ON`, `OFF`, `UNSPECIFIED`.
Conflicting repeated exams are not silently deduplicated. `UPSIT` may fall
back to `SC` only when its month is strictly earlier than the `BL` visit-start
month, within 6 months. BL exams may be same-month and cannot be guaranteed
to predate a clinical diagnosis. BioFIND remains BL-only and post-dose motor
measurements are not equated with PPMI motor totals.

`scripts/doctor_patient_env_v0.py` exposes `list_actions()` and `order_test()`.
These methods have no diagnosis-label API. This is an application-level boundary,
not operating-system/database security isolation; keep the private source DB protected.

Task framing: **retrospective evidence acquisition from recorded data**,
not a prospective, time-faithful diagnostic reconstruction or a training-ready
world model. Do not interpret unrecorded tests as negative results.

## Phase 10 — Patient split and availability-shortcut audit

**Prerequisite:** Phase 8 `pks_v2.duckdb` and Phase 9 `episodes_v0.sqlite` on the cluster.
This stage checks whether the **availability of tests alone** can predict
the PD/HC label and produces a **patient-level, local-only** split index.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase10_leakage_splits.py \
  --source-db ./outputs/patient_db/pks_v2.duckdb \
  --source-engine duckdb \
  --manifest-db ./outputs/episode_manifests/episodes_v0.sqlite \
  --split-db ./outputs/episode_splits/splits_v0.sqlite \
  --report-dir ./outputs/leakage_report \
  --seed 20261018
```

Outputs are **aggregate only**: `split_summary.csv`,
`action_availability_bias.csv`, `availability_only_baseline.csv`,
`anchor_by_label.csv`, `summary.json`. Upload those reports, **not**
`splits_v0.sqlite`, `episodes_v0.sqlite`, or `pks_v2.duckdb`.

PPMI is stratified by candidate label and availability of a baseline-month
anchor, with patient-level train/validation/test approximately 70/15/15.
BioFIND receives external_test only. The split is deterministic from the
seed; different seeds produce different splits. Do not regenerate or replace
the existing local split index when training experiments have begun.

Availability-only Bernoulli Naive Bayes is fitted on PPMI train and evaluated
on PPMI validation/test to expose shortcuts that rely on test-ordering or
collection patterns, rather than clinical evidence. These metrics are
diagnostics, not evidence of usable clinical diagnosis prediction.

Phase 10 also makes unavailable actions return generic `not_available` from
the Doctor-facing API, without exposing the source QC failure reason.
That does not remove availability as a possible shortcut; robust benchmarking
must explicitly compare the availability-only baseline.

## Phase 11 — Train-only protocol preflight (after Phase 10)

**Do not regenerate the Phase 10 split.** Phase 10 test metrics have already
been inspected and should be treated as exploratory. This preflight fits
availability-only Bernoulli models using PPMI TRAIN only, evaluated with
5-fold patient-level cross-validation. No validation, test, or BioFIND
predictions are computed.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase11_protocol_preflight.py \
  --source-db ./outputs/patient_db/pks_v2.duckdb \
  --source-engine duckdb \
  --manifest-db ./outputs/episode_manifests/episodes_v0.sqlite \
  --split-db ./outputs/episode_splits/splits_v0.sqlite \
  --report-dir ./outputs/protocol_preflight \
  --folds 5 --seed 20261018
```

Reports: `train_only_shortcut_cv.csv` (raw-action, collapsed-motor,
nonmotor-only baselines), `train_action_family_bias.csv`,
`frozen_split_counts.csv`, `summary.json`. Upload **only** the aggregate
`outputs/protocol_preflight/` folder, NOT any DB.

The collapsed-motor representation combines ON/OFF/UNSPECIFIED
UPDRS-III *availability*, not raw test scores and not clinical equivalence.
All clinical test values, incomplete timestamps, and candidate labels require
further validation before any deployment or scientific performance claim.

## Phase 12 — Fixed-budget, state-blind Doctor–Patient smoke benchmark

**Research status:** retrospective evidence-retrieval simulation only.
These checks are NOT diagnostic accuracy measurements.

This adds a `BudgetedDoctorPatientEnv` with a fixed, non-personalized
action catalog: age (cost 0), sleep item (1), cognitive test (1), autonomic
item (1), state-blind motor exam (2), smell test (2).
All actions must be explicitly requested; even an unavailable action costs
its advertised amount. Repeated requests have no additional cost or result.

Motor ON/OFF/UNSPECIFIED are hidden from the Doctor request/response;
the internal deterministic precedence is UNSPECIFIED > OFF > ON.
**This does NOT make motor scores physiologically equivalent** or erase
treatment confounding. It only removes the ON/OFF readiness shortcut.
All responses omit source paths, dates, patient ID, treatment state and QC
failure reasons. The patient never volunteers any test result.

Only PPMI TRAIN subjects from the existing locked Phase 10 split are replayed
under fixed clinical order, reverse order and seeded random order.
Outputs are aggregate cost, requested-action counts, returned-value counts
and unavailability, with no per-patient trajectories and no labels.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase12_smoke_benchmark.py \
  --source-db ./outputs/patient_db/pks_v2.duckdb \
  --source-engine duckdb \
  --manifest-db ./outputs/episode_manifests/episodes_v0.sqlite \
  --split-db ./outputs/episode_splits/splits_v0.sqlite \
  --report-dir ./outputs/phase12_smoke \
  --budget 5 --seed 20261018
```

Only upload `outputs/phase12_smoke/` reports, not the source DB,
episode manifest, split DB or raw files. Results do not establish
unbiased patient diagnosis or chronological evidence availability.

## Phase 13 — TRAIN-only numerical evidence and adaptive test ordering

This first diagnostic-policy proof of concept uses only PPMI TRAIN patients
from the existing Phase 10 fixed split, with five-fold inner cross-validation.
The input is the actual recorded clinical values returned by selected Phase 9
manifest observations; unavailable tests never update a label prediction.
The policy cannot see data before it makes a request.

Baselines: clinical fixed order, smell-first fixed order, seeded random order,
and adaptive posterior expected information gain per action cost.
All policies use a Laplace-smoothed, prespecified-binned Naive Bayes model
trained in each inner training fold, with a balanced label prior.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase13_numeric_policy_cv.py \
  --source-db ./outputs/patient_db/pks_v2.duckdb \
  --source-engine duckdb \
  --manifest-db ./outputs/episode_manifests/episodes_v0.sqlite \
  --split-db ./outputs/episode_splits/splits_v0.sqlite \
  --report-dir ./outputs/phase13_policy_cv \
  --folds 5 --budget 5 --seed 20261018
```

Aggregate-only reports: `policy_numeric_cv.csv`, `summary.json`.
Share those reports, not any patient-level DB.

WARNING: numeric score discretization, drug-state-blind motor value choice,
and candidate diagnosis labels have not been clinically validated.
No Phase 10 held-out val/test or BioFIND data are evaluated.
Existing train/test shortcuts and missingness remain scientific caveats.

## Phase 14 — Explain high numerical AUROC before improving Doctor

Phase 13 reached approximately 0.98 train-OOF AUROC from retrospective
clinical scores. Before promoting the result, examine which modalities
carry the signal and whether motor medication state or cohort labels
may be influencing the comparison.

Use the frozen Phase10 split, without accessing PPMI validation/test
or external BioFIND patients:

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase14_feature_attribution.py \
  --source-db ./outputs/patient_db/pks_v2.duckdb \
  --source-engine duckdb \
  --manifest-db ./outputs/episode_manifests/episodes_v0.sqlite \
  --split-db ./outputs/episode_splits/splits_v0.sqlite \
  --report-dir ./outputs/phase14_feature_ablation \
  --folds 5 --budget 5 --seed 20261018 --bootstraps 500
```

Outputs (aggregate-only): feature_ablation_cv.csv,
adaptive_vs_clinical_paired_bootstrap.csv, numeric_value_qc.csv,
numeric_bin_distributions.csv, selected_motor_state_by_label.csv, summary.json.

Important: feature ablations use ALL existing acceptable observations
in a subject (they are NOT cost-constrained Doctor policies). Paired
bootstrap uses identical PPMI train-fold subjects for adaptive and
clinical ordering. Do not view validation/test/external patients or
conclude clinical generalization from these exploratory results.
Never upload or commit private patient database, manifest, or split DB.

## Phase 15 — Motor sensitivity and cost-budget curves

Phase 14 showed that Motor-only discriminates retrospective PPMI candidate
PD/HC with AUROC ~0.97, while the Adaptive-minus-Clinical paired bootstrap
interval includes zero. Phase 15 examines budget dependence and motor-state
selection using only the previously frozen **PPMI TRAIN** split.

The comparisons include budgets 1, 2, 3, 4, 5, four request policies,
and two action spaces: motor allowed vs motor entirely excluded.
Two populations: all PPMI TRAIN patients and a selection-biased
motor-UNSPECIFIED-ready TRAIN subgroup. A subgroup is not a new test cohort.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase15_motor_budget.py \
  --source-db ./outputs/patient_db/pks_v2.duckdb \
  --source-engine duckdb \
  --manifest-db ./outputs/episode_manifests/episodes_v0.sqlite \
  --split-db ./outputs/episode_splits/splits_v0.sqlite \
  --report-dir ./outputs/phase15_motor_budget \
  --folds 5 --budgets 1 2 3 4 5 \
  --seed 20261018 --bootstraps 300
```

Outputs: `budget_policy_cv.csv`, `paired_policy_differences.csv`,
`population_counts.csv`, and `summary.json`. Only upload this aggregate
report folder. Never upload source, manifest or split databases.

All results remain exploratory: clinical labels, motor-condition
interpretation and true diagnostic timestamps have not been validated.

## Phase 16 — Static IG vs adaptive IG (PPMI TRAIN only)

Phase 15 showed large low-budget improvements versus fixed clinical order,
but Motor-only is already very discriminative. Phase 16 isolates dynamic
adaptation by introducing two stronger controls:

- `motor_first`: a fixed, deterministic order starting with Motor.
- `static_ig`: globally ranked information-gain-per-cost action order,
  fitted on each training fold without updating for patient responses.

Compare with fixed `clinical` and posterior-updating `adaptive_ig` at
budgets 1–5. Analyze full PPMI TRAIN only, 5-fold out-of-fold. Both
motor-allowed and motor-excluded action spaces are tested.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase16_static_adaptive.py \
  --source-db ./outputs/patient_db/pks_v2.duckdb \
  --source-engine duckdb \
  --manifest-db ./outputs/episode_manifests/episodes_v0.sqlite \
  --split-db ./outputs/episode_splits/splits_v0.sqlite \
  --report-dir ./outputs/phase16_static_vs_adaptive \
  --folds 5 --budgets 1 2 3 4 5 \
  --seed 20261018 --bootstraps 300
```

Reports: `policy_comparison_cv.csv`, `paired_adaptivity.csv`,
`requested_action_counts.csv`, `summary.json`. **Only upload the reports**,
never the source/manifest/split DBs. The analysis does NOT touch
PPMI validation/test or BioFIND.

Key interpretation: improvement over a weak fixed sequence does NOT
establish that adapting to a patient's responses adds value. Compare
`adaptive_ig` directly against `static_ig` and `motor_first`.

## Phase 17 — Should the Doctor STOP rather than spend the whole budget?

Phase 16 found that Adaptive IG and Static IG have nearly identical
diagnostic performance and dynamic order changes did not produce a
measurable advantage. Phase 17 therefore studies early stopping and
cost/utility instead of asserting an RSI contribution.

Run only on the frozen PPMI TRAIN set (5-fold OOF); no validation, test,
or BioFIND patient records are loaded. Compare Clinical, Motor-first,
Static IG, and Adaptive IG; budgets 2, 3, 5; motor allowed vs excluded.
Stop if uncalibrated Naive Bayes score confidence passes one of the
prespecified exploratory thresholds 0.70, 0.80, 0.90, 0.95, 0.99;
also run a no-stop reference. Stopping requires at least one valid
paid-test result.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase17_early_stopping.py \
  --source-db ./outputs/patient_db/pks_v2.duckdb \
  --source-engine duckdb \
  --manifest-db ./outputs/episode_manifests/episodes_v0.sqlite \
  --split-db ./outputs/episode_splits/splits_v0.sqlite \
  --report-dir ./outputs/phase17_stopping \
  --folds 5 --budgets 2 3 5 --seed 20261018
```

Outputs: `stopping_policy_cv.csv` (aggregated balanced accuracy, AUROC,
macro-F1, average cost, result count, stop rate, class-balanced Brier),
`summary.json`. Upload only the aggregated report folder, NOT DBs.

These are retrospective research simulations. The Naive Bayes posterior
is NOT calibrated; stop thresholds are score thresholds, not validated
medical confidence or safety decision rules. Candidate labels and
mixed motor drug states remain unresolved.

## Phase 18 — Nested confidence calibration for early stopping

Phase 17 suggests Static IG can reduce mean examination cost from 5.000
to 2.464 at raw confidence 0.90 with nearly identical balanced accuracy
but lower AUROC. A raw Naive Bayes score is NOT a calibrated probability.

Phase 18 uses only existing frozen PPMI TRAIN patients. Within each
outer training fold, split patients into proper model-fit and disjoint
calibration subsets; fit class-balanced weighted isotonic regression on
partial-score trajectories. Evaluate the resulting stop/no-stop policies
on the outer TRAIN-held-out fold, never on calibration subjects.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase18_nested_calibration.py \
  --source-db ./outputs/patient_db/pks_v2.duckdb \
  --source-engine duckdb \
  --manifest-db ./outputs/episode_manifests/episodes_v0.sqlite \
  --split-db ./outputs/episode_splits/splits_v0.sqlite \
  --report-dir ./outputs/phase18_calibration \
  --folds 5 --budget 5 --seed 20261018 --bootstraps 300
```

Aggregate reports only: `nested_stopping_metrics.csv`,
`paired_cost_risk_uncertainty.csv`, `nested_calibration_counts.csv`,
`summary.json`. Do not upload the private `.duckdb`/`.sqlite` files.

CAUTION: Prior Phase17 results already exposed the 0.90 threshold;
the new evaluation is exploratory, NOT a preregistered independent
confirmation. Isotonic regression uses class-balanced rather than
population-prevalence calibration. Neither score threshold is a
clinically justified safety threshold. Candidate diagnosis and
motor medication-state semantics remain unvalidated.

## Phase 19 — Select a stopping threshold without using outer evaluation patients

Phase 18 reports suggest ~54% mean cost savings with Motor at a calibrated
0.90 stopping score. However, that threshold was previously inspected,
and confidence calibration used only about 23–24 HC patients per fold.

This exploratory protocol runs exclusively on the locked **PPMI TRAIN**
population. In each outer CV iteration, partition its fitting patients into
**proper model fit**, **disjoint isotonic calibration**, and **disjoint
threshold selection** sets. Evaluate the chosen rule on the outer-held-out
TRAIN fold. The selection objective is minimum mean examination cost,
subject to a stratified paired-bootstrap upper 95% bound on balanced
accuracy loss of at most 0.02. If no threshold qualifies, do not stop.

**Safety guard:** calibrated confidence must point to the same class as
the original model log-odds, or that point cannot trigger stopping.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase19_risk_constrained_stop.py \
  --source-db ./outputs/patient_db/pks_v2.duckdb \
  --source-engine duckdb \
  --manifest-db ./outputs/episode_manifests/episodes_v0.sqlite \
  --split-db ./outputs/episode_splits/splits_v0.sqlite \
  --report-dir ./outputs/phase19_threshold_selection \
  --folds 5 --budget 5 --seed 20261018 \
  --max-balanced-accuracy-loss 0.02 --bootstraps 300
```

Reports: `selected_stopping_metrics.csv`,
`selected_vs_no_stop_paired.csv`, `inner_selection_by_fold.csv`,
`summary.json`. Share only aggregates; NEVER upload any patient-level
DB, patient manifest, split DB, clinical score or subject-level prediction.

IMPORTANT: the bootstrap bound is only an exploratory selection heuristic.
It is NOT a guarantee of patient safety or prospective diagnostic risk.
The same original PPMI TRAIN data were inspected in prior phases, so this
is not fully independent external confirmation.

## Phase 20 — Can PPMI support a longitudinal World Model target?

Phase 19 showed cost savings from stopping, but static PD-versus-HC
classification remains motor-dominated and lacks demonstrated adaptive
reordering gains. This phase audits a **different** candidate scientific
task: follow-up diagnosis code changes and repeated motor observations
among PPMI prodromal participants.

Run on the **original read-only CSVs** (the code/data directories are separate):

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase20_longitudinal_feasibility.py \
  --data-root /scratch/ylong66/data/pks/data \
  --output-dir ./outputs/phase20_longitudinal \
  --min-cell-count 10
```

Outputs: `longitudinal_feasibility.csv`, `prodromal_transition_summary.csv`,
`transition_time_bins.csv`, `raw_followup_code_counts.csv`,
`quality_flags.csv`, `report.md`, `summary.json`. Upload **only** these
aggregate reports. Do NOT upload CSV source, individual patient records,
or prior local databases.

Current definitions are provisional: enrollment `COHORT=4` with BL
`PRIMDIAG=25` for prodromal baseline; future visits with `PRIMDIAG=1`
strictly later by month count as diagnosis-**code transition candidates**.
This is NOT clinician-adjudicated PD conversion, and patients missing
follow-up must not be classified as stable. Month-only timestamps make
within-month order unknowable. This audit is not a survival model or a
longitudinal training dataset. Consult official source documentation
before defining a target, time-to-event or censoring semantics.

## Phase 21 — Longitudinal target readiness and censoring preflight

Phase20 found 1,874 strict prodromal baseline subjects, 1,025 with
a dated later diagnosis visit, and 95 with a later candidate PD diagnosis code.
Those 95 are **not** confirmed conversion events, and missing follow-up
must not be treated as stable.

Phase21 distinguishes at 12, 24, 36 months: later PD code observed by
horizon, follow-up documented through horizon without an earlier PD code,
and insufficient follow-up. It also checks whether same-medication-state
motor scores are observable at baseline and later months without averaging
ON and OFF exams or conflicting within-month scores.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase21_longitudinal_target_readiness.py \
  --data-root /scratch/ylong66/data/pks/data \
  --output-dir ./outputs/phase21_target_readiness \
  --min-cell-count 10
```

Aggregate-only reports: `candidate_conversion_support.csv`,
`diagnosis_observability_horizons.csv`, `longitudinal_motor_by_state.csv`,
`baseline_group_counts.csv`, `quality_flags.csv`, `summary.json`.
Upload only that directory, NEVER any raw source or patient-level database.

These are feasibility/observability categories, **not** validated
PD conversion labels or survival analysis. The codebook definition,
follow-up schedule, adjudication and censoring process must be checked
before defining a longitudinal training task. Within-calendar-month
event order is unknowable from MM/YYYY alone.

## Phase 22 — Strict baseline/follow-up motor pairs

Phase21 showed promising longitudinal motor coverage, but its audit could
count `EVENT_ID=BL` motor records with an exam month later than the BL anchor
as if they were follow-up. Phase22 eliminates that ambiguity before any
new longitudinal training manifest is created.

Requirements: (1) Motor baseline must have `EVENT_ID=BL` and the exam
month must exactly match `Baseline_Visit_Start` month. (2) Later motor
records must be associated with an actual non-BL/non-SC visit and a
strictly later exam month. (3) A patient/month/medication-state cell
containing conflicting totals is excluded. (4) ON, OFF and UNSPECIFIED
must never be averaged together, and UNSPECIFIED is not proof of stable
treatment. (5) For target month 6±2, 12±3, and 24±4 choose the closest
unique valid month; reject equally close alternatives.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase22_strict_motor_pairs.py \
  --data-root /scratch/ylong66/data/pks/data \
  --output-dir ./outputs/phase22_strict_motor_pairs \
  --min-cell-count 10
```

Only upload the aggregate outputs: `strict_pair_window_counts.csv`,
`motor_state_quality.csv`, `aggregate_motor_change.csv`,
`baseline_group_counts.csv`, `quality_flags.csv`, `summary.json`.
Raw PPMI files and local patient databases remain on the cluster.

These outputs establish pair feasibility only, not treatment-controlled
progression trajectories or a training-ready World Model. Real-world
follow-up and medication-state confounding remain to be addressed.

## Phase 23 — Private prodromal 12-month Motor manifest

Fetch Phase 23 from this branch; manual zip upload is no longer needed.
Input data live separately at `/scratch/ylong66/data/pks/data`.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase23_manifest.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase23_prodromal_manifest \
  --seed 20261018
```

Split the complete Prodromal baseline population 70/15/15 before
selecting observed 12±3-month Motor targets. 6±2 and 24±4 months are
sensitivity windows. The local SQLite manifest contains patient IDs
and individual scores including **future** target values. It must
stay on the research cluster and must NEVER be uploaded or committed.
Only share the aggregate reports in `outputs/phase23_prodromal_manifest`.

Do not recreate an existing manifest under the same filename; the script
refuses to overwrite it. UNSPECIFIED medication-state records are not
verified treatment-matched measurements. Longitudinal clinical code
definitions and informative follow-up remain unresolved.

## Phase 24 — 12-month Motor regression baselines (TRAIN only)

Phase23 froze 1,874 Prodromal PPMI patients before conditioning on follow-up:
Train 1,312 (586 paired at 12±3 months), validation 281 (130 paired),
test 281 (129 paired). Phase24 reads ONLY the 586 eligible TRAIN target
scores and performs patient-level five-fold out-of-fold evaluation.

Baselines: persistence (next NP3TOT = baseline score), training-fold mean
delta, training-fold median delta, and ridge-regularized linear delta.
Report MAE, RMSE, mean signed error, baseline severity stratification,
and paired bootstrap MAE differences versus persistence. Future visit
month is used only to establish the pre-existing 12±3-month target;
it is NEVER a baseline model feature. Validation/test targets remain sealed.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase24_motor_baselines.py \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase24_motor_baselines \
  --folds 5 --seed 20261018 --bootstraps 500
```

Aggregate outputs: `train_oof_motor_metrics.csv`, `paired_vs_persistence.csv`,
`severity_stratified_train_errors.csv`, `train_cv_fold_counts.csv`,
`summary.json`. Share only the aggregate reports. Do NOT upload any
longitudinal manifest SQLite file, row-level scores, clinical records or
subject-level predictions.

CAUTION: UNSPECIFIED medication codes cannot prove matched examination
conditions. Complete-case follow-up can be informatively missing.
This is an exploratory recorded-score prediction benchmark, NOT
clinically validated progression forecasting or a World Model yet.

## Phase 25 — Baseline multimodal feature preflight (train only)

Phase24: among 586 eligible prodromal TRAIN pairs, persistence had MAE
3.08362 / RMSE 4.65701; the ridge-delta baseline had MAE 3.11974 /
RMSE 4.45060. No simple baseline improved MAE over persistence.

Before fitting a multimodal longitudinal predictor, Phase25 checks
baseline-only availability of age, MoCA, RBD item, SCOPA-AUT item
and UPSIT, with numeric QC and repeat conflict rejection.
Only the locked Phase23 TRAIN patient IDs are queried from Phase8's
PPMI clinical DB. BL exams must have a month matching the BL anchor.
UPSIT may use SC only if BL is absent AND the SC month is 1..6 months
before BL. The observed/unobserved 12-month target flag is used only
for **aggregate selection-bias assessment**, never as a predictor.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase25_baseline_feature_audit.py \
  --source-db ./outputs/patient_db/pks_v2.duckdb \
  --source-engine duckdb \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase25_baseline_features \
  --min-cell-count 10
```

Aggregate reports: `train_feature_status_by_followup.csv`,
`train_joint_coverage.csv`, `train_feature_value_summaries.csv`,
`summary.json`. Upload only these files, NOT individual patient records
or the private DBs.

NOTE: questionnaire single-item numeric ranges are provisional;
future 12-month outcomes are not queried, and **val/test remain unopened**.
Phase25 is a feature feasibility audit, not a trained World Model.

## Phase 26 — TRAIN-only multimodal prodromal Motor prediction

Phase25's 586 TRAIN patients with a 12-month Motor target had baseline
MoCA 578/586, RBD 577/586, SCOPA-AUT 577/586, UPSIT 296/586,
and age 0/586 under the strict month rules. Age is EXCLUDED pending
a separate source-date audit. UPSIT is evaluated as an optional
feature with proper-fit-fold median imputation and no missingness flags.

Models use identical Phase24 TRAIN patient folds: persistence, Phase24
motor-only ridge, and fixed-penalty multivariable ridge for (1) motor+MoCA,
(2) motor+MoCA+RBD+SCOPA, (3) motor+MoCA+RBD+SCOPA+UPSIT.
All normalization and missing value imputation are fitted on each
inner training fold only. No future month-gap, followup readiness
or post-BL observation is used as a model input. All testing is
TRAIN out-of-fold, not final validation/test.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase26_multimodal_oof.py \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --source-db ./outputs/patient_db/pks_v2.duckdb \
  --source-engine duckdb \
  --report-dir ./outputs/phase26_multimodal_oof \
  --folds 5 --seed 20261018 --bootstraps 500
```

Share only aggregated reports `multimodal_train_oof.csv`,
`multimodal_paired_vs_persistence.csv`,
`paired_train_feature_coverage.csv`,
`age_upsit_date_provenance.csv`, `train_cv_fold_counts.csv`,
and `summary.json`.

Do NOT upload SQLite/DuckDB databases. PPMI source coding, actual
medication state and informative followup remain unvalidated.

## Phase 27 — Fair incremental-value test of multimodal baseline features

Phase26 (586 Prodromal TRAIN patients, 12±3 months) found no MAE or
RMSE improvement from adding MoCA, RBD, SCOPA or UPSIT over the
motor-only ridge baseline. Phase27 tests this directly using identical
ridge implementations and nested regularization selection.

Models: persistence, motor-only ridge, motor+MoCA, motor+MoCA/RBD/SCOPA,
and motor+MoCA/RBD/SCOPA/UPSIT. Outer 5 folds are identical to Phase24.
Each outer fit fold performs an inner 3-fold RMSE search over fixed ridge
penalties (0.01, 0.1, 1, 10, 100); imputation and standardization are
re-fitted inside each proper training subset. Outer held-out TRAIN patients
are evaluated once per model, then paired bootstrap contrasts directly
compare multimodal models versus motor-only ridge and persistence.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase27_fair_incremental.py \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --source-db ./outputs/patient_db/pks_v2.duckdb \
  --source-engine duckdb \
  --report-dir ./outputs/phase27_fair_incremental \
  --folds 5 --seed 20261018 --bootstraps 500
```

Only share aggregate reports: `nested_multimodal_oof.csv`,
`incremental_paired_differences.csv`, `inner_penalty_selection.csv`,
`severity_stratified_errors.csv`, `train_feature_status.csv`,
`outer_cv_fold_counts.csv`, `summary.json`.

Age remains excluded pending provenance validation. All patient-level
databases stay on the cluster. This is exploratory within the same
previously inspected TRAIN population, not independent confirmation.

## Phase 28 — TRAIN-only longitudinal signal and multistep feasibility

Phase27 showed no stable incremental benefit from MoCA, RBD, SCOPA or UPSIT
over a motor-only ridge model at 12±3 months. Before fitting another model,
Phase28 inspects the existing frozen Phase23 TRAIN manifest to determine
how much usable change and multi-step trajectory structure it contains.

Reports cover: 12-month delta score distribution; baseline severity and
follow-up observability; actual target-month gaps; overlap among 6-,12-
and 24-month observed targets; direction of change BL→12 versus 12→24.
Readiness and future observations are used only for aggregate auditing,
NOT baseline Agent input. Val/test remain unopened.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest discover -s tests -v
python scripts/phase28_longitudinal_signal.py \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase28_longitudinal_signal \
  --min-cell-count 10
zip -r pks_phase28_report.zip outputs/phase28_longitudinal_signal
```

Only upload the `pks_phase28_report.zip` aggregate report, NEVER
the private longitudinal SQLite manifest or individual patient rows.

Unspecified medication state is not proof of controlled treatment.
Large or reversing score changes may reflect measurement variability,
regression to mean or true change; this audit alone cannot separate them.
Window-matched measures are not guaranteed exact monthly sampling.


## Phase 30 — Smoothing versus state transition

Train-only PPMI Prodromal 12-to-24 Motor forecast, using the same 332 subjects and outer folds as Phase29. Compare 12-month persistence, 50/50 BL+12-month smoothing, outer-fit-selected convex blend, last-value Ridge and two-visit Ridge. Inspect completeness selection among all TRAIN subjects with 12-month targets. Never upload the private manifest.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest tests.test_phase30_smoothing -v
python scripts/phase30_smoothing.py --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite --report-dir ./outputs/phase30_smoothing --folds 5 --seed 20261018 --bootstraps 500
zip -r pks_phase30_report.zip outputs/phase30_smoothing
```

Upload only aggregate Phase30 reports. Model improvement from two observations could be smoothing of measurement variability rather than disease dynamics.


## Phase 31 — TRAIN-only repeated Motor-record audit

Phase30 showed that averaging BL and 12-month Motor scores improves both MAE and RMSE relative to last-value persistence, while the two-visit Ridge model was not better on both metrics. This does **not** establish that observation noise is the cause. Phase31 classifies raw PPMI MDS-UPDRS III duplicated patient-month keys as exact repeated rows, distinct visits with equal scores, conflicting visits with different scores, conflicting values within an event, or invalid values. Baseline requires `EVENT_ID=BL` and the same BL anchor month; later visits require a non-BL event and strictly later exam month. Only Phase23 TRAIN participants are processed.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest tests.test_phase31_motor_repetitions -v
python scripts/phase31_motor_repetitions.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase31_motor_repetitions \
  --min-cell-count 10
zip -r pks_phase31_report.zip outputs/phase31_motor_repetitions
```

Upload only aggregate reports. Never upload the private manifest, identifiers, clinical scores, dates, or raw data. This audit does not reconcile the frozen manifest or infer test-retest measurement error from different exams.


## Phase 32 — Reconcile Phase23 exclusions with Phase31 raw records

Phase31 showed that zero TRAIN keys had different scores from repeated
visits, while invalid/missing NP3TOT dominates the 6-month window.
Phase23's 583 excluded keys counted *all* Prodromal splits; Phase31 used
TRAIN only and aggregated by target window. Phase32 reproduces exactly
the Phase23 per-patient BL/FU/month key and verifies the frozen TRAIN
manifest without rewriting it. It reports missing-only, sentinel, out-of-
range, invalid+valid and genuine repeated-score categories, with source
visit-category frequencies. Any nonzero target/readiness mismatch blocks
interpreting Phase23-derived modeling results until resolved.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest tests.test_phase32_exclusion_forensics -v
python scripts/phase32_exclusion_forensics.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase32_exclusion_audit \
  --min-cell-count 10
zip -r pks_phase32_report.zip outputs/phase32_exclusion_audit
```

Upload **only** the aggregate Phase32 zip, not source CSVs or patient
manifest. This script does not change the previously frozen split or targets.


## Phase 33 — Why are 6-month Part III totals blank?

Phase32 verified **zero** mismatches to the frozen Phase23 TRAIN manifest: 34/586/377 targets at 6/12/24 months, respectively. The 410 excluded TRAIN patient-month keys all had missing `NP3TOT` rather than conflicting numeric scores; 286 of these were in the 6-month window. This read-only preflight reports privacy-suppressed literal source `EVENT_ID` frequencies, valid/empty Motor totals and the number of populated `NP3*` item fields when the total is missing. It does **not** reconstruct totals or alter the split.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest tests.test_phase33_visit_audit -v
python scripts/phase33_visit_audit.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase33_visit_audit \
  --min-cell-count 10
zip -r pks_phase33_report.zip outputs/phase33_visit_audit
```

Only upload the aggregated report ZIP; never upload patient IDs, scores, source CSVs or the private manifest. The audit does not validate PPMI event codes or test-retest noise.


## Phase 34 — R01 remote Motor missing-total investigation

Phase33 revealed all 322 six-month source records in PPMI TRAIN were `R01`, with 286 empty `NP3TOT`. The generated summary's `NP3_item_columns=6` is a **bug** in Phase33's report metadata (variable shadowing), since `10plus` populated item fields were recorded for those rows; Phase33 has been patched and should be rerun to validate item counts. The PPMI NUPDRS3 documentation explains that remote visits can mark five rigidity scores and postural stability as `UR` (unable to rate). Phase34 inspects these six item codes, restricted to frozen TRAIN patients and blank-total `R01`/`V04`/`V06` rows, and exports aggregate counts only.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest tests.test_phase33_visit_audit tests.test_phase34_remote_motor -v
python scripts/phase33_visit_audit.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase33_visit_audit_corrected
python scripts/phase34_remote_motor.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase34_remote_motor
zip -r pks_phase34_report.zip outputs/phase33_visit_audit_corrected outputs/phase34_remote_motor
```

Upload ONLY the aggregate zip. Do not reconstruct or replace missing NP3TOT or alter the frozen manifest based on these findings. Source documentation: https://blackfynn-ppmi2-curation.readthedocs-hosted.com/en/stable-documentation-branch/documentation_files/motor_assessments/tables/NUPDRS3.html .


## Phase 35 — Confirm numeric 101 at remote R01 and longitudinal coverage

The corrected Phase33 summary confirms 33 NP3 item-like columns. Phase34 found that among 286 six-month R01 rows with blank NP3TOT, the five rigidity fields and postural-stability field almost always contain a non-0-to-4 code, **not** literal `UR`. Phase35 checks how often those fields are numeric `101` (unable to rate in the PPMI paper CRF), counts other NP3 item presence, and audits whether the same TRAIN patients later have a valid in-person V04 total. The later V04 is NOT a paired same-day reference assessment.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest tests.test_phase35_remote_101 -v
python scripts/phase35_remote_101.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase35_remote_101 \
  --min-cell-count 10
zip -r pks_phase35_report.zip outputs/phase35_remote_101
```

Only upload aggregate Phase35 report files. Never upload PPMI source CSV, private manifest, individual values, or identifiers. This does not reconstruct totals or validate a clinical agent.


## Phase 36 — Same-visit masked in-person Motor benchmark feasibility

Phase35 confirms 274/286 blank-total R01 records have numeric 101 in all six hands-on items, and 265/286 have 20+ additional numeric item fields. Importantly, 227 TRAIN patients have such a partial remote exam followed months later by a valid V04 clinic total. The later V04 score is **NOT** same-visit ground truth for R01.

To test whether a simulated remote Agent can infer missing items without temporal leakage, Phase36 first audits original V04 on-site Part III exams with all 33 clinical item scores and checks whether the 33-item sum equals the recorded NP3TOT. Those eligible exams may be masked at the **same visit** (hide the six remote-unobservable items) in a future patient-level local benchmark. Actual R01 remote exam visible-item distribution is separately audited for domain shift; no imputations are performed by Phase36.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m unittest tests.test_phase36_masked_site_feasibility -v
python scripts/phase36_masked_site_feasibility.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase36_masked_site \
  --min-cell-count 10
zip -r pks_phase36_report.zip outputs/phase36_masked_site
```

Upload only the aggregate report ZIP, not PPMI source CSVs or patient manifest. Phase36 is TRAIN-only, descriptive, and does not assert any remote-to-onsite simultaneous truth pairs. The 33-item sum check must be passed before creating any local patient-level masked benchmarking episodes.


## Phase 37 — Train-only same-visit masked remote Motor benchmark

Phase36 identified **605** frozen Prodromal TRAIN participants with a complete V04 onsite MDS-UPDRS Part III exam: all 33 item scores are numeric, and the item sum exactly equals the recorded `NP3TOT`. Among them, the six remotely unobservable items sum to zero for **399 (66.0%)**, making an always-zero hidden-score baseline essential. A separate 229 real R01 partial exams have 27 visible items, and 199 of these patients later have an eligible V04, but those are NOT same-day labels. Domain shift between R01 and V04 must be investigated before real-world application.

Phase37 builds an **in-memory, TRAIN-only** supervised experiment using the 605 onsite exams. It simulates remote examination by masking six same-visit items; models only see the 27 visible items and predict the six hidden items' sum (0..24). Compare always-zero, proper-fold hidden-score mean, visible-sum ridge, and 27-visible-item ridge with inner penalty tuning and patient-level outer CV. The recorded full total and hidden sum are **labels**, never input predictors. Primary metric is hidden-sum MAE, with RMSE and nonzero-hidden AUROC secondary. No validation/test patients or future visits enter the task.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m py_compile scripts/phase37_masked_site_benchmark.py
python -m unittest tests.test_phase37_masked_site_benchmark -v
python scripts/phase37_masked_site_benchmark.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase37_masked_site_benchmark \
  --folds 5 --seed 20261018 --bootstraps 500 --min-cell-count 10
zip -r pks_phase37_report.zip outputs/phase37_masked_site_benchmark
```

Only upload the aggregate Phase37 ZIP. The reported CV results are exploratory and **do not validate real R01 performance**, examination costs, or clinical safety. Keep all patient-level PPMI files and manifests on the research cluster.

## Phase 38 — Budgeted same-visit simulated acquisition (TRAIN only)

Phase37 among 605 TRAIN V04 exams: always-zero hidden-six MAE 0.77851,
visible-27 Ridge MAE 0.84765, but Ridge RMSE 1.34707 vs 1.69929 for
always-zero. Hidden six sum >0 in 206/605 patients; Ridge-visible-27
did not clearly exceed Ridge-visible-sum. Next ask whether an observed
risk score can select whom to send for extra examination.

Phase38 simulates fixed budgets 0/10/20/30/40/50/100% within each
held-out TRAIN outer fold. Acquisition perfectly reveals the hidden
six items from the SAME V04 visit; unselected patients retain zero.
Compare random, visible 27-item sum, held-out Ridge-visible-sum and
held-out Ridge-visible-27 risk ranking. A perfect-label oracle is an
UNATTAINABLE upper bound, not a deployable policy. Report residual
hidden-sum MAE, hidden-positive recall and paired bootstrap contrasts.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m py_compile scripts/phase38_budgeted_acquisition.py
python -m unittest tests.test_phase38_budgeted_acquisition -v
python scripts/phase38_budgeted_acquisition.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase38_budgeted_acquisition \
  --folds 5 --seed 20261018 --bootstraps 500 --min-cell-count 10
zip -r pks_phase38_report.zip outputs/phase38_budgeted_acquisition
```

This is an IDEALIZED same-V04 masking experiment, not real remote R01
referral, actual onsite costs, patient safety, or prospective validation.
No Phase23 validation/test subjects are accessed, and labels are never
used for deployable acquisition rankings. Upload ONLY the aggregate
Phase38 ZIP; keep patient-level files on the cluster.

## Phase 39 — Does V04 ranking transfer to real R01 observation distributions?

Phase38 shows visible-total ranking consistently beats one seeded random
acquisition at fixed 10–50% V04 masking budgets, but the 27-item Ridge
ranker does not materially beat visible total. Before simulating further
Doctor policies, audit transportability: real R01 observations and
same-visit-masked V04 observations can have different distributions.

Phase39 compares the 27 visible clinical items and their sum in eligible
R01 (six UR, 27 valid) versus eligible V04 (all 33 valid; total verified).
It reports mean differences, descriptive standardized differences and
two-sample KS distances. For patients present at both visits, report
R01-to-later-V04 visible-score changes (NOT same-time clinical truth).
Finally transfer each V04 visible-total budget cutoff to R01 unchanged
and report the resulting selection rate and paired selection switches.
The cutoff includes all tied scores; realized V04 selection fraction
can be larger than nominal budget. It DOES NOT estimate R01 clinical
benefit because R01 hidden-six outcomes are not observed.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m py_compile scripts/phase39_remote_domain_shift.py
python -m unittest tests.test_phase39_remote_domain_shift -v
python scripts/phase39_remote_domain_shift.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase39_domain_shift \
  --bootstraps 500 --seed 20261018 --min-cell-count 10
zip -r pks_phase39_report.zip outputs/phase39_domain_shift
```

Share only aggregate Phase39 report ZIP. Never upload PPMI source data,
patient-level longitudinal manifests, or per-subject decisions.

## Phase 40 — Exact-budget remote ranking calibration (TRAIN only)

Phase39 shows 229 eligible real remote R01 exams, 605 complete V04
onsite exams and 199 patients observed at both visits. Mean visible
27-item score is 3.459 at R01 versus 4.782 at V04 (SMD -0.271).
V04 score cutoffs transfer poorly to R01; a nominal 20% V04 budget
selects only 13.54% of R01, while ties inflate onsite selection.

Phase40 separates score ranking from exact capacity. It selects EXACT
top-k patient counts independently among R01 and V04 at 10/20/30/40/50%
budgets, breaking ties using stable seed-based SHA256 order. It audits
ties at the boundary and compares the V04 threshold transfer against
exact R01 top-k selection. For 199 patients with both R01 and later V04,
it computes tied Spearman rank correlation (bootstrap interval) and
top-k overlap/switching. This is a ranking stability test, NOT proof
that remote referrals have benefit: R01 hidden-six outcomes are unobserved.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m py_compile scripts/phase40_budget_calibration.py
python -m unittest tests.test_phase40_budget_calibration -v
python scripts/phase40_budget_calibration.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase40_budget_calibration \
  --bootstraps 500 --seed 20261018 --min-cell-count 10
zip -r pks_phase40_report.zip outputs/phase40_budget_calibration
```

Only send the aggregate ZIP; raw PPMI data, private SQLite, identifiers,
individual ranks, decisions and visit scores must remain on the cluster.

## Phase 41 — Observational future onsite risk from real R01 (TRAIN only)

Phase40 found that exact budgets are feasible, but patient risk ranks
change over the six-month R01-to-V04 interval: Spearman 0.5992
(bootstrap 95% 0.4901–0.6909); 20% top-k selection retained 22/40
patients across exams, not a stable identical patient list.

Phase41 is a DIFFERENT, observable research endpoint: only use a
patient's 27-item VISIBLE score from actual earlier remote R01 to rank
risk that the SIX hidden items at their LATER V04 clinic exam are
nonzero. This is NOT the missing same-visit R01 ground truth, a clinical
referral effect, or a causal value-of-information estimate.
Only 199 or fewer paired TRAIN patients with verified V04 item sums
may be eligible. Compare R01 visible-score ordering against repeated
random exact-quota rankings, at budgets 10/20/30/40/50%. Provide
tie-seed sensitivity, patient bootstrap intervals for recall lift over
random-budget expectation, nonzero AUROC/AP, outcome prevalence
by R01 score stratum, and selection into complete V04 followup.
V04 visible-score and full hidden-six outcome rankings are shown as
RETROSPECTIVE references ONLY, never as R01 policy inputs.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m py_compile scripts/phase41_future_v04_risk.py
python -m unittest tests.test_phase41_future_v04_risk -v
python scripts/phase41_future_v04_risk.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase41_future_risk \
  --seed 20261018 --bootstraps 500 \
  --random-repeats 500 --tie-repeats 100 --min-cell-count 10
zip -r pks_phase41_report.zip outputs/phase41_future_risk
```

Only upload the aggregated Phase41 ZIP, not PPMI raw data, private
longitudinal SQLite, individual predictions, clinical labels or IDs.

## Phase 42 — Full R01 cohort bounds under missing future V04 labels

Phase41 used 199 TRAIN patients with complete real R01 to later V04
observations and showed real R01 visible-total AUROC 0.68055, AP
0.49334 for a FUTURE V04 nonzero six-item score. At a 20% budget
among complete cases, 22/58 future-positive people were selected
(37.93%) versus about 20.13% with repeated random selection. But
30/229 eligible R01 subjects lack a qualifying V04 future label.

Phase42 re-ranks ALL 229 R01 subjects at fixed 10/20/30/40/50% exact
budgets with Phase41's identical visible-total score and deterministic
hash tie breaks. An unobserved future outcome remains UNKNOWN, never
imputed as zero. The output provides sharp identification bounds for
precision@k and recall@k across all assignments of the missing binary
labels, plus tie-seed sensitivity and later-V04 observability by
R01 score. These are NOT sampling confidence intervals or a causal
estimate of referral benefit. Future labels are for months-later V04,
not R01 same-day hidden scores.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m py_compile scripts/phase42_missing_followup_bounds.py
python -m unittest tests.test_phase42_missing_followup_bounds -v
python scripts/phase42_missing_followup_bounds.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase42_missing_followup_bounds \
  --seed 20261018 --tie-repeats 100 --min-cell-count 10
zip -r pks_phase42_report.zip outputs/phase42_missing_followup_bounds
```

Only share the aggregate Phase42 report ZIP. Never upload raw PPMI
records, patient-level followup labels, decisions, or SQLite manifest.

## Phase 43 — ONE-SHOT frozen validation of real R01 future-risk policy

Phase42 on all 229 PPMI Prodromal TRAIN R01 patients showed exact
20% top-k worst-case future V04 six-positive recall of 26.19%, above
the 20.09% random-quota expectation, even if all 30 missing V04
outcomes are adversarially assigned. Across 100 different tie seeds,
the minimum worst-case recall was 24.71%. This is an identification
bound, not a sampling confidence interval or causal policy benefit.

STOP further tuning on TRAIN. The Phase43 validation protocol was
precommitted in docs/phase43_frozen_validation_protocol.md BEFORE
validation scores were accessed. It locks a simple, no-fit policy:
R01 visible 27-item sum descending; exact 20% quota; SHA256 tie
break with seed 20261018. Primary statistic is worst-case recall
lower minus random expected recall among ALL eligible VAL R01 subjects,
allowing every missing future V04 outcome to be positive or negative.
Extra budgets (10/30/40/50%) and labeled AUROC/AP are descriptive only.
Only frozen PPMI VAL patients are read; TEST remains unopened.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m py_compile scripts/phase43_locked_validation.py
python -m unittest tests.test_phase43_locked_validation -v
python scripts/phase43_locked_validation.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase43_locked_validation
zip -r pks_phase43_report.zip outputs/phase43_locked_validation
```

Upload ONLY the aggregate Phase43 report ZIP. Do NOT tune based on
this validation result or open TEST. This remains observational risk
stratification about later V04 scores, not evidence that an onsite
referral at R01 benefits patients.

## Phase 44 — Pre-registered final TEST (do not run casually)

Phase43 one-shot Validation had 49 eligible remote R01 patients,
41 with known future onsite V04 label, and 13 future six-positive.
Locked 20% R01 top-k selected 10 of 49. Missing-outcome worst-case
recall was 25.00%, versus 20.41% expected for random allocation
(directional primary +4.59 percentage points). Complete-case AUROC
was 0.61401 and AP 0.42286. These are small-sample observational
results, not significance or causal clinic-referral evidence.

Final TEST was NOT used to choose any strategy. A frozen, unchangeable
one-shot Phase44 protocol is in docs/phase44_final_test_protocol.md.
The evaluator is a strict copy of Phase43's eligibility, fixed 20%
budget and risk-score hash tie policy, except it accesses ONLY original
Phase23 split='test'. It rejects invocation without the explicit
--confirm-final-test flag and refuses an already existing report path.

Only after deciding to permanently consume the final TEST split:

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m py_compile scripts/phase44_locked_final_test.py
python -m unittest tests.test_phase44_locked_final_test -v
python scripts/phase44_locked_final_test.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir ./outputs/phase44_locked_final_test \
  --confirm-final-test
zip -r pks_phase44_report.zip outputs/phase44_locked_final_test
```

Do not use TEST to retune ranking, budgets, or model choice; report
negative/inconclusive results as such. No patient-level data exported.

## Frozen research record through Phase 44 (2026-10-19)

**Do not retune after held-out final testing:** the original Phase44
TEST split is already consumed. The evidence ledger and final policy:

- [Verified Phase23–44 results, assumptions and locked 20% policy](docs/RESEARCH_SNAPSHOT_PHASE23_44.md)
- [Reproduction guide and safe metadata commands](docs/phase23_44_reproduction.md)
- [Original private report ZIP SHA256 manifest](docs/phase23_44_report_index.csv)
- [Architecture decision: multi-agent, World Model and RSI](docs/ADR_AGENT_WM_RSI.md)
- [Precommitted VAL policy](docs/phase43_frozen_validation_protocol.md)
- [Precommitted final TEST policy](docs/phase44_final_test_protocol.md)

Original aggregate report ZIPs and all subject-level data remain outside
the public repository. Report-level percentages may disclose small cells;
do not copy raw report ZIPs into GitHub.

## Phase45: WM + Doctor/Patient Agent + RSI-lite experimental pilot

**New TRAIN-only CPU runnable baseline**, with no Hugging Face model
download, no GPU requirement and no access to consumed Phase44 TEST.
The model is a same-visit categorical clinical-observation generator,
not yet a disease-progressing temporal World Model. RSI-lite explores
and verifies candidate planning/WM configurations on inner TRAIN dev
folds with rollback; separate outer TRAIN folds evaluate outcomes.

- [Design, scripts, cluster commands, offline model caching](docs/PHASE45_WM_RSI_PILOT.md)
- Implementation: `scripts/phase45_world_model.py` and
  `scripts/phase45_wm_rsi_agents.py`
- Test: `tests/test_phase45_wm_rsi_agents.py`
- Dusky CPU: `sbatch slurm/phase45_dusky_cpu.sbatch`
- Nibi GPU fallback: `sbatch slurm/phase45_nibi_gpu_optional.sbatch`
  (GPU remains unused by V0; CPU partition is strongly preferred)
- Optional future model prefetch on networked login nodes:
  `scripts/prefetch_optional_agent_models.py`

Only aggregate reports are returned, always stored under
`/scratch/ylong66/pks_agent_outputs/`. Patient-level source files
and prior frozen manifest remain private.


## Phase46 — Real Qwen3 Doctor backbone (GPU, offline)

Phase45 has no LLM; Phase46 introduces actual local Qwen3-4B-Instruct-2507 generation in a constrained Doctor tool-calling interface. The Patient evidence service and observation WM remain grounded/non-LLM. Compare `llm_plain` vs `llm_wm` against equal-question non-LLM policies on TRAIN-only simulated V04 exams. This is not yet LLM-RSI or a temporal WM. **Never use the Phase44 consumed final TEST again.**

- [Complete login-node download and GPU Slurm instructions](docs/PHASE46_QWEN_DOCTOR.md)
- Login node: `python scripts/prefetch_optional_agent_models.py --kind doctor` (downloads to `/scratch/ylong66/models/`)
- Test: `python -m unittest tests.test_phase46_qwen_doctor -v`
- Dusky L40S: `sbatch slurm/phase46_dusky_qwen_doctor.sbatch`
- Nibi H100 MIG: `sbatch slurm/phase46_nibi_qwen_doctor.sbatch`

No model downloads are attempted on network-isolated compute nodes.

## Phase46 Conda repair for old glibc compute nodes

The initial Dusky Phase46 Qwen job failed on incompatible ms127
torch / transformers versions and a safetensors pip binary linked
to glibc 2.18. **Leave ms127 untouched.** New isolated GPU
environment prefix is `/scratch/ylong66/conda/envs/pks_qwen46`.

- [Complete conda-first installation and troubleshooting](docs/PHASE46_CONDA_GPU_FIX.md)
- Conda environment definition: `envs/phase46_qwen_conda.yaml`
- On login node: `bash scripts/setup_phase46_conda.sh`
- On GPU compute job: `scripts/phase46_env_check.py --mode gpu`
- Updated Dusky/Nibi Phase46 Slurm jobs automatically use new env.


### Phase46 legacy glibc CUDA11.8 repair (2026-10-19)

The first new CUDA12.1 environment imported neither Torch nor Transformers because its NVIDIA `libcurand.so.10` required `GLIBC_2.27` on glibc 2.17 Dusky. It is **not fixed by just overriding Conda's glibc solver variable**. We now recommend a **separate pinned NVIDIA CUDA 11.8** Conda environment on scratch, while keeping `ms127` and the failed environment untouched. [Read updated fix](docs/PHASE46_CONDA_GPU_FIX.md).

From a networked login node:

```bash
cd /home/ylong66/workspace/pks_agent
CONDA_EXE="$(conda info --base)/bin/conda" bash scripts/setup_phase46_conda_cu118.sh
```

Only after `torch` imports correctly, submit `slurm/phase46_dusky_qwen_doctor.sbatch` or `slurm/phase46_nibi_qwen_doctor.sbatch`; both now run offline from `/scratch/ylong66/conda/envs/pks_qwen46_cu118/`.


## Phase47: Qwen Doctor 4/8/12/16 query-budget curve

The 100-patient TRAIN Phase46 q=4 run found Qwen-only recall 12/29 and AUROC 0.7438, versus WM-guided Qwen 10/29 and AUROC 0.6486. WM-guided Qwen and static/dynamic WM selected the same 20 patients, highlighting the need to test longer sequences and actual adaptive behavior. The [Phase47 protocol](docs/PHASE47_QUERY_BUDGET_CURVES.md) records one Qwen rollout to q=16 and aggregate risk/cost snapshots at q=4/8/12/16, with no further model downloads or use of consumed TEST.

```bash
sbatch slurm/phase47_dusky_qwen_budget.sbatch
# or: sbatch slurm/phase47_nibi_qwen_budget.sbatch
```


## Phase48 — Masked Transformer probabilistic observation WM

The Phase47 results do **not** establish that Qwen-only is generally superior to WM-guided Qwen, because only a small Naive Bayes WM was used. Phase48 implements a patient-split neural masked-item Transformer ensemble, with explicit missing token vs observed score 0, class-conditional item probabilities, and estimated next-question information gain. It evaluates heldout TRAIN masked-item NLL, Brier and AUROC **before** connecting this WM to Qwen. This remains same-visit V04 observation modelling, not a disease-progression World Model.

- [Methods, limits and Slurm instructions](docs/PHASE48_NEURAL_WM.md)
- `scripts/phase48_masked_transformer_wm.py`
- `tests/test_phase48_masked_transformer_wm.py`
- Dusky CPU: `sbatch slurm/phase48_dusky_cpu_wm.sbatch`

All results remain exploratory TRAIN analyses; the final Phase44 TEST remains consumed.
