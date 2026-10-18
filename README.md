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
