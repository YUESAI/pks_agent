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
