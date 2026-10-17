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
