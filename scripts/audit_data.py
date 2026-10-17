#!/usr/bin/env python3
"""Read-only, sample-based inventory for PPMI, BioFIND and Fox Insight datasets.

Only schema/statistics are exported. No patient-level cell values are written.
Standard library is sufficient for CSV/TSV and .gz files; optional readers add
Excel, Parquet and SAS support.

Example:
  python scripts/audit_data.py --data-root /scratch/ylong66/data/pks/data \
      --output-dir ./outputs/data_audit
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

DATASETS = ("PPMI", "BioFIND", "FoxInsight")
MISSING = {"", "na", "n/a", "nan", "null", "none", "missing", "unknown", "unk", "."}
ID_NAMES = {
    "patno", "patient_id", "patientid", "participant_id", "participantid",
    "subject_id", "subjectid", "subjid", "study_id", "research_id", "record_id",
    "respondent_id", "response_id", "foxid", "fox_id",
}
VISIT_NAMES = {
    "event_id", "eventid", "visit", "visit_id", "visitid", "visit_code",
    "visit_name", "visit_date", "visitdate", "infodt", "assessment_date",
    "study_visit", "timepoint", "time_point", "wave", "survey_date",
}
LABEL_NAMES = {
    "apprdx", "diagnosis", "diagnosis_group", "diagnosis_category",
    "enroll_cat", "cohort", "pd_status", "pd_diagnosis",
    "primary_diagnosis", "diagnostic_group", "dx",
}
CATEGORY_HINTS = {
    "demographics": ("demo", "age", "gender", "sex", "race", "ethnic"),
    "diagnosis_or_cohort": ("diagnos", "cohort", "enroll", "apprdx", "dx"),
    "motor": ("updrs", "motor", "tremor", "rigid", "gait", "bradykin"),
    "cognition": ("moca", "mmse", "cognit"),
    "nonmotor": ("upsit", "smell", "olfact", "rbd", "sleep", "scopa", "autonom"),
    "medication": ("medicat", "levodopa", "dopamin", "drug"),
    "imaging": ("datscan", "spect", "mri", "pet", "imaging", "scan"),
    "biospecimen": ("biospec", "assay", "csf", "plasma", "serum", "protein"),
    "genetics": ("genetic", "genotyp", "variant", "snp", "genomic"),
}
TABULAR_SUFFIXES = (".csv", ".tsv", ".csv.gz", ".tsv.gz", ".xlsx", ".parquet", ".sas7bdat", ".xpt")


def norm(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def table_kind(path: Path) -> str:
    name = path.name.lower()
    for ext in TABULAR_SUFFIXES:
        if name.endswith(ext):
            return ext.lstrip(".")
    return "other"


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING
    try:
        return value != value
    except Exception:
        return False


def classify_file(relative_path: str, columns: list[str]) -> str:
    text = norm(relative_path + " " + " ".join(columns))
    scores = {}
    for category, keywords in CATEGORY_HINTS.items():
        scores[category] = sum(1 for word in keywords if word in text)
    best = max(scores, key=scores.get)
    return best if scores[best] else "unclassified"


def role(name: str) -> str:
    n = norm(name)
    if n in ID_NAMES:
        return "patient_key_candidate"
    if n in VISIT_NAMES:
        return "visit_or_date_candidate"
    if n in LABEL_NAMES:
        return "label_or_leakage_candidate"
    return ""


def format_row(row: Iterable[Any], width: int) -> list[Any]:
    out = list(row)
    if len(out) < width:
        out += [None] * (width - len(out))
    return out[:width]


def csv_rows(path: Path, kind: str) -> tuple[list[str], Iterator[list[Any]], Any]:
    opener = gzip.open if kind.endswith(".gz") else open
    handle = opener(path, "rt", encoding="utf-8-sig", errors="replace", newline="")
    try:
        snippet = handle.read(16_384)
        handle.seek(0)
        delimiter = "\t" if kind.startswith("tsv") else ","
        try:
            dialect = csv.Sniffer().sniff(snippet, delimiters=",\t;|")
            reader = csv.reader(handle, dialect)
        except csv.Error:
            reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, [])
        return [str(x).strip() for x in header], reader, handle
    except Exception:
        handle.close()
        raise


def read_schema_sample(path: Path, kind: str, limit: int) -> tuple[list[str], list[list[Any]], str, int | None]:
    """Read at most limit data rows per table. Return columns, sample, note, known total rows."""
    if kind in {"csv", "tsv", "csv.gz", "tsv.gz"}:
        header, reader, handle = csv_rows(path, kind)
        try:
            return header, [format_row(row, len(header)) for _, row in zip(range(limit), reader)], "", None
        finally:
            handle.close()
    if kind == "xlsx":
        try:
            import openpyxl
        except ImportError:
            raise RuntimeError("Install optional dependency: pip install openpyxl")
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            ws = wb.worksheets[0]
            rows = ws.iter_rows(values_only=True)
            header = [str(c) if c is not None else "" for c in next(rows, ())]
            sample = [format_row(row, len(header)) for _, row in zip(range(limit), rows)]
            return header, sample, "first_sheet=" + str(ws.title), None
        finally:
            wb.close()
    if kind == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise RuntimeError("Install optional dependency: pip install pyarrow")
        pf = pq.ParquetFile(path)
        header = list(pf.schema_arrow.names)
        batch_iter = pf.iter_batches(batch_size=max(1, limit)) if limit else iter(())
        first = next(batch_iter, None)
        sample = []
        if first is not None:
            cols = [column.to_pylist() for column in first.columns]
            sample = [list(row) for row in zip(*cols)][:limit]
        return header, sample, "", pf.metadata.num_rows
    if kind in {"sas7bdat", "xpt"}:
        try:
            import pyreadstat
        except ImportError:
            raise RuntimeError("Install optional dependency: pip install pyreadstat")
        reader = pyreadstat.read_sas7bdat if kind == "sas7bdat" else pyreadstat.read_xport
        frame, meta = reader(str(path), row_limit=limit, metadataonly=(limit == 0))
        return list(meta.column_names), frame.values.tolist(), "", meta.number_rows
    raise RuntimeError("Unsupported table format: " + kind)


def export_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def profile_column(values: list[Any]) -> tuple[int, str]:
    present = [v for v in values if not is_missing(v)]
    if not present:
        return len(values), "empty"
    numerical = 0
    for value in present:
        try:
            float(value)
            numerical += 1
        except (ValueError, TypeError, OverflowError):
            pass
    if numerical == len(present):
        inferred = "numeric"
    elif numerical == 0:
        inferred = "text_or_category"
    else:
        inferred = "mixed"
    return len(values) - len(present), inferred


def scan(root: Path, out: Path, max_rows: int, cohorts: list[str]) -> dict[str, Any]:
    files_out: list[dict[str, Any]] = []
    cols_out: list[dict[str, Any]] = []
    join_out: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    extensions: Counter[str] = Counter()

    for cohort in cohorts:
        folder = root / cohort
        if not folder.is_dir():
            issues.append({"cohort": cohort, "file": "", "issue": "Missing dataset directory"})
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(folder).as_posix()
            kind = table_kind(path)
            bytes_ = path.stat().st_size
            counts[cohort] += 1
            extensions[kind] += 1
            entry: dict[str, Any] = {
                "cohort": cohort, "relative_path": rel, "format": kind,
                "size_bytes": bytes_, "size_mb": round(bytes_ / 1048576, 3),
                "status": "metadata_only", "sample_rows": 0, "known_total_rows": "",
                "n_columns": "", "category_hint": "", "notes": "",
            }
            if kind != "other":
                try:
                    header, rows, note, known_rows = read_schema_sample(path, kind, max_rows)
                    if not header or not any(header):
                        raise ValueError("Empty or invalid table header")
                    entry.update(
                        status="sampled", sample_rows=len(rows), n_columns=len(header),
                        category_hint=classify_file(rel, header), notes=note,
                        known_total_rows=known_rows if known_rows is not None else "",
                    )
                    patient = [c for c in header if role(c) == "patient_key_candidate"]
                    visit = [c for c in header if role(c) == "visit_or_date_candidate"]
                    label = [c for c in header if role(c) == "label_or_leakage_candidate"]
                    join_out.append({
                        "cohort": cohort, "relative_path": rel,
                        "patient_key_candidates": " | ".join(patient),
                        "visit_key_candidates": " | ".join(visit),
                        "label_or_leakage_candidates": " | ".join(label),
                        "sample_rows": len(rows),
                        "category_hint": entry["category_hint"],
                    })
                    for idx, name in enumerate(header):
                        values = [row[idx] for row in rows]
                        n_missing, dtype = profile_column(values)
                        cols_out.append({
                            "cohort": cohort, "relative_path": rel, "column_index": idx,
                            "column_name": name, "normalized_name": norm(name),
                            "role_hint": role(name), "inferred_type_sample": dtype,
                            "sample_rows": len(rows), "missing_rows_sample": n_missing,
                            "missing_pct_sample": round(100 * n_missing / len(rows), 2) if rows else "",
                        })
                    if len(header) != len(set(header)):
                        issues.append({"cohort": cohort, "file": rel, "issue": "Duplicate column names"})
                except Exception as e:
                    entry["status"] = "error"
                    # Error type only: avoid leaking raw cell data from parser exceptions.
                    entry["notes"] = type(e).__name__
                    issues.append({"cohort": cohort, "file": rel, "issue": type(e).__name__})
            files_out.append(entry)

    export_csv(out / "files.csv", files_out,
               ["cohort", "relative_path", "format", "size_bytes", "size_mb", "status",
                "sample_rows", "known_total_rows", "n_columns", "category_hint", "notes"])
    export_csv(out / "columns.csv", cols_out,
               ["cohort", "relative_path", "column_index", "column_name", "normalized_name",
                "role_hint", "inferred_type_sample", "sample_rows", "missing_rows_sample",
                "missing_pct_sample"])
    export_csv(out / "join_candidates.csv", join_out,
               ["cohort", "relative_path", "patient_key_candidates", "visit_key_candidates",
                "label_or_leakage_candidates", "sample_rows", "category_hint"])
    export_csv(out / "issues.csv", issues, ["cohort", "file", "issue"])

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root), "cohorts": cohorts,
        "max_rows_per_file": max_rows, "files_by_cohort": dict(counts),
        "file_formats": dict(extensions), "sampled_tables": len(join_out),
        "column_entries": len(cols_out), "issues": len(issues),
        "note": "Sample-based structural audit. No patient-level values are exported; row totals are unknown for CSV.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    top = sorted(join_out,
                 key=lambda item: (
                     bool(item["patient_key_candidates"]),
                     bool(item["visit_key_candidates"]),
                     item["sample_rows"]
                 ), reverse=True)[:40]
    lines = [
        "# Parkinson datasets: structural audit",
        "",
        "> This is a **sample-based inventory**, not a validated medical data dictionary.",
        "> No patient-level values have been exported.",
        "",
        f"- Dataset root: \`{root}\`",
        f"- Rows sampled per tabular file: **at most {max_rows}**",
        f"- Sampled tables: **{len(join_out)}**",
        f"- Column entries: **{len(cols_out)}**",
        f"- Issues: **{len(issues)}**",
        "",
        "## Files by cohort",
        "",
    ]
    for cohort in cohorts:
        lines.append(f"- {cohort}: {counts[cohort]} files")
    lines += ["", "## Formats", ""]
    for fmt, num in sorted(extensions.items()):
        lines.append(f"- {fmt}: {num}")
    lines += ["", "## Tables to inspect first (heuristic, not validated)", "",
              "| Cohort | Relative path | Candidate patient key | Candidate visit key | Suspected label / leakage columns |",
              "|---|---|---|---|---|"]
    for item in top:
        def safe(v: Any) -> str:
            return str(v).replace("|", "/").replace("\n", " ")[:120]
        lines.append("| " + " | ".join(safe(item[key]) for key in (
            "cohort", "relative_path", "patient_key_candidates",
            "visit_key_candidates", "label_or_leakage_candidates")) + " |")
    lines += [
        "", "## Interpretation and next steps", "",
        "1. Review \`files.csv\` for data availability and unsupported formats.",
        "2. Review \`join_candidates.csv\`: candidate keys **must be verified**, including uniqueness, grain, and dates.",
        "3. Review \`columns.csv\` for coverage and candidate clinical measures.",
        "4. Confirm cohort labels and possible target leakage **before** merging or modeling.",
        "5. Audit reports contain schema and local relative filenames; review before sharing publicly.",
        "6. Special missing-value codes (e.g. -9/-1) are NOT automatically treated as missing.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Absolute path to folder containing PPMI, BioFIND, FoxInsight")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data_audit"),
                        help="Output path, independent of data-root; default: ./outputs/data_audit")
    parser.add_argument("--max-rows", type=int, default=500,
                        help="Max sampled data rows per file (default: 500, 0 = headers only)")
    parser.add_argument("--cohorts", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args(argv)
    if args.max_rows < 0:
        parser.error("--max-rows must be non-negative")
    root = args.data_root.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error("Data root directory does not exist: " + str(root))
    if out == root or root in out.parents:
        parser.error("Output directory must be outside the raw data directory")
    if out == Path.cwd().resolve():
        parser.error("Choose a dedicated output directory, not the repository root")
    out.mkdir(parents=True, exist_ok=True)
    result = scan(root, out, args.max_rows, args.cohorts)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nReports written to:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
