#!/usr/bin/env python3
"""Phase 7: privacy-safe date-format and motor-exam QC for Parkinson CSVs.

Read-only. Exports no patient IDs, actual dates or clinical scores. Runs with
Python standard library only. Reports data format, chronology (visit-start,
NOT diagnostic-decision time), and repeated motor exam stratification.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

FILES = {
    "PPMI": {
        "baseline_start": "Baseline_Visit_Start_",
        "diagnosis": "Primary_Research_Diagnosis_",
        "motor": "MDS-UPDRS_Part_III_",
        "moca": "Montreal_Cognitive_Assessment__MoCA__",
        "upsit": "University_of_Pennsylvania_Smell_Identification_Test_UPSIT_",
        "rbd": "REM_Sleep_Behavior_Disorder_Screening_Questionnaire_",
        "scopa": "SCOPA-AUT_",
    },
    "BioFIND": {
        "screening": "Screening_Demographics_",
        "diagnosis": "Primary_Diagnosis_",
        "motor_post_dose": "MDS_UPDRS_Part_III__Post_Dose__",
        "moca": "Montreal_Cognitive_Assessment__MoCA__",
    },
}
DATE_COLS = ("INFODT", "EXAMDT", "PDDXDT", "SXDT", "ORIG_ENTRY", "LAST_UPDATE")
MOTOR_COLS = ("NP3TOT", "NP3GAIT", "PDSTATE", "PDTRTMNT", "ONEXAM", "OFFEXAM", "ONOFFORDER", "EXAMDT")
VALID_VISITS = {"SC", "BL"}
EMPTY = {"", "n/a", "na", "nan", "none", "null", ".", "missing", "nat"}


def nonempty(value):
    return value is not None and str(value).strip().lower() not in EMPTY


def shape(value):
    """Character-class signature; never export the actual date string."""
    s = str(value).strip()
    if not s:
        return "EMPTY"
    if len(s) > 128:
        return "TOO_LONG"
    return "".join("9" if ch.isdigit() else "A" if ch.isalpha() else
                   "_" if ch.isspace() else ch for ch in s)


def parse_date(value):
    """A strictly format-driven parser, never fuzzy-guess day/month order."""
    if not nonempty(value):
        return None
    s = str(value).strip()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    for fmt in (
        "%m/%d/%Y", "%d/%m/%Y", "%d-%b-%Y", "%d-%b-%y",
        "%d%b%Y", "%Y/%m/%d", "%Y%m%d", "%m/%d/%y",
        "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S", "%d-%b-%Y %H:%M:%S",
    ):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def find_file(folder, prefix):
    pattern = re.compile(re.escape(prefix) + r"\d{2}[A-Za-z]{3}\d{4}\.csv$")
    paths = [p for p in folder.glob(prefix + "*.csv") if pattern.fullmatch(p.name)]
    if len(paths) != 1:
        raise ValueError(f"Expected one version of {prefix}; got {len(paths)}")
    return paths[0]


def iter_safe_rows(path):
    """Handle duplicate PATNO/EVENT_ID columns conservatively."""
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        header = [x.strip() for x in next(reader, [])]
        positions = defaultdict(list)
        for i, key in enumerate(header):
            positions[key].append(i)
        def get(row, column):
            vals = [row[i].strip() for i in positions.get(column, []) if i < len(row)]
            clean = [value for value in vals if nonempty(value)]
            if len(set(clean)) > 1:
                raise ValueError("Inconsistent duplicated column")
            return clean[0] if clean else ""
        for row in reader:
            yield positions, get, row


def export_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def date_and_anchor_audit(root, output, minimum):
    formats, summary_rows, timing_rows, issues = [], [], [], []
    anchor_dates = defaultdict(set)
    # Scan BL date anchor only for PPMI. Date parsing is deliberately provisional.
    try:
        path = find_file(root / "PPMI", FILES["PPMI"]["baseline_start"])
        for positions, get, row in iter_safe_rows(path):
            if get(row, "EVENT_ID") != "BL":
                continue
            pid = get(row, "PATNO")
            if nonempty(pid):
                anchor_dates[pid].add(parse_date(get(row, "INFODT")))
    except (OSError, ValueError, csv.Error) as exc:
        issues.append({"cohort": "PPMI", "table": "baseline_start", "error": type(exc).__name__})
    resolved_anchor = {}
    ambiguous = 0
    for pid, vals in anchor_dates.items():
        defined = {v for v in vals if v is not None}
        if len(defined) == 1:
            resolved_anchor[pid] = next(iter(defined))
        elif len(defined) > 1:
            ambiguous += 1
    for cohort, tables in FILES.items():
        for name, prefix in tables.items():
            try:
                path = find_file(root / cohort, prefix)
                fmt = Counter()
                date_presence = Counter()
                pair_timing = Counter()
                for positions, get, row in iter_safe_rows(path):
                    pid = get(row, "PATNO")
                    event = get(row, "EVENT_ID") if "EVENT_ID" in positions else ""
                    if not nonempty(pid):
                        continue
                    for column in DATE_COLS:
                        if column not in positions:
                            continue
                        value = get(row, column)
                        signature = shape(value)
                        parsed = parse_date(value)
                        status = "empty" if not nonempty(value) else "parseable" if parsed else "unparsed"
                        fmt[(column, status, signature)] += 1
                        date_presence[(column, status)] += 1
                        if (cohort == "PPMI" and column in ("INFODT", "EXAMDT")
                            and event in VALID_VISITS and name != "baseline_start"):
                            anchor = resolved_anchor.get(pid)
                            if anchor is None:
                                category = "no_parseable_BL_anchor"
                            elif parsed is None:
                                category = "missing_or_unparsed_test_date"
                            elif parsed < anchor:
                                category = "before_BL_visit_start"
                            elif parsed > anchor:
                                category = "after_BL_visit_start"
                            else:
                                category = "same_day_BL_visit_start"
                            pair_timing[(column, event, category)] += 1
                for (column, status, signature), n in sorted(fmt.items()):
                    if n >= minimum:
                        formats.append(dict(cohort=cohort, table=name, field=column,
                                            parse_status=status, character_signature=signature,
                                            rows=n))
                for (column, status), n in sorted(date_presence.items()):
                    summary_rows.append(dict(cohort=cohort, table=name, field=column,
                                             parse_status=status, rows=n))
                for (column, event, category), n in sorted(pair_timing.items()):
                    timing_rows.append(dict(cohort=cohort, table=name, field=column,
                                            visit=event, comparison=category, rows=n))
            except (OSError, ValueError, csv.Error, UnicodeError) as exc:
                issues.append({"cohort": cohort, "table": name, "error": type(exc).__name__})
    summary = dict(
        unique_BL_anchor_patients=len(resolved_anchor),
        conflicting_dated_BL_anchor_patients=ambiguous,
        rows_in_date_summary=len(summary_rows),
        warning="BL visit start is not a validated diagnosis decision timestamp",
    )
    export_csv(output / "date_patterns.csv",
               ["cohort", "table", "field", "parse_status", "character_signature", "rows"],
               formats)
    export_csv(output / "date_parse_summary.csv",
               ["cohort", "table", "field", "parse_status", "rows"], summary_rows)
    export_csv(output / "visit_start_timing.csv",
               ["cohort", "table", "field", "visit", "comparison", "rows"], timing_rows)
    return summary, issues


def motor_repeat_audit(root, output, minimum):
    result = []
    for cohort, table_name in (("PPMI", "motor"), ("BioFIND", "motor_post_dose")):
        path = find_file(root / cohort, FILES[cohort][table_name])
        groups = defaultdict(list)
        key_conflicts = 0
        for positions, get, row in iter_safe_rows(path):
            try:
                pid, visit = get(row, "PATNO"), get(row, "EVENT_ID")
                if not nonempty(pid) or visit not in VALID_VISITS:
                    continue
                groups[(pid, visit)].append({c: get(row, c) for c in MOTOR_COLS if c in positions})
            except ValueError:
                key_conflicts += 1
        counts = Counter()
        for (pid, visit), rows in groups.items():
            if len(rows) <= 1:
                continue
            counts[(visit, "patient_visits_with_repeated_motor_records")] += 1
            counts[(visit, "extra_motor_rows")] += len(rows) - 1
            for name in MOTOR_COLS:
                values = {r.get(name, "") for r in rows if nonempty(r.get(name, ""))}
                if len(values) > 1:
                    counts[(visit, "conflicting_" + name)] += 1
                elif len(values) == 1:
                    counts[(visit, "consistent_" + name)] += 1
            if any(len({r.get(c, "") for r in rows if nonempty(r.get(c, ""))}) > 1
                   for c in ("PDSTATE", "PDTRTMNT", "OFFEXAM", "ONEXAM", "ONOFFORDER")):
                counts[(visit, "different_treatment_or_state_metadata")] += 1
            if any(len({r.get(c, "") for r in rows if nonempty(r.get(c, ""))}) > 1
                   for c in ("EXAMDT", "INFODT")):
                counts[(visit, "different_exam_dates")] += 1
        result.extend(dict(cohort=cohort, visit=v, finding=f,
                           patient_visits=str(n) if n >= minimum or n == 0 else f"<{minimum}")
                      for (v, f), n in sorted(counts.items()))
        result.append(dict(cohort=cohort, visit="ALL", finding="conflicting_key_rows",
                           patient_visits=str(key_conflicts) if key_conflicts >= minimum or key_conflicts == 0
                           else f"<{minimum}"))
    export_csv(output / "motor_repeat_factors.csv",
               ["cohort", "visit", "finding", "patient_visits"], result)
    return len(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/date_motor_probe"))
    parser.add_argument("--min-cell-count", type=int, default=10)
    args = parser.parse_args()
    root, output = args.data_root.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if not root.is_dir() or output == root or root in output.parents or args.min_cell_count < 10:
        parser.error("A valid data root, separate output directory and minimum cell >=10 are required")
    output.mkdir(parents=True, exist_ok=True)
    summary, issues = date_and_anchor_audit(root, output, args.min_cell_count)
    try:
        summary["motor_repeat_factor_rows"] = motor_repeat_audit(root, output, args.min_cell_count)
    except (OSError, ValueError, csv.Error) as exc:
        issues.append({"cohort": "ALL", "table": "motor_repeat", "error": type(exc).__name__})
    export_csv(output / "issues.csv", ["cohort", "table", "error"], issues)
    summary.update(issues=len(issues), generated_utc=datetime.now(timezone.utc).isoformat(),
                   note="Aggregate schema and counts; never exports IDs, dates, or score values")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Share only reports from:", output)


if __name__ == "__main__":
    main()
