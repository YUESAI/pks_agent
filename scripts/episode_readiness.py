#!/usr/bin/env python3
"""Phase 6: read-only patient-episode readiness audit (aggregate outputs only).

Compares BL-only vs strictly dated SC/BL measurements without exporting IDs.
This does NOT certify clinical labels or diagnosis-time causal validity.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

COMBINATIONS = {
    "PPMI": [
        ("updrs3_total", "moca_total", "rbd_vivid_dreams"),
        ("updrs3_total", "moca_total", "rbd_vivid_dreams", "scopa_aut_item1"),
        ("updrs3_total", "moca_total", "rbd_vivid_dreams", "upsit_total"),
    ],
    "BioFIND": [
        ("updrs3_gait_post_dose", "moca_total", "rbd_vivid_dreams"),
    ],
}
MISSING = {"", "na", "n/a", "nan", "null", "none", "missing", "."}
SENTINELS = {"-1", "-7", "-8", "-9", "-99", "999", "9999"}
IMPUTED_FALSE = {"", "false", "0", "no", "n", "f"}


def nonempty(value):
    return value is not None and str(value).strip().lower() not in MISSING


def date_parse(text):
    if not nonempty(text):
        return None
    value = str(text).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%d%b%Y", "%Y/%m/%d",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    return None


def bucket_raw_date(text):
    if not nonempty(text):
        return "empty"
    if date_parse(text) is not None:
        return "parseable"
    s = str(text).strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return "numeric_unparsed"
    if re.fullmatch(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}", s):
        return "separated_unparsed"
    return "other_unparsed"


def load_bl_anchor(folder):
    """Use PPMI Baseline_Visit_Start INFODT, not diagnosis-form INFODT."""
    prefix = "Baseline_Visit_Start_"
    candidates = sorted(folder.glob(prefix + "*.csv"))
    pattern = re.compile(re.escape(prefix) + r"\d{2}[A-Za-z]{3}\d{4}\.csv$")
    files = [p for p in candidates if pattern.fullmatch(p.name)]
    if len(files) != 1:
        raise ValueError("Expected exactly one PPMI baseline visit start table")
    raw = defaultdict(set)
    stats = Counter()
    with files[0].open(encoding="utf-8-sig", newline="", errors="replace") as h:
        reader = csv.DictReader(h)
        hdr = reader.fieldnames or []
        if not {"PATNO", "EVENT_ID", "INFODT"}.issubset(hdr):
            raise ValueError("Baseline visit table is missing PATNO/EVENT_ID/INFODT")
        for row in reader:
            if (row.get("EVENT_ID") or "").strip() != "BL":
                continue
            pid = (row.get("PATNO") or "").strip()
            if not nonempty(pid):
                stats["missing_patient_id_rows"] += 1
                continue
            raw_date = row.get("INFODT")
            stats["rows_BL"] += 1
            stats["date_format_" + bucket_raw_date(raw_date)] += 1
            parsed = date_parse(raw_date)
            raw[pid].add(parsed)
    resolved = {}
    states = Counter()
    for pid, values in raw.items():
        good = {v for v in values if v is not None}
        if len(good) > 1:
            states["conflicting_dated_anchors"] += 1
        elif len(good) == 1:
            resolved[pid] = next(iter(good))
            if None in values:
                states["dated_anchor_with_undated_duplicate"] += 1
            else:
                states["unique_dated_anchor"] += 1
        else:
            states["only_missing_or_unparsed_anchor"] += 1
    return resolved, stats, states


def open_db(path, engine):
    if engine == "duckdb":
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("Install duckdb: pip install duckdb") from exc
        return duckdb.connect(str(path), read_only=True)
    return sqlite3.connect("file:" + str(path.resolve()) + "?mode=ro", uri=True)


def load_candidates(db):
    labels = defaultdict(dict)
    for cohort, patient, label in db.execute(
        "SELECT cohort,patient_id,candidate_label FROM restricted_labels "
        "WHERE candidate_label IN ('PD','HC')"
    ).fetchall():
        labels[cohort][patient] = label
    return labels


def load_observations(db, allowed):
    observed = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    query = ("SELECT cohort,patient_id,event_id,concept,value_raw,exam_date,quality_status,"
             "medication_state,imputed_flag FROM observations "
             "WHERE event_id IN ('SC','BL')")
    cursor = db.execute(query)
    while True:
        batch = cursor.fetchmany(5000)
        if not batch:
            break
        for cohort, pid, visit, concept, value, exam_date, quality, med, imputed in batch:
            if pid not in allowed.get(cohort, {}):
                continue
            observed[cohort][pid][concept][visit].append({
                "value": (value or "").strip(), "date": date_parse(exam_date),
                "quality": (quality or "").strip(), "med": (med or "").strip(),
                "imputed": (imputed or "").strip().lower()
            })
    return observed


def resolve(rows):
    """Return (readiness, date). Duplicate exams never silently choose a value."""
    if not rows:
        return "absent", None
    safe = []
    for row in rows:
        if not nonempty(row["value"]):
            continue
        if row["value"] in SENTINELS or row["quality"] != "observed_unverified":
            continue
        if row["imputed"] not in IMPUTED_FALSE:
            continue
        safe.append(row)
    if not safe:
        return "quality_blocked", None
    values = {r["value"] for r in safe}
    medications = {r["med"] for r in safe if r["med"]}
    dates = {r["date"] for r in safe}
    if len(values) != 1:
        return "conflicting_values", None
    if len(medications) > 1:
        return "conflicting_medication", None
    if len(dates) > 1:
        return "conflicting_dates", None
    return ("single" if len(safe) == 1 else "identical_repeats"), next(iter(dates))


def qualify(patient_features, concept, idx_date, window_days):
    """Conservative availability; never certifies prospective decision timing."""
    records = patient_features.get(concept, {})
    bl = records.get("BL", [])
    sc = records.get("SC", [])
    b_state, b_date = resolve(bl)
    s_state, s_date = resolve(sc)
    accepted = {"single", "identical_repeats"}
    b_safe = b_state in accepted
    s_safe = s_state in accepted

    def dated(day):
        return (idx_date is not None and day is not None and
                timedelta(days=0) <= idx_date - day <= timedelta(days=window_days))

    return {
        "bl_any": bool(bl),
        "sc_any": bool(sc),
        "bl_resolved": b_safe,
        "sc_resolved": s_safe,
        "bl_dated": b_safe and dated(b_date),
        "sc_dated": s_safe and dated(s_date),
        "union_any": bool(bl or sc),
        "union_resolved": b_safe or s_safe,
        "union_dated": (b_safe and dated(b_date)) or (s_safe and dated(s_date)),
        "bl_state": b_state,
        "sc_state": s_state,
        "bl_after_anchor": b_safe and idx_date is not None and b_date is not None and b_date > idx_date,
        "sc_after_anchor": s_safe and idx_date is not None and s_date is not None and s_date > idx_date,
    }


def redact_count(value, minimum):
    if value == 0 or value >= minimum:
        return str(value)
    return "<" + str(minimum)


def to_csv(path, names, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit(dbpath, engine, root, out, window_days, min_count):
    out.mkdir(parents=True, exist_ok=True)
    db = open_db(dbpath, engine)
    try:
        labels = load_candidates(db)
        obs = load_observations(db, labels)
    finally:
        db.close()

    anchors, formats, anchor_states = load_bl_anchor(root / "PPMI")
    all_features = defaultdict(set)
    for cohort, features in COMBINATIONS.items():
        for combo in features:
            all_features[cohort].update(combo)
    date_rows = []
    feature_rows = []
    state_rows = []
    joint_rows = []
    names = ("bl_any", "bl_resolved", "bl_dated", "sc_any", "sc_resolved",
             "sc_dated", "union_any", "union_resolved", "union_dated")
    for cohort, patient_labels in labels.items():
        for group in ("PD", "HC"):
            subset = [pid for pid, value in patient_labels.items() if value == group]
            indexed = sum(pid in anchors for pid in subset) if cohort == "PPMI" else 0
            date_rows.append({"cohort": cohort, "label": group, "candidate_patients": len(subset),
                              "BL_visit_anchor_date_patients": indexed,
                              "no_or_ambiguous_date_patients": len(subset)-indexed})
            facts = {}
            for pid in subset:
                index = anchors.get(pid) if cohort == "PPMI" else None
                tests = obs.get(cohort, {}).get(pid, {})
                facts[pid] = {feat: qualify(tests, feat, index, window_days)
                              for feat in all_features.get(cohort, ())}
            for feat in sorted(all_features.get(cohort, ())):
                counters = Counter()
                for pid in subset:
                    outcome = facts[pid][feat]
                    for key in names + ("bl_after_anchor", "sc_after_anchor"):
                        counters[key] += bool(outcome[key])
                    counters["bl_state:" + outcome["bl_state"]] += 1
                    counters["sc_state:" + outcome["sc_state"]] += 1
                feature_rows.append({"cohort": cohort, "label": group, "feature": feat,
                                     "n_patients": len(subset),
                                     **{key: redact_count(counters[key], min_count) for key in names},
                                     "bl_after_anchor": redact_count(counters["bl_after_anchor"], min_count),
                                     "sc_after_anchor": redact_count(counters["sc_after_anchor"], min_count)})
                for state_key, n in sorted(counters.items()):
                    if state_key.startswith(("bl_state:", "sc_state:")):
                        state_rows.append({"cohort": cohort, "label": group, "feature": feat,
                                           "state": state_key, "patients": redact_count(n, min_count)})
            for combo in COMBINATIONS[cohort]:
                for flag in ("bl_any", "bl_resolved", "bl_dated", "union_any",
                             "union_resolved", "union_dated"):
                    n = sum(all(facts[pid][f][flag] for f in combo) for pid in subset)
                    joint_rows.append({"cohort": cohort, "label": group,
                                       "features": "+".join(combo), "view": flag,
                                       "patients": redact_count(n, min_count),
                                       "denominator": len(subset)})
    date_summary = Counter()
    for key, n in sorted(formats.items()):
        date_summary[key] = n
    for key, n in sorted(anchor_states.items()):
        date_summary[key] = n
    to_csv(out / "baseline_date_coverage.csv",
           ["cohort", "label", "candidate_patients", "BL_visit_anchor_date_patients",
            "no_or_ambiguous_date_patients"], date_rows)
    to_csv(out / "date_source_status.csv", ["category", "count"],
           [{"category": k, "count": redact_count(n, min_count)} for k, n in date_summary.items()])
    to_csv(out / "feature_readiness.csv",
           ["cohort", "label", "feature", "n_patients"] + list(names) +
           ["bl_after_anchor", "sc_after_anchor"], feature_rows)
    to_csv(out / "duplicate_resolution.csv", ["cohort", "label", "feature", "state", "patients"],
           state_rows)
    to_csv(out / "joint_readiness.csv",
           ["cohort", "label", "features", "view", "patients", "denominator"], joint_rows)
    summary = {
        "status": "audit_only_not_training_ready",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "date_anchor_source": "PPMI/Baseline_Visit_Start_*:INFODT at EVENT_ID=BL",
        "screening_window_days": window_days,
        "n_candidate_PPMI": len(labels.get("PPMI", {})),
        "n_candidate_BioFIND": len(labels.get("BioFIND", {})),
        "report_files": ["baseline_date_coverage.csv", "date_source_status.csv",
                         "feature_readiness.csv", "duplicate_resolution.csv", "joint_readiness.csv"],
        "warnings": [
            "Candidate labels are retrospective cohort plus BL diagnosis, not clinically validated.",
            "Baseline Visit Start date is NOT verified diagnosis decision time.",
            "SC/BL exam dates are checked only against baseline visit start, not decision timestamp.",
            "Same-day tests do not establish prospective clinical ordering.",
            "Not a model training manifest.",
            "BioFIND needs an independently validated BL date anchor for dated availability.",
            "Repeated exams only resolve when usable value, date, and medication state agree.",
        ]
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n",
                                      encoding="utf-8")
    lines = ["# Phase 6 — Episode readiness audit", "",
             "Aggregate-only; candidate labels are not clinically validated.", "",
             "## Baseline visit-date anchors", "",
             "| Cohort | Label | Candidate patients | BL anchor date patients |",
             "|---|---|---:|---:|"]
    for r in date_rows:
        lines.append("| {cohort} | {label} | {candidate_patients} | {BL_visit_anchor_date_patients} |".format(**r))
    lines += ["", "## Interpretation", "",
              "bl_any counts any BL measurement, even low-quality or duplicated.",
              "bl_resolved excludes flagged imputed/sentinel values and unresolved duplicate exams.",
              "bl_dated also requires a source date at/before BL Visit Start within the window.",
              "union_dated allows SC or BL only if dated within the window before BL Visit Start.",
              "These are retrospective checks, NOT certification of clinical decision timing.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-path", type=Path, required=True)
    ap.add_argument("--engine", choices=("duckdb", "sqlite"), default="duckdb")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/episode_readiness"))
    ap.add_argument("--screening-window-days", type=int, default=180)
    ap.add_argument("--min-cell-count", type=int, default=10)
    args = ap.parse_args()
    root = args.data_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    db = args.db_path.expanduser().resolve()
    if not root.is_dir() or not db.is_file():
        ap.error("Raw data root or DB does not exist")
    if output == root or root in output.parents:
        ap.error("Output must be outside raw patient data root")
    if args.screening_window_days < 1 or args.min_cell_count < 10:
        ap.error("Window must be positive and min cell count >= 10")
    result = audit(db, args.engine, root, output, args.screening_window_days, args.min_cell_count)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Upload ONLY the aggregate output directory, NEVER the database")

if __name__ == "__main__":
    main()
