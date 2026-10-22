#!/usr/bin/env python3
"""Phase 9: create an on-demand, retrospective baseline episode manifest.

Input is a local v2 clinical DB, plus original BL visit-start file.
Output manifest is patient-level and MUST stay on the authorized cluster.
Report directory is aggregate-only and may be reviewed before sharing.
This is NOT a clinically validated diagnosis-time reconstruction.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ACTIONS = {
    "PPMI": {
        "updrs3_total": ("ON", "OFF", "UNSPECIFIED"),
        "updrs3_gait": ("ON", "OFF", "UNSPECIFIED"),
        "updrs3_bradykinesia": ("ON", "OFF", "UNSPECIFIED"),
        "moca_total": (None,),
        "rbd_vivid_dreams": (None,),
        "scopa_aut_item1": (None,),
        "upsit_total": (None,),
        "age_at_visit": (None,),
    },
    "BioFIND": {
        "updrs3_gait_post_dose": (None,),
        "updrs3_bradykinesia_post_dose": (None,),
        "moca_total": (None,),
        "rbd_vivid_dreams": (None,),
    },
}
EMPTY = {"", "nan", "na", "n/a", "null", "none", "missing", "."}
BAD_SENTINELS = {"-1", "-7", "-8", "-9", "-99", "999", "9999"}
FALSE_FLAGS = {"", "0", "false", "no", "n", "f"}


def valid(value):
    return value is not None and str(value).strip().lower() not in EMPTY


def month_index(month):
    if not month or not re.fullmatch(r"\d{4}-\d{2}", month):
        return None
    year, mm = map(int, month.split("-"))
    if not 1 <= mm <= 12:
        return None
    return 12 * year + mm


def load_baseline_months(root):
    prefix = "Baseline_Visit_Start_"
    pattern = re.compile(re.escape(prefix) + r"\d{2}[A-Za-z]{3}\d{4}\.csv$")
    files = [p for p in (root / "PPMI").glob(prefix + "*.csv") if pattern.fullmatch(p.name)]
    if len(files) != 1:
        raise ValueError("Expected one PPMI Baseline_Visit_Start CSV")
    anchors = defaultdict(set)
    with files[0].open(encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not {"PATNO", "EVENT_ID", "INFODT"}.issubset(reader.fieldnames or []):
            raise ValueError("Missing BL anchor columns")
        for row in reader:
            if (row["EVENT_ID"] or "").strip() != "BL":
                continue
            pid = (row["PATNO"] or "").strip()
            raw = (row["INFODT"] or "").strip()
            match = re.fullmatch(r"(\d{2})/(\d{4})", raw)
            normalized = None
            if match:
                mm, yy = map(int, match.groups())
                if 1 <= mm <= 12 and 1900 <= yy <= 2100:
                    normalized = f"{yy:04d}-{mm:02d}"
            if pid:
                anchors[pid].add(normalized)
    return {pid: next(iter(vals)) for pid, vals in anchors.items()
            if len(vals) == 1 and None not in vals}


def source_connection(path, engine):
    if engine == "duckdb":
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("Install duckdb: pip install duckdb") from exc
        return duckdb.connect(str(path), read_only=True)
    return sqlite3.connect("file:" + str(path.resolve()) + "?mode=ro", uri=True)


def load_data(db):
    cols = [entry[1] for entry in db.execute("PRAGMA table_info('observations')").fetchall()]
    if not {"exam_month", "date_precision", "source_file", "source_row"}.issubset(cols):
        raise ValueError("Requires Phase 8 v2 patient DB")
    labels = defaultdict(dict)
    for cohort, pid, label in db.execute(
        "SELECT cohort,patient_id,candidate_label FROM restricted_labels "
        "WHERE candidate_label IN ('PD','HC')").fetchall():
        if cohort in ACTIONS:
            labels[cohort][pid] = label
    obs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    cursor = db.execute("""SELECT cohort,patient_id,event_id,concept,value_raw,exam_month,
                               date_precision,quality_status,medication_state,imputed_flag,
                               source_file,source_row FROM observations
                            WHERE event_id IN ('BL','SC')""")
    while True:
        records = cursor.fetchmany(5000)
        if not records:
            break
        for cohort, pid, event, concept, value, month, precision, quality, med, imp, source, row in records:
            if pid not in labels.get(cohort, {}) or concept not in ACTIONS[cohort]:
                continue
            obs[(cohort, pid)][concept][event].append({
                "value": (value or "").strip(), "month": (month or "").strip(),
                "precision": (precision or "").strip(),
                "quality": (quality or "").strip(),
                "med": (med or "").strip(), "imputed": (imp or "").strip().lower(),
                "source": source, "row": row})
    return labels, obs


def state_matches(record, requested):
    if requested is None:
        return True
    state = record["med"].upper()
    return (state == requested if requested != "UNSPECIFIED" else state == "")


def select_measurement(records, anchor, cohort, visit, months, allow_undated=False):
    """Select only identical-quality observations; return status and one provenance row."""
    if not records:
        return "not_recorded", None
    good = []
    reasons = Counter()
    for row in records:
        if (not valid(row["value"]) or row["value"] in BAD_SENTINELS or
            row["quality"] != "observed_unverified" or row["imputed"] not in FALSE_FLAGS):
            reasons["quality_blocked"] += 1
            continue
        if not allow_undated:
            source_month = month_index(row["month"])
            target_month = month_index(anchor)
            if cohort == "PPMI" and target_month is not None:
                if source_month is None:
                    reasons["unknown_date"] += 1
                    continue
                delta = target_month - source_month
                if visit == "SC" and not 1 <= delta <= months:
                    reasons["SC_outside_preBL_window"] += 1
                    continue
                if visit == "BL" and delta < 0:
                    reasons["BL_later_than_visit_start"] += 1
                    continue
            elif visit == "SC":
                reasons["missing_BL_anchor"] += 1
                continue
        good.append(row)
    if not good:
        return sorted(reasons, key=lambda r: (-reasons[r], r))[0] if reasons else "quality_blocked", None
    # Even identical values with different dates or medication states require review.
    signatures = {(r["value"], r["month"], r["med"].upper()) for r in good}
    if len(signatures) != 1:
        return "conflicting_repeat", None
    first = min(good, key=lambda r: (str(r["source"]), int(r["row"])))
    return "ready", first


def action_name(concept, state):
    return concept if state is None else concept + "@" + state


def action_candidates(cohort, features, anchor, months):
    for concept, states in ACTIONS[cohort].items():
        for state in states:
            key = action_name(concept, state)
            records = features.get(concept, {})
            bl = [r for r in records.get("BL", []) if state_matches(r, state)]
            status, selected = select_measurement(bl, anchor, cohort, "BL", months,
                                                  allow_undated=(concept == "age_at_visit"))
            chosen_visit = "BL"
            # Screening is enabled ONLY for UPSIT and only if it precedes BL by 1..N months.
            if status != "ready" and concept == "upsit_total" and cohort == "PPMI":
                candidate = records.get("SC", [])
                fallback_status, fallback = select_measurement(candidate, anchor, cohort, "SC", months)
                if fallback_status == "ready":
                    status, selected, chosen_visit = fallback_status, fallback, "SC"
                elif status == "not_recorded":
                    status = fallback_status
            yield key, concept, status, chosen_visit, selected


def suppressed(n, threshold):
    return str(n) if n == 0 or n >= threshold else f"<{threshold}"


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(root, source_db, engine, manifest_db, report_dir, months=6, minimum=10):
    root, source_db = root.resolve(), source_db.resolve()
    manifest_db, report_dir = manifest_db.resolve(), report_dir.resolve()
    if not root.is_dir() or not source_db.is_file():
        raise ValueError("Missing raw data root or source DB")
    if (root in manifest_db.parents or root in report_dir.parents or manifest_db == source_db or
        manifest_db.exists()):
        raise ValueError("Use an existing raw data root and a NEW local manifest DB outside raw data")
    manifest_db.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    anchor = load_baseline_months(root)
    src = source_connection(source_db, engine)
    try:
        labels, observations = load_data(src)
    finally:
        src.close()
    dst = sqlite3.connect(manifest_db)
    try:
        dst.executescript("""
          CREATE TABLE episode_candidates (
              cohort TEXT NOT NULL, patient_id TEXT NOT NULL, baseline_month TEXT,
              task_name TEXT NOT NULL, PRIMARY KEY (cohort, patient_id));
          CREATE TABLE action_catalog (
              cohort TEXT NOT NULL, action_key TEXT NOT NULL,
              concept TEXT NOT NULL, required_medication_state TEXT,
              PRIMARY KEY (cohort,action_key));
          CREATE TABLE episode_actions (
              cohort TEXT NOT NULL, patient_id TEXT NOT NULL, action_key TEXT NOT NULL,
              status TEXT NOT NULL, selected_visit TEXT,
              source_file TEXT, source_row INTEGER, record_month TEXT,
              PRIMARY KEY (cohort, patient_id, action_key));
          CREATE TABLE metadata (name TEXT PRIMARY KEY, value TEXT);
        """)
        for cohort, choices in ACTIONS.items():
            dst.executemany("INSERT INTO action_catalog VALUES (?,?,?,?)", [
                (cohort, action_name(concept, state), concept, state) for concept, states in choices.items()
                for state in states])
        actions = []
        episode_rows = []
        statuses = Counter()
        coverage = defaultdict(set)
        label_total = Counter()
        anchors = Counter()
        for cohort, subjects in labels.items():
            for pid, label in subjects.items():
                a = anchor.get(pid) if cohort == "PPMI" else None
                episode_rows.append((cohort, pid, a, "retrospective_BL_package"))
                label_total[(cohort, label)] += 1
                if a:
                    anchors[(cohort, label)] += 1
                features = observations.get((cohort, pid), {})
                for key, concept, status, visit, row in action_candidates(cohort, features, a, months):
                    statuses[(cohort, label, key, status)] += 1
                    if status == "ready":
                        coverage[(cohort, label, pid)].add(key)
                    actions.append((cohort, pid, key, status, visit,
                                    row["source"] if row else None,
                                    row["row"] if row else None,
                                    row["month"] if row else None))
        dst.executemany("INSERT INTO episode_candidates VALUES (?,?,?,?)", episode_rows)
        dst.executemany("INSERT INTO episode_actions VALUES (?,?,?,?,?,?,?,?)", actions)
        dst.executemany("INSERT INTO metadata VALUES (?,?)", [
            ("task", "retrospective_BL_package"),
            ("clinical_status", "UNVALIDATED_not_diagnosis_time_reconstruction"),
            ("source_db", str(source_db)),
            ("created_utc", datetime.now(timezone.utc).isoformat()),
        ])
        dst.commit()
    finally:
        dst.close()
    report_rows = []
    for (cohort, label, key, status), num in sorted(statuses.items()):
        report_rows.append({"cohort": cohort, "label": label, "action": key,
                            "status": status, "patients": suppressed(num, minimum)})
    write_csv(report_dir / "action_status_by_label.csv",
              ["cohort", "label", "action", "status", "patients"], report_rows)
    joint = []
    for cohort in ACTIONS:
        for label in ("PD", "HC"):
            pids = labels.get(cohort, {})
            for name, required in (
                ("motor_any+moca+rbd", ("motor_any", "moca_total", "rbd_vivid_dreams")),
                ("motor_any+moca+rbd+upsit", ("motor_any", "moca_total", "rbd_vivid_dreams", "upsit_total")),
                ("motor_off+moca+rbd", ("motor_off", "moca_total", "rbd_vivid_dreams")),
            ):
                if cohort == "BioFIND" and ("upsit" in name or "motor_off" in name):
                    continue
                def satisfies(got, item):
                    if item == "motor_any":
                        names = (["updrs3_total@ON", "updrs3_total@OFF", "updrs3_total@UNSPECIFIED"]
                                 if cohort == "PPMI" else ["updrs3_gait_post_dose"])
                        return any(k in got for k in names)
                    if item == "motor_off":
                        return "updrs3_total@OFF" in got
                    return item in got
                n = sum(all(satisfies(coverage.get((cohort, label, pid), set()), r) for r in required)
                        for pid, y in pids.items() if y == label)
                joint.append({"cohort": cohort, "label": label, "feature_group": name,
                              "ready_patients": suppressed(n, minimum)})
    write_csv(report_dir / "joint_ready_by_label.csv",
              ["cohort", "label", "feature_group", "ready_patients"], joint)
    summary = {
        "status": "unvalidated_retrospective_snapshot_only",
        "source_schema": "v2_month_precision",
        "episodes": len(episode_rows), "candidate_label_counts": [
            {"cohort": c, "label": l, "patients": n, "baseline_month_anchor_patients": anchors[(c,l)]}
            for (c,l), n in sorted(label_total.items())],
        "actions_total": len(actions), "screening_window_months": months,
        "interpretation": [
            "This is an evidence-retrieval game on retrospectively completed records, not clinical test-time ordering.",
            "Patient labels are NOT stored in the manifest DB; they remain in the protected source DB.",
            "SC fallback is allowed for UPSIT only when strictly earlier than PPMI BL visit-start month.",
            "BL measurements can be same-month and do not establish ordering before diagnosis.",
            "Repeated scores, medication states and dates are never averaged.",
            "BioFIND has no verified BL month anchor and is retrospective BL-only.",
            "Source values/IDs are never written to reports; patient-level manifest remains on cluster.",
        ]}
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--source-engine", choices=("duckdb", "sqlite"), default="duckdb")
    parser.add_argument("--manifest-db", type=Path, default=Path("outputs/episode_manifests/episodes_v0.sqlite"))
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/episode_manifest_report"))
    parser.add_argument("--screening-window-months", type=int, default=6)
    parser.add_argument("--min-cell-count", type=int, default=10)
    args = parser.parse_args()
    if args.screening_window_months < 1 or args.min_cell_count < 10:
        parser.error("screening-window-months > 0 and min-cell-count >= 10 required")
    result = build(args.data_root, args.source_db, args.source_engine, args.manifest_db,
                   args.report_dir, args.screening_window_months, args.min_cell_count)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Upload ONLY aggregate report directory, NEVER the patient-level SQLite manifest.")


if __name__ == "__main__":
    main()
