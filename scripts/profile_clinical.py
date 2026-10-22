#!/usr/bin/env python3
"""Full-row clinical coverage profiler for three Parkinson cohorts.

Reads only selected clinical CSV tables. Patient IDs are used in memory for
counts and are never exported. Output must be outside the raw data tree.

python scripts/profile_clinical.py --data-root /scratch/ylong66/data/pks/data \
  --output-dir ./outputs/clinical_profile
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# (cohort, name, filename prefix, patient key, visit key, concept=>source columns, anchor)
SPECS = [
 ("PPMI","participant_status","Participant_Status_","PATNO",None,
  {"cohort_label":"COHORT","cohort_description":"COHORT_DEFINITION"},True),
 ("PPMI","primary_diagnosis","Primary_Research_Diagnosis_","PATNO","EVENT_ID",
  {"diagnosis_label":"PRIMDIAG"},False),
 ("PPMI","clinical_diagnosis","Clinical_Diagnosis_","PATNO","EVENT_ID",
  {"diagnosis_label":"NEWDIAG"},False),
 ("PPMI","age","Age_at_visit_","PATNO","EVENT_ID",
  {"age_at_visit":"AGE_AT_VISIT"},False),
 ("PPMI","demographics","Demographics_","PATNO","EVENT_ID",
  {"sex":"SEX"},False),
 ("PPMI","motor","MDS-UPDRS_Part_III_","PATNO","EVENT_ID",
  {"updrs3_total":"NP3TOT","updrs3_gait":"NP3GAIT",
   "updrs3_bradykinesia":"NP3BRADY","medication_state":"PDSTATE"},False),
 ("PPMI","motor_part_i","MDS-UPDRS_Part_I_","PATNO","EVENT_ID",
  {"updrs1_cognition":"NP1COG"},False),
 ("PPMI","motor_part_ii","MDS_UPDRS_Part_II__Patient_Questionnaire_","PATNO","EVENT_ID",
  {"updrs2_speech":"NP2SPCH"},False),
 ("PPMI","moca","Montreal_Cognitive_Assessment__MoCA__","PATNO","EVENT_ID",
  {"moca_total":"MCATOT"},False),
 ("PPMI","upsit","University_of_Pennsylvania_Smell_Identification_Test_UPSIT_","PATNO","EVENT_ID",
  {"upsit_total":"TOTAL_CORRECT","upsit_imputed":"IMPUTED_DATA"},False),
 ("PPMI","rbd","REM_Sleep_Behavior_Disorder_Screening_Questionnaire_","PATNO","EVENT_ID",
  {"rbd_vivid_dreams":"DRMVIVID","rbd_sleep_injury":"SLPINJUR"},False),
 ("PPMI","scopa","SCOPA-AUT_","PATNO","EVENT_ID",
  {"scopa_item1":"SCAU1"},False),
 ("PPMI","diagnosis_history","PD_Diagnosis_History_","PATNO","EVENT_ID",
  {"symptom_onset_date":"SXDT","diagnosis_date":"PDDXDT"},False),
 ("BioFIND","screening","Screening_Demographics_","PATNO","EVENT_ID",
  {"screening_diagnosis_label":"APPRDX","sex":"GENDER"},True),
 ("BioFIND","primary_diagnosis","Primary_Diagnosis_","PATNO","EVENT_ID",
  {"diagnosis_label":"PRIMDIAG"},False),
 ("BioFIND","motor_post_dose","MDS_UPDRS_Part_III__Post_Dose__","PATNO","EVENT_ID",
  {"gait_post_dose":"NP3GAIT","bradykinesia_post_dose":"NP3BRADY"},False),
 ("BioFIND","motor_part_i","MDS_UPDRS_Part_I_","PATNO","EVENT_ID",
  {"updrs1_cognition":"NP1COG"},False),
 ("BioFIND","motor_part_ii","MDS_UPDRS_Part_II__Patient_Questionnaire_","PATNO","EVENT_ID",
  {"updrs2_speech":"NP2SPCH"},False),
 ("BioFIND","moca","Montreal_Cognitive_Assessment__MoCA__","PATNO","EVENT_ID",
  {"moca_total":"MCATOT"},False),
 ("BioFIND","rbd","REM_Sleep_Disorder_Questionnaire_","PATNO","EVENT_ID",
  {"rbd_vivid_dreams":"DRMVIVID","rbd_sleep_injury":"SLPINJUR"},False),
 ("BioFIND","medical_history","General_Medical_History_","PATNO","EVENT_ID",
  {"medical_history_text":"MHTERM"},False),
 ("FoxInsight","users","Users","fox_insight_id",None,
  {"self_reported_pd_at_enrollment":"InitPDDiag","age_at_enrollment":"AgeAtEnrollment"},True),
 ("FoxInsight","general","General","fox_insight_id",None,
  {"self_reported_current_pd":"CurrPDDiag"},False),
 ("FoxInsight","movement","Movement","fox_insight_id","days_elapsed",
  {"reported_tremor":"MoveTremor","reported_walking":"MoveWalk"},False),
 ("FoxInsight","nonmovement","NonMovement","fox_insight_id","days_elapsed",
  {"reported_smell":"NonMoveSmell","reported_sleep":"NonMoveSleep"},False),
 ("FoxInsight","return_control","ReturnCTR","fox_insight_id","days_elapsed",
  {"self_reported_new_pd":"NewPDDiagCTR"},False),
]
LABEL_CONCEPTS = {
 "cohort_label","diagnosis_label","screening_diagnosis_label",
 "self_reported_pd_at_enrollment","self_reported_current_pd","self_reported_new_pd",
}
LEAK_CONCEPTS = LABEL_CONCEPTS | {"cohort_description","diagnosis_date","symptom_onset_date"}
MISSING = {"", "na", "n/a", "null", "none", "nan", "missing", "."}

def find_table(folder, cohort, prefix):
    if cohort == "FoxInsight":
        matches = [folder / (prefix + ".csv")]
        matches = [p for p in matches if p.is_file()]
    else:
        pat = re.compile(re.escape(prefix) + r"\d{2}[A-Za-z]{3}\d{4}\.csv$")
        matches = [p for p in folder.glob(prefix + "*.csv") if pat.fullmatch(p.name)]
    if len(matches) != 1:
        raise ValueError("Expected exactly one table for %s; found %d" % (prefix, len(matches)))
    return matches[0]

def good(value):
    return value is not None and str(value).strip().lower() not in MISSING

def scan(path, pid, visit, features, row_limit):
    patients = set()
    keys = set()
    duplicate_rows = 0
    missing_pid = 0
    missing_visit = 0
    rows = 0
    present_rows = Counter()
    feature_patients = defaultdict(set)
    feature_visits = defaultdict(set)
    cat_counts = defaultdict(Counter)
    events = Counter()
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader)
        if len(headers) != len(set(headers)):
            raise ValueError("Duplicate raw headers; requires manual handling")
        pos = {col: i for i, col in enumerate(headers)}
        required = [pid] + ([visit] if visit else [])
        if any(c not in pos for c in required):
            raise ValueError("Missing required join key: " + ",".join(c for c in required if c not in pos))
        available = {concept: col for concept, col in features.items() if col in pos}
        missing_features = [col for col in features.values() if col not in pos]
        def cell(row, col):
            i = pos[col]
            return row[i].strip() if i < len(row) else ""
        for row in reader:
            if row_limit and rows >= row_limit:
                break
            rows += 1
            subject = cell(row, pid)
            if not good(subject):
                missing_pid += 1
                continue
            patients.add(subject)
            event = cell(row, visit) if visit else ""
            if visit and not good(event):
                missing_visit += 1
                event = ""
            if good(event):
                events[event] += 1
            pair = (subject, event)
            if pair in keys:
                duplicate_rows += 1
            keys.add(pair)
            for concept, field in available.items():
                value = cell(row, field)
                if not good(value):
                    continue
                present_rows[concept] += 1
                feature_patients[concept].add(subject)
                if good(event):
                    feature_visits[concept].add(pair)
                if concept in LABEL_CONCEPTS:
                    cat_counts[concept][value] += 1
    return dict(
        n_rows=rows, patient_ids=patients, unique_keys=len(keys), duplicates=duplicate_rows,
        missing_pid=missing_pid, missing_visit=missing_visit, feature_patients=feature_patients,
        feature_visits=feature_visits, feature_rows=present_rows, categories=cat_counts,
        events=events, missing_features=missing_features,
    )

def write_table(path, fields, entries):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(entries)

def profile(root, out, cohorts, row_limit, threshold):
    out.mkdir(parents=True, exist_ok=True)
    anchors = {}
    tables, features, diagnoses, event_rows, issues = [], [], [], [], []
    specs = sorted((s for s in SPECS if s[0] in cohorts), key=lambda s: not s[6])
    for cohort, name, prefix, pid, visit, mapping, anchor in specs:
        try:
            path = find_table(root / cohort, cohort, prefix)
            stats = scan(path, pid, visit, mapping, row_limit)
        except (OSError, UnicodeError, csv.Error, ValueError, StopIteration) as exc:
            issues.append(dict(cohort=cohort, table=name, error_type=type(exc).__name__))
            continue
        if anchor:
            anchors[cohort] = stats["patient_ids"]
        ref = anchors.get(cohort, set())
        nref = len(ref)
        overlapping = len(stats["patient_ids"] & ref)
        tables.append(dict(
            cohort=cohort, table=name, relative_path=path.relative_to(root / cohort).as_posix(),
            role="anchor" if anchor else "feature_or_label", patient_key=pid, visit_key=visit or "",
            rows_scanned=stats["n_rows"], distinct_patients=len(stats["patient_ids"]),
            distinct_patient_visit_keys=stats["unique_keys"],
            repeated_patient_visit_rows=stats["duplicates"],
            missing_patient_id_rows=stats["missing_pid"],
            missing_visit_id_rows=stats["missing_visit"],
            patients_in_anchor=overlapping if nref else "",
            anchor_patient_count=nref or "",
            pct_anchor_patients=round(100 * overlapping / nref, 2) if nref else "",
            missing_expected_columns="|".join(stats["missing_features"]),
        ))
        for concept, col in mapping.items():
            if col in stats["missing_features"]:
                continue
            members = stats["feature_patients"][concept]
            features.append(dict(
                cohort=cohort, table=name, concept_hint=concept, raw_column=col,
                rows_with_value=stats["feature_rows"][concept],
                distinct_patients_with_value=len(members),
                distinct_patient_visits_with_value=len(stats["feature_visits"][concept]) if visit else "",
                pct_of_table_patients=round(100 * len(members) / len(stats["patient_ids"]), 2)
                     if stats["patient_ids"] else "",
                pct_of_anchor_patients=round(100 * len(members & ref) / nref, 2) if nref else "",
                diagnosis_or_leakage_candidate="yes" if concept in LEAK_CONCEPTS else "no",
                interpretation_status="UNVALIDATED",
            ))
            for value, count in stats["categories"][concept].most_common():
                if count >= threshold and len(value) <= 64:
                    diagnoses.append(dict(cohort=cohort, table=name, raw_column=col,
                                          category=value, n_rows=count))
        for event, count in stats["events"].most_common(30):
            if count >= threshold and len(event) <= 32:
                event_rows.append(dict(cohort=cohort, table=name, event_code=event, n_rows=count))
    write_table(out / "table_coverage.csv",
                ["cohort","table","relative_path","role","patient_key","visit_key","rows_scanned",
                 "distinct_patients","distinct_patient_visit_keys","repeated_patient_visit_rows",
                 "missing_patient_id_rows","missing_visit_id_rows","patients_in_anchor",
                 "anchor_patient_count","pct_anchor_patients","missing_expected_columns"], tables)
    write_table(out / "feature_coverage.csv",
                ["cohort","table","concept_hint","raw_column","rows_with_value",
                 "distinct_patients_with_value","distinct_patient_visits_with_value",
                 "pct_of_table_patients","pct_of_anchor_patients",
                 "diagnosis_or_leakage_candidate","interpretation_status"], features)
    write_table(out / "label_distributions.csv",
                ["cohort","table","raw_column","category","n_rows"], diagnoses)
    write_table(out / "event_distributions.csv",
                ["cohort","table","event_code","n_rows"], event_rows)
    write_table(out / "issues.csv", ["cohort","table","error_type"], issues)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root), "scan": "full" if row_limit == 0 else "first_%d_rows" % row_limit,
        "anchor_patient_counts": {k: len(v) for k, v in anchors.items()},
        "tables_processed": len(tables), "features_profiled": len(features),
        "issues": len(issues),
        "interpretation": [
            "Patient IDs are used in memory and not exported.",
            "Any-visit patient coverage is not same-visit feature intersection.",
            "Repeated key rows may be legitimate; do not merge without visit and exam-state review.",
            "Clinical diagnosis, cohort labels and diagnosis history must be hidden from Doctor.",
            "Fox Insight diagnosis is self-report; cross-cohort labels are NOT yet harmonized.",
            "Missing-value sentinel codes are not yet decoded.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    lines = ["# Clinical coverage report", "", "This is a structural profile, NOT validated diagnosis data.",
             "", "Anchor subjects: " + json.dumps(summary["anchor_patient_counts"]), "",
             "| Cohort | Table | Patients | Anchor coverage | Repeated patient-visit rows |",
             "|---|---|---:|---:|---:|"]
    for table in tables:
        lines.append("| {cohort} | {table} | {distinct_patients} | {pct_anchor_patients}% | {repeated_patient_visit_rows} |".format(**table))
    lines += ["", "Review feature_coverage.csv, label_distributions.csv, event_distributions.csv, and issues.csv.",
              "The outputs contain only aggregate counts and source field names; review before sharing."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/clinical_profile"))
    parser.add_argument("--cohorts", nargs="+", choices=["PPMI","BioFIND","FoxInsight"],
                        default=["PPMI","BioFIND","FoxInsight"])
    parser.add_argument("--max-rows", type=int, default=0,
                        help="0 (default): ALL rows in selected core CSV tables")
    parser.add_argument("--min-category-count", type=int, default=10,
                        help="Minimum shown aggregated category size; must be >=10")
    args = parser.parse_args()
    if args.max_rows < 0 or args.min_category_count < 10:
        parser.error("max-rows >= 0 and min-category-count >= 10 are required")
    root, out = args.data_root.resolve(), args.output_dir.resolve()
    if not root.is_dir():
        parser.error("Raw data directory does not exist")
    if out == root or root in out.parents:
        parser.error("Output must not be inside raw dataset folder")
    summary = profile(root, out, args.cohorts, args.max_rows, args.min_category_count)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Reports:", out)

if __name__ == "__main__":
    main()
