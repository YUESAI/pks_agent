#!/usr/bin/env python3
"""Phase 20. Read-only feasibility audit for longitudinal PPMI research.

Only aggregate counts are exported. Raw clinical scores, IDs and dates stay on
the cluster. Future diagnoses must never enter baseline Doctor observations.
MM/YYYY and PRIMDIAG codes are provisional until official definitions are
validated. A later PRIMDIAG=1 is NOT an adjudicated PD conversion event.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PREFIXES={
    "enrollment":"Participant_Status_",
    "baseline_start":"Baseline_Visit_Start_",
    "diagnosis":"Primary_Research_Diagnosis_",
    "motor":"MDS-UPDRS_Part_III_",
}
EMPTY={"","na","nan","n/a","null","none","missing","."}
EXCLUDED_EVENTS={"","SC","BL","CONSENT"}
GROUPS=("prodromal_concordant","cohort4_not_BL25","BL25_not_cohort4",
        "PD_concordant","HC_concordant","other_or_unlabeled")


def present(v):
    return v is not None and str(v).strip().lower() not in EMPTY


def parse_month(text):
    matched=re.fullmatch(r"(\d{2})/(\d{4})",str(text or "").strip())
    if not matched:
        return None
    month,year=map(int,matched.groups())
    return 12*year+month if 1<=month<=12 and 1900<=year<=2100 else None


def dated_file(folder,prefix):
    pattern=re.compile(re.escape(prefix)+r"\d{2}[A-Za-z]{3}\d{4}\.csv$")
    files=[f for f in folder.glob(prefix+"*.csv") if pattern.fullmatch(f.name)]
    if len(files)!=1:
        raise ValueError("Expected exactly one dated CSV for "+prefix)
    return files[0]


def read_columns(path,required,quality):
    """Reconcile duplicated headers only when all populated values agree."""
    with path.open(encoding="utf-8-sig",errors="replace",newline="") as handle:
        reader=csv.reader(handle)
        headers=[x.strip() for x in next(reader,[])]
        pos=defaultdict(list)
        for i,col in enumerate(headers):
            pos[col].append(i)
        if set(required)-set(pos):
            raise ValueError("Missing columns in "+path.name+": "+",".join(sorted(set(required)-set(pos))))
        columns=("PATNO","EVENT_ID","COHORT","PRIMDIAG",
                 "INFODT","EXAMDT","NP3TOT","PDSTATE")
        for row in reader:
            def get(name):
                vals=[row[i].strip() for i in pos.get(name,[]) if i<len(row)]
                defined=[v for v in vals if present(v)]
                if len(set(defined))>1:
                    raise ValueError("Conflicting duplicated column")
                return defined[0] if defined else ""
            try:
                yield {name:get(name) for name in columns if name in pos}
            except ValueError:
                quality["conflicting_duplicate_column_rows"]+=1


def classify(cohort,diagnosis):
    if cohort=="4" and diagnosis=="25":return "prodromal_concordant"
    if cohort=="4":return "cohort4_not_BL25"
    if diagnosis=="25":return "BL25_not_cohort4"
    if cohort=="1" and diagnosis=="1":return "PD_concordant"
    if cohort=="2" and diagnosis=="17":return "HC_concordant"
    return "other_or_unlabeled"


def time_bin(delta):
    if delta<6:return "01_to_05_months"
    if delta<12:return "06_to_11_months"
    if delta<24:return "12_to_23_months"
    if delta<60:return "24_to_59_months"
    return "60plus_months"


def show(n,min_cell):
    return str(n) if n==0 or n>=min_cell else "<"+str(min_cell)


def save(path,fields,rows):
    with path.open("w",newline="",encoding="utf-8") as handle:
        w=csv.DictWriter(handle,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def audit(root,out,min_cell=10):
    root=Path(root).expanduser().resolve()
    out=Path(out).expanduser().resolve()
    if not (root/"PPMI").is_dir() or out==root or root in out.parents:
        raise ValueError("Invalid data root or output is inside source directory")
    if min_cell<10:
        raise ValueError("Small cell suppression must be >=10")
    folder=root/"PPMI"
    paths={name:dated_file(folder,prefix) for name,prefix in PREFIXES.items()}
    quality=Counter()

    enrollment=defaultdict(set)
    for row in read_columns(paths["enrollment"],("PATNO","COHORT"),quality):
        if present(row["PATNO"]) and present(row["COHORT"]):
            enrollment[row["PATNO"]].add(row["COHORT"])
    cohort_codes={}
    for pid,codes in enrollment.items():
        if len(codes)==1:cohort_codes[pid]=next(iter(codes))
        else:quality["conflicting_enrollment_codes_patients"]+=1

    raw_anchors=defaultdict(set)
    for row in read_columns(paths["baseline_start"],
                            ("PATNO","EVENT_ID","INFODT"),quality):
        if row["EVENT_ID"]=="BL" and present(row["PATNO"]):
            parsed=parse_month(row["INFODT"])
            raw_anchors[row["PATNO"]].add(parsed)
            if parsed is None:
                quality["BL_start_rows_missing_or_unparsed_month"]+=1
    anchors={}
    for pid,months in raw_anchors.items():
        if len(months)==1 and None not in months:
            anchors[pid]=next(iter(months))
        elif len(months)>1:
            quality["ambiguous_baseline_month_patients"]+=1

    # Build one diagnosis code and one month per PATNO/EVENT_ID.
    # Ambiguous patient/visit combinations are quarantined.
    raw_visits=defaultdict(lambda:defaultdict(lambda:{"codes":set(),"months":set()}))
    for row in read_columns(paths["diagnosis"],
                            ("PATNO","EVENT_ID","PRIMDIAG","INFODT"),quality):
        pid,event=row["PATNO"],row["EVENT_ID"]
        if not present(pid) or not present(event):
            quality["diagnosis_rows_missing_patient_or_visit"]+=1
            continue
        v=raw_visits[pid][event]
        if present(row["PRIMDIAG"]):
            v["codes"].add(row["PRIMDIAG"])
        v["months"].add(parse_month(row["INFODT"]))
    baseline={}
    follow=defaultdict(list)
    for pid,events in raw_visits.items():
        for event,v in events.items():
            if len(v["codes"])!=1:
                quality["empty_or_conflicting_visit_diagnoses"]+=1
                continue
            code=next(iter(v["codes"]))
            if event=="BL":
                baseline[pid]=code
                continue
            if event in EXCLUDED_EVENTS:
                continue
            if len(v["months"])!=1 or None in v["months"]:
                quality["undated_or_conflicting_followup_visit_months"]+=1
                continue
            follow[pid].append((next(iter(v["months"])),event,code))

    people=set(cohort_codes)|set(baseline)|set(anchors)
    groups={pid:classify(cohort_codes.get(pid),baseline.get(pid)) for pid in people}
    total=Counter(groups.values())
    coverage=Counter()
    candidate_bins=Counter()
    follow_codes=Counter()
    anchored={}
    for pid,grp in groups.items():
        if pid not in anchors:
            coverage[(grp,"anchor_missing")]+=1
            continue
        anchor=anchors[pid]
        anchored[pid]=(anchor,grp)
        coverage[(grp,"BL_month_present")]+=1
        later=[(m,event,code) for m,event,code in follow.get(pid,[]) if m>anchor]
        if later:
            coverage[(grp,"dated_later_diagnosis")]+=1
        if len({m for m,event,code in later})>=2:
            coverage[(grp,"two_or_more_followup_months")]+=1
        if later and max(m for m,event,code in later)-anchor>=12:
            coverage[(grp,"followup_span_12plus_months")]+=1
        if later and max(m for m,event,code in later)-anchor>=24:
            coverage[(grp,"followup_span_24plus_months")]+=1
        for m,event,code in later:
            follow_codes[(grp,code)]+=1
        # Strict inclusion only for compatible prodromal baseline coding.
        pd_months=sorted(m for m,event,code in later if code=="1")
        if grp=="prodromal_concordant" and pd_months:
            coverage[(grp,"later_PD_code_candidate")]+=1
            candidate_bins[time_bin(pd_months[0]-anchor)]+=1
            if any(m>pd_months[0] and code!="1" for m,event,code in later):
                quality["candidate_transitions_with_later_nonPD_code"]+=1

    motor_months=defaultdict(set)
    motor_state=defaultdict(set)
    for row in read_columns(paths["motor"],
                            ("PATNO","EVENT_ID","NP3TOT"),quality):
        pid,event=row["PATNO"],row.get("EVENT_ID","")
        if pid not in anchored or event in EXCLUDED_EVENTS:
            continue
        try:
            score=float(row["NP3TOT"])
        except (ValueError,TypeError):
            continue
        if not 0<=score<=132:
            quality["invalid_postbaseline_motor_total_rows"]+=1
            continue
        anchor,grp=anchored[pid]
        raw_date=row.get("EXAMDT") or row.get("INFODT") or ""
        m=parse_month(raw_date)
        if m is None:
            quality["motor_followup_rows_missing_month"]+=1
        elif m>anchor:
            motor_months[pid].add(m)
            motor_state[(pid,event)].add(row.get("PDSTATE") or "UNSPECIFIED")
        elif m==anchor:
            quality["motor_followup_same_BL_month_uncertain_order"]+=1
    for pid,(anchor,grp) in anchored.items():
        months=motor_months.get(pid,set())
        if months:coverage[(grp,"later_month_motor")]+=1
        if len(months)>=2:coverage[(grp,"two_or_more_motor_months")]+=1
        if months and max(months)-anchor>=12:
            coverage[(grp,"motor_span_12plus_months")]+=1
    for values in motor_state.values():
        if len(values)>1:
            quality["followup_motor_patient_visit_multiple_states"]+=1

    rows=[]
    for grp in GROUPS:
        rows.append({
            "baseline_group":grp,
            "baseline_patients":show(total[grp],min_cell),
            "strict_BL_month":show(coverage[(grp,"BL_month_present")],min_cell),
            "later_dated_diagnosis":show(coverage[(grp,"dated_later_diagnosis")],min_cell),
            "diagnosis_followup_12plus_months":show(coverage[(grp,"followup_span_12plus_months")],min_cell),
            "two_plus_diagnosis_months":show(coverage[(grp,"two_or_more_followup_months")],min_cell),
            "later_motor_available":show(coverage[(grp,"later_month_motor")],min_cell),
            "two_plus_motor_months":show(coverage[(grp,"two_or_more_motor_months")],min_cell),
        })
    out.mkdir(parents=True,exist_ok=True)
    save(out/"longitudinal_feasibility.csv",
         ["baseline_group","baseline_patients","strict_BL_month",
          "later_dated_diagnosis","diagnosis_followup_12plus_months",
          "two_plus_diagnosis_months","later_motor_available","two_plus_motor_months"],rows)
    prod="prodromal_concordant"
    transition_counts=[
        ("cohort4_and_BL_PRIMDIAG25",total[prod]),
        ("with_strict_BL_visit_start_month",coverage[(prod,"BL_month_present")]),
        ("with_later_dated_diagnosis_visit",coverage[(prod,"dated_later_diagnosis")]),
        ("with_later_PRIMDIAG1_candidate",coverage[(prod,"later_PD_code_candidate")]),
    ]
    save(out/"prodromal_transition_summary.csv",["criterion","patients"],
         [{"criterion":k,"patients":show(n,min_cell)} for k,n in transition_counts])
    save(out/"transition_time_bins.csv",["elapsed_month_bin","patients"],
         [{"elapsed_month_bin":key,"patients":show(n,min_cell)}
          for key,n in sorted(candidate_bins.items())])
    save(out/"raw_followup_code_counts.csv",
         ["baseline_group","followup_diagnosis_code","visit_records"],
         [{"baseline_group":grp,"followup_diagnosis_code":code,
           "visit_records":show(n,min_cell)}
          for (grp,code),n in sorted(follow_codes.items()) if n>=min_cell])
    save(out/"quality_flags.csv",["metric","count"],
         [{"metric":key,"count":show(n,min_cell)}
          for key,n in sorted(quality.items())])
    summary={
        "status":"aggregate_longitudinal_feasibility_not_conversion_labels",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "source_files":{k:v.name for k,v in paths.items()},
        "strict_prodromal_baseline":"COHORT 4 AND BL PRIMDIAG 25",
        "candidate_later_diagnosis":"subsequent event with PRIMDIAG 1 and strictly later month",
        "date_precision":"MM/YYYY provisional; no same-month ordering inferred",
        "notes":[
            "A diagnosis code change is not clinically confirmed incident Parkinson disease.",
            "Missing follow-up cannot be interpreted as absence of conversion.",
            "Later diagnosis codes must NEVER enter baseline Doctor inputs.",
            "Do not create survival labels until censoring, visit schedule and codes are verified.",
            "Motor measures are only counted for coverage and not averaged over drug states.",
            "Reports suppress small cells and contain no IDs, clinical scores or raw dates.",
            "If longitudinal endpoints are sparse, do not force a conversion modeling task.",
        ],
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    lines=["# Phase 20 longitudinal feasibility","",
           "Exploratory, aggregate-only. No confirmed conversion labels.","",
           "| Baseline group | Patients | Dated later diagnosis | Motor at 2+ months |",
           "|---|---:|---:|---:|"]
    for row in rows:
        lines.append("| {baseline_group} | {baseline_patients} | {later_dated_diagnosis} | {two_plus_motor_months} |".format(**row))
    lines+=["","Review official codes, visit schedule, censoring and longitudinal",
            "follow-up before constructing World Model targets.",""]
    (out/"report.md").write_text("\n".join(lines),encoding="utf-8")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,
                   default=Path("outputs/phase20_longitudinal"))
    p.add_argument("--min-cell-count",type=int,default=10)
    args=p.parse_args()
    print(json.dumps(audit(args.data_root,args.output_dir,args.min_cell_count),
                     indent=2,ensure_ascii=False))
    print("Share only aggregated Phase20 reports; never upload raw patient files.")


if __name__=="__main__":
    main()
