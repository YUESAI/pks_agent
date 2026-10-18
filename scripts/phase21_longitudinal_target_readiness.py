#!/usr/bin/env python3
"""Phase 21: longitudinal target-readiness audit (aggregate-only).

For PPMI prodromal patients, distinguish candidate diagnosis-code events,
observable non-events and insufficient follow-up at 12/24/36 months.
Measure same-medication-state motor follow-up coverage in PPMI baseline groups.

NO definitive PD conversions, survival labels, patient IDs, raw dates or
motor scores are exported. Month-only dates and diagnosis codes remain
provisional. Never pass future measurements to a baseline Doctor model.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

EMPTY={"","none","nan","na","n/a","null","missing","."}
VISIT_PREFIXES={
    "enrollment":"Participant_Status_",
    "diagnosis":"Primary_Research_Diagnosis_",
    "baseline":"Baseline_Visit_Start_",
    "motor":"MDS-UPDRS_Part_III_",
}
HORIZONS=(12,24,36)
GROUPS=("prodromal","PD","HC")
MOTOR_STATES=("ON","OFF","UNSPECIFIED","OTHER")


def nonempty(v):
    return v is not None and str(v).strip().lower() not in EMPTY


def month_number(raw):
    matched=re.fullmatch(r"(\d{2})/(\d{4})",str(raw or "").strip())
    if not matched:return None
    mm,yyyy=map(int,matched.groups())
    return yyyy*12+mm if 1<=mm<=12 and 1900<=yyyy<=2100 else None


def one_file(directory,prefix):
    pat=re.compile(re.escape(prefix)+r"\d{2}[A-Za-z]{3}\d{4}\.csv$")
    found=[p for p in directory.glob(prefix+"*.csv") if pat.fullmatch(p.name)]
    if len(found)!=1:
        raise ValueError("Expected exactly one version of "+prefix)
    return found[0]


def read_rows(path,required,quality):
    """Resolve repeated headers only when populated values agree."""
    with path.open(newline="",encoding="utf-8-sig",errors="replace") as handle:
        reader=csv.reader(handle)
        pos=defaultdict(list)
        for i,col in enumerate(next(reader,[])):
            pos[col.strip()].append(i)
        if set(required)-set(pos):
            raise ValueError("Missing columns from "+path.name+": "+str(sorted(set(required)-set(pos))))
        names=("PATNO","EVENT_ID","COHORT","PRIMDIAG","INFODT",
               "EXAMDT","NP3TOT","PDSTATE")
        for record in reader:
            def get(name):
                vals=[record[i].strip() for i in pos.get(name,[]) if i<len(record)]
                populated=[x for x in vals if nonempty(x)]
                if len(set(populated))>1:
                    raise ValueError("Duplicate-header conflict")
                return populated[0] if populated else ""
            try:
                yield {name:get(name) for name in names if name in pos}
            except ValueError:
                quality["conflicting_duplicate_column_rows"]+=1


def single(values):
    good={x for x in values if x is not None}
    return next(iter(good)) if len(good)==1 else None


def group_name(cohort,code):
    if cohort=="4" and code=="25":return "prodromal"
    if cohort=="1" and code=="1":return "PD"
    if cohort=="2" and code=="17":return "HC"
    return None


def motor_state(raw):
    s=str(raw or "").strip().upper()
    return s if s in ("ON","OFF") else "UNSPECIFIED" if not s else "OTHER"


def show(n,k):
    return str(n) if n==0 or n>=k else "<"+str(k)


def save(path,headers,items):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=headers)
        writer.writeheader()
        writer.writerows(items)


def evaluate(root,out,minimum=10):
    root,out=Path(root).expanduser().resolve(),Path(out).expanduser().resolve()
    if not (root/"PPMI").is_dir() or root==out or root in out.parents:
        raise ValueError("Data root missing or output placed inside raw data")
    if minimum<10:
        raise ValueError("Small cell suppression must be >=10")
    files={k:one_file(root/"PPMI",v) for k,v in VISIT_PREFIXES.items()}
    qc=Counter()

    cohorts=defaultdict(set)
    for r in read_rows(files["enrollment"],("PATNO","COHORT"),qc):
        if nonempty(r["PATNO"]) and nonempty(r["COHORT"]):
            cohorts[r["PATNO"]].add(r["COHORT"])

    baseline_raw=defaultdict(set)
    diag_by_visit=defaultdict(lambda:defaultdict(lambda:defaultdict(set)))
    for r in read_rows(files["diagnosis"],("PATNO","EVENT_ID","PRIMDIAG","INFODT"),qc):
        pid,event=r["PATNO"],r["EVENT_ID"]
        if not nonempty(pid) or not nonempty(event):
            qc["diagnosis_rows_missing_keys"]+=1
            continue
        record=diag_by_visit[pid][event]
        record["codes"].add(r["PRIMDIAG"] if nonempty(r["PRIMDIAG"]) else None)
        record["months"].add(month_number(r["INFODT"]))
        if event=="BL":
            baseline_raw[pid].add(r["PRIMDIAG"] if nonempty(r["PRIMDIAG"]) else None)
    strict_anchor=defaultdict(set)
    for r in read_rows(files["baseline"],("PATNO","EVENT_ID","INFODT"),qc):
        if r["EVENT_ID"]=="BL" and nonempty(r["PATNO"]):
            strict_anchor[r["PATNO"]].add(month_number(r["INFODT"]))

    population={}
    for pid,candidates in baseline_raw.items():
        if len(cohorts.get(pid,set()))!=1 or len(candidates)!=1 or None in candidates:
            qc["ambiguous_baseline_cohort_or_diagnosis_patients"]+=1
            continue
        cohort=next(iter(cohorts[pid]))
        diagnosis=next(iter(candidates))
        group=group_name(cohort,diagnosis)
        if group:
            a=strict_anchor.get(pid,set())
            anchor=next(iter(a)) if len(a)==1 and None not in a else None
            population[pid]=(group,anchor)
    totals=Counter(group for group,anchor in population.values())
    anchors=Counter(group for group,anchor in population.values() if anchor is not None)

    # One code per patient+visit AND one diagnosis code per patient+month.
    # Mixed codes in a calendar month are ambiguous at month precision.
    follow_months=defaultdict(lambda:defaultdict(set))
    excluded_months=Counter()
    for pid,(group,anchor) in population.items():
        if anchor is None:
            continue
        for event,v in diag_by_visit.get(pid,{}).items():
            if event in ("BL","SC","CONSENT",""):
                continue
            if len(v["codes"])!=1 or None in v["codes"]:
                qc["ambiguous_followup_diagnosis_visits"]+=1
                continue
            if len(v["months"])!=1 or None in v["months"]:
                qc["undated_or_conflicting_followup_visits"]+=1
                continue
            measured=next(iter(v["months"]))
            if measured<=anchor:
                qc["followup_diagnosis_not_after_BL_month"]+=1
                continue
            follow_months[pid][measured].add(next(iter(v["codes"])))

    series={}
    for pid,(group,anchor) in population.items():
        if anchor is None:continue
        unique={}
        for measured,codes in follow_months.get(pid,{}).items():
            if len(codes)!=1:
                qc["discordant_diagnosis_months"]+=1
                excluded_months[pid]+=1
                continue
            unique[measured]=next(iter(codes))
        series[pid]=sorted(unique.items())

    transition=Counter()
    horizon_counts=Counter()
    candidate_ids=set()
    candidate_details=Counter()
    for pid,(group,anchor) in population.items():
        if group!="prodromal" or anchor is None:continue
        sequence=series[pid]
        last=sequence[-1][0] if sequence else None
        events=[(m,code) for m,code in sequence if code=="1"]
        first=events[0][0] if events else None
        if sequence:transition["prodromal_with_postBL_diagnosis"]+=1
        if excluded_months[pid]:
            transition["prodromal_with_ambiguous_followup_months"]+=1
        if first is not None:
            candidate_ids.add(pid)
            transition["first_later_PD_code"]+=1
            candidate_details["second_later_month_PD_code"]+=int(
                any(m>first and code=="1" for m,code in sequence))
            candidate_details["later_nonPD_code_after_first_PD"]+=int(
                any(m>first and code!="1" for m,code in sequence))
            candidate_details["last_observed_code_PD"]+=int(
                bool(sequence) and sequence[-1][1]=="1")
        for horizon in HORIZONS:
            if first is not None and first-anchor<=horizon:
                status="PD_code_seen_by_horizon"
            elif last is not None and last-anchor>=horizon:
                status="followup_through_horizon_no_PD_code"
            else:
                status="not_observable_to_horizon"
            horizon_counts[(horizon,status)]+=1

    # Motor progression feasibility: retain every source state, never average
    # ON/OFF, and block conflicting scores for one patient-month-state.
    raw_scores=defaultdict(set)
    for r in read_rows(files["motor"],("PATNO","EVENT_ID","NP3TOT"),qc):
        pid,event=r["PATNO"],r.get("EVENT_ID","")
        if pid not in population or population[pid][1] is None:
            continue
        if not nonempty(event) or event in ("SC","CONSENT"):
            continue
        if not nonempty(r["NP3TOT"]):
            continue
        try:value=float(r["NP3TOT"])
        except ValueError:
            qc["non_numeric_motor_score_rows"]+=1
            continue
        if not 0<=value<=132:
            qc["out_of_range_motor_score_rows"]+=1
            continue
        measured=month_number(r.get("EXAMDT") or r.get("INFODT"))
        if measured is None:
            qc["motor_exam_month_missing_rows"]+=1
            continue
        state=motor_state(r.get("PDSTATE"))
        raw_scores[(pid,measured,state)].add(value)

    motor_valid=defaultdict(lambda:defaultdict(set))
    for (pid,measured,state),values in raw_scores.items():
        if len(values)>1:
            qc["patient_month_state_conflicting_motor_scores"]+=1
            continue
        motor_valid[pid][state].add(measured)

    motor_counts=Counter()
    for pid,(group,anchor) in population.items():
        if anchor is None:continue
        for state in MOTOR_STATES:
            months=motor_valid.get(pid,{}).get(state,set())
            have_bl=anchor in months
            later={m for m in months if m>anchor}
            if have_bl:
                motor_counts[(group,state,"baseline_motor")]+=1
            if later:
                motor_counts[(group,state,"later_motor")]+=1
            if have_bl and later:
                motor_counts[(group,state,"baseline_plus_followup")]+=1
            if have_bl and any(m-anchor>=6 for m in later):
                motor_counts[(group,state,"baseline_plus_6mo")]+=1
            if have_bl and any(m-anchor>=12 for m in later):
                motor_counts[(group,state,"baseline_plus_12mo")]+=1
            if len(later)>=2:
                motor_counts[(group,state,"two_distinct_later_months")]+=1

    out.mkdir(parents=True,exist_ok=True)
    save(out/"candidate_conversion_support.csv",
         ["criterion","patients"],
         [{"criterion":name,"patients":show(transition[name],minimum)}
          for name in ("prodromal_with_postBL_diagnosis",
                       "prodromal_with_ambiguous_followup_months",
                       "first_later_PD_code")] +
         [{"criterion":name,"patients":show(candidate_details[name],minimum)}
          for name in ("second_later_month_PD_code",
                       "later_nonPD_code_after_first_PD",
                       "last_observed_code_PD")])
    save(out/"diagnosis_observability_horizons.csv",
         ["horizon_months","status","patients"],
         [{"horizon_months":h,"status":status,
           "patients":show(horizon_counts[(h,status)],minimum)}
          for h in HORIZONS for status in (
             "PD_code_seen_by_horizon",
             "followup_through_horizon_no_PD_code",
             "not_observable_to_horizon")])
    save(out/"longitudinal_motor_by_state.csv",
         ["baseline_group","medication_state","readiness","patients"],
         [{"baseline_group":group,"medication_state":state,"readiness":criterion,
           "patients":show(motor_counts[(group,state,criterion)],minimum)}
          for group in GROUPS for state in MOTOR_STATES for criterion in (
             "baseline_motor","later_motor","baseline_plus_followup",
             "baseline_plus_6mo","baseline_plus_12mo",
             "two_distinct_later_months")])
    save(out/"baseline_group_counts.csv",
         ["baseline_group","candidate_patients","strict_BL_month_anchor"],
         [{"baseline_group":grp,"candidate_patients":show(totals[grp],minimum),
           "strict_BL_month_anchor":show(anchors[grp],minimum)}
          for grp in GROUPS])
    save(out/"quality_flags.csv",["check","rows_or_patients"],
         [{"check":name,"rows_or_patients":show(n,minimum)}
          for name,n in sorted(qc.items())])
    result={
        "status":"aggregate_target_feasibility_no_training_labels",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "baseline_prodromal_candidate":totals["prodromal"],
        "baseline_prodromal_strict_month_anchor":anchors["prodromal"],
        "candidate_first_later_PD_code_patients":transition["first_later_PD_code"],
        "note":"Raw diagnostic code change is NOT clinically confirmed conversion.",
        "caveats":[
            "Horizon statuses are follow-up observability summaries, NOT survival labels.",
            "Missing follow-up is not a stable non-event. No formal censoring dates.",
            "Month precision forbids same-month diagnosis ordering.",
            "Conflicting codes in the same visit or calendar month are excluded.",
            "Repeat motor scores differing within patient-month-state are excluded.",
            "UNSPECIFIED medication state does not mean untreated or OFF.",
            "Future observations MUST NEVER be given to a baseline Doctor.",
            "Original clinical code dictionary and trial visit schedule need validation.",
            "No patient identifiers, dates, individual motor scores or labels are exported.",
        ],
    }
    (out/"summary.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,
                        default=Path("outputs/phase21_target_readiness"))
    parser.add_argument("--min-cell-count",type=int,default=10)
    args=parser.parse_args()
    result=evaluate(args.data_root,args.output_dir,args.min_cell_count)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print("Share only aggregate reports; never upload patient-level data.")


if __name__=="__main__":
    main()
