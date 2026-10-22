#!/usr/bin/env python3
"""Phase22: strict longitudinal motor pairs within 6/12/24-month windows.

Read-only raw PPMI CSVs. All outputs are aggregates. Source patient IDs,
scores, dates and paired rows never leave the authorized cluster.

Baseline measurement MUST be EVENT_ID=BL and its EXAMDT/INFODT month must
match BL Visit Start. A follow-up measurement MUST NOT be BL/SC/CONSENT
and its exam month must be strictly later. ON, OFF, UNSPECIFIED are not
interchangeable; UNSPECIFIED does NOT prove the same treatment condition.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean,median

FILES={"enrollment":"Participant_Status_","diagnosis":"Primary_Research_Diagnosis_",
       "anchor":"Baseline_Visit_Start_","motor":"MDS-UPDRS_Part_III_"}
WINDOWS=((6,2),(12,3),(24,4))
GROUPS=("prodromal","PD","HC")
STATES=("UNSPECIFIED","ON","OFF")
MISSING={"","na","n/a","nan","null","none","missing","."}
EXCLUDE_EVENTS={"","BL","SC","CONSENT"}

def present(v):
    return v is not None and str(v).strip().lower() not in MISSING

def parse_month(value):
    matched=re.fullmatch(r"(\d{2})/(\d{4})",str(value or "").strip())
    if not matched:return None
    m,y=map(int,matched.groups())
    return 12*y+m if 1<=m<=12 and 1900<=y<=2100 else None

def one_file(folder,prefix):
    pat=re.compile(re.escape(prefix)+r"\d{2}[A-Za-z]{3}\d{4}\.csv$")
    matches=[p for p in folder.glob(prefix+"*.csv") if pat.fullmatch(p.name)]
    if len(matches)!=1:raise ValueError("Expected one version of "+prefix)
    return matches[0]

def records(path,required,issues):
    with path.open(encoding="utf-8-sig",errors="replace",newline="") as f:
        reader=csv.reader(f);columns=defaultdict(list)
        for i,name in enumerate(next(reader,[])):columns[name.strip()].append(i)
        if set(required)-set(columns):
            raise ValueError("Missing required columns from "+path.name)
        names=("PATNO","EVENT_ID","COHORT","PRIMDIAG",
               "INFODT","EXAMDT","NP3TOT","PDSTATE")
        for row in reader:
            def get(field):
                found=[row[i].strip() for i in columns.get(field,[]) if i<len(row)]
                values=[x for x in found if present(x)]
                if len(set(values))>1:raise ValueError("Conflicting duplicate headers")
                return values[0] if values else ""
            try:yield {field:get(field) for field in names if field in columns}
            except ValueError:issues["conflicting_duplicate_header_rows"]+=1

def baseline_group(cohort,diag):
    if cohort=="4" and diag=="25":return "prodromal"
    if cohort=="1" and diag=="1":return "PD"
    if cohort=="2" and diag=="17":return "HC"
    return None

def state_name(value):
    code=str(value or "").strip().upper()
    if code in ("ON","OFF"):return code
    return "UNSPECIFIED" if not code else "OTHER"

def safe(n,threshold):
    return str(n) if n==0 or n>=threshold else "<"+str(threshold)

def save(path,columns,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

def target_for_window(month_to_score,baseline,horizon,tolerance):
    """Reject tied nearest months rather than choosing arbitrarily."""
    possible=sorted((abs(month-baseline-horizon),month,score)
                    for month,score in month_to_score.items()
                    if month>baseline and abs(month-baseline-horizon)<=tolerance)
    if not possible:return "no_in_window_target",None
    if len(possible)>1 and possible[0][0]==possible[1][0]:
        return "equidistant_months",None
    distance,month,value=possible[0]
    return "paired",(month,value)

def quartile(values,quantile):
    ordered=sorted(values)
    at=(len(ordered)-1)*quantile
    index=int(at)
    return ordered[index]*(1-(at-index))+ordered[min(index+1,len(ordered)-1)]*(at-index)

def audit(root,out,min_cell=10):
    root=Path(root).expanduser().resolve();out=Path(out).expanduser().resolve()
    if not (root/"PPMI").is_dir() or root==out or root in out.parents:
        raise ValueError("Invalid root or report directory inside private raw data")
    if min_cell<10:raise ValueError("Small cell suppression must be >=10")
    files={name:one_file(root/"PPMI",prefix) for name,prefix in FILES.items()}
    issues=Counter();cohorts=defaultdict(set);diagnoses=defaultdict(set)
    months=defaultdict(set)
    for row in records(files["enrollment"],("PATNO","COHORT"),issues):
        if present(row["PATNO"]) and present(row["COHORT"]):
            cohorts[row["PATNO"]].add(row["COHORT"])
    for row in records(files["diagnosis"],("PATNO","EVENT_ID","PRIMDIAG"),issues):
        if row.get("EVENT_ID")=="BL" and present(row["PATNO"]):
            diagnoses[row["PATNO"]].add(row["PRIMDIAG"] if present(row["PRIMDIAG"]) else None)
    for row in records(files["anchor"],("PATNO","EVENT_ID","INFODT"),issues):
        if row["EVENT_ID"]=="BL" and present(row["PATNO"]):
            months[row["PATNO"]].add(parse_month(row["INFODT"]))

    people={};group_totals=Counter();unanchored=Counter()
    for pid in set(cohorts)|set(diagnoses):
        codes=cohorts.get(pid,set());baseline=diagnoses.get(pid,set())
        if len(codes)!=1 or len(baseline)!=1 or None in baseline:
            issues["ambiguous_baseline_group_patients"]+=1
            continue
        group=baseline_group(next(iter(codes)),next(iter(baseline)))
        if group is None:continue
        group_totals[group]+=1
        anchor=months.get(pid,set())
        if len(anchor)!=1 or None in anchor:
            unanchored[group]+=1
            continue
        people[pid]=(group,next(iter(anchor)))
    issues["missing_month_anchors_among_baseline_groups"]=sum(unanchored.values())

    # Keep every patient, visit-class, month and medication state separate.
    raw=defaultdict(lambda:{"values":set(),"invalid":False,"events":set()})
    for row in records(files["motor"],("PATNO","EVENT_ID","NP3TOT"),issues):
        pid,event=row.get("PATNO",""),row.get("EVENT_ID","")
        if pid not in people or not present(event):continue
        group,anchor=people[pid]
        m=parse_month(row.get("EXAMDT") or row.get("INFODT"))
        if m is None:
            issues["motor_rows_missing_exam_month"]+=1
            continue
        if event=="BL":
            if m!=anchor:
                issues["BL_event_exam_month_mismatch"]+=1
                continue
            visit_class="baseline"
        elif event in EXCLUDE_EVENTS:
            continue
        elif m<=anchor:
            issues["followup_event_at_or_before_anchor_month"]+=1
            continue
        else:visit_class="followup"
        state=state_name(row.get("PDSTATE"))
        if state=="OTHER":
            issues["motor_rows_unknown_medication_code"]+=1
            continue
        key=(pid,visit_class,m,state)
        entry=raw[key]
        entry["events"].add(event)
        try:
            score=float(row.get("NP3TOT",""))
            if not 0<=score<=132:raise ValueError("Invalid motor range")
        except (TypeError,ValueError):
            issues["motor_rows_non_numeric_or_out_of_range"]+=1
            entry["invalid"]=True
            continue
        entry["values"].add(score)

    baseline_value=defaultdict(dict)
    later_values=defaultdict(lambda:defaultdict(dict))
    conflicts=Counter()
    for (pid,visit_class,m,state),entry in raw.items():
        group,anchor=people[pid]
        if entry["invalid"] or len(entry["values"])!=1 or len(entry["events"])>1:
            conflicts[(group,state,visit_class)]+=1
            continue
        value=next(iter(entry["values"]))
        if visit_class=="baseline":baseline_value[pid][state]=value
        else:later_values[pid][state][m]=value

    selected_pairs=defaultdict(list)
    selected_gap=defaultdict(list)
    readiness=Counter()
    union_subjects=defaultdict(set)
    for pid,(group,anchor) in people.items():
        for state in STATES:
            if state not in baseline_value[pid]:continue
            readiness[(group,state,"baseline_ready")]+=1
            if later_values[pid].get(state):
                readiness[(group,state,"later_ready")]+=1
            for horizon,tol in WINDOWS:
                status,selected=target_for_window(
                    later_values[pid].get(state,{}),anchor,horizon,tol)
                readiness[(group,state,horizon,status)]+=1
                if status=="paired":
                    target_m,target_v=selected
                    selected_pairs[(group,state,horizon)].append(
                        target_v-baseline_value[pid][state])
                    selected_gap[(group,state,horizon)].append(target_m-anchor)
                    union_subjects[(group,horizon)].add(pid)

    rows=[];changes=[];quality=[]
    anchored=Counter(group for group,a in people.values())
    for group in GROUPS:
        for state in STATES:
            quality.extend([
                {"baseline_group":group,"medication_state":state,
                 "check":label,"patients_or_keys":safe(count,min_cell)}
                for label,count in (
                    ("baseline_ready_patients",readiness[(group,state,"baseline_ready")]),
                    ("later_ready_patients",readiness[(group,state,"later_ready")]),
                    ("conflicting_baseline_patient_month_keys",conflicts[(group,state,"baseline")]),
                    ("conflicting_followup_patient_month_keys",conflicts[(group,state,"followup")]),
                )])
            for horizon,tolerance in WINDOWS:
                key=(group,state,horizon)
                count=readiness[(*key,"paired")]
                rows.append({
                    "baseline_group":group,"medication_state":state,
                    "target_month":horizon,"tolerance_months":tolerance,
                    "anchor_patients":safe(anchored[group],min_cell),
                    "baseline_motor_ready":safe(readiness[(group,state,"baseline_ready")],min_cell),
                    "unique_strict_pairs":safe(count,min_cell),
                    "equidistant_target_months":safe(readiness[(*key,"equidistant_months")],min_cell),
                    "no_in_window_target":safe(readiness[(*key,"no_in_window_target")],min_cell),
                })
                values=selected_pairs[key]
                if len(values)>=min_cell:
                    gaps=selected_gap[key]
                    changes.append({
                        "baseline_group":group,"medication_state":state,
                        "target_month":horizon,"n_pairs":len(values),
                        "mean_score_change":round(mean(values),2),
                        "median_score_change":round(median(values),2),
                        "q25_score_change":round(quartile(values,.25),2),
                        "q75_score_change":round(quartile(values,.75),2),
                        "median_gap_months":round(median(gaps),2),
                        "worsening_percent":round(100*sum(x>0 for x in values)/len(values),1),
                        "unchanged_percent":round(100*sum(x==0 for x in values)/len(values),1),
                    })
    out.mkdir(parents=True,exist_ok=True)
    save(out/"strict_pair_window_counts.csv",
         ["baseline_group","medication_state","target_month","tolerance_months",
          "anchor_patients","baseline_motor_ready","unique_strict_pairs",
          "equidistant_target_months","no_in_window_target"],rows)
    save(out/"motor_state_quality.csv",
         ["baseline_group","medication_state","check","patients_or_keys"],quality)
    save(out/"aggregate_motor_change.csv",
         ["baseline_group","medication_state","target_month","n_pairs",
          "mean_score_change","median_score_change","q25_score_change",
          "q75_score_change","median_gap_months","worsening_percent",
          "unchanged_percent"],changes)
    save(out/"baseline_group_counts.csv",
         ["baseline_group","baseline_concordant","strict_anchor_patients",
          "any_strict_pair_6mo","any_strict_pair_12mo","any_strict_pair_24mo"],[
         {"baseline_group":group,
          "baseline_concordant":safe(group_totals[group],min_cell),
          "strict_anchor_patients":safe(anchored[group],min_cell),
          **{"any_strict_pair_%dmo"%h: safe(len(union_subjects[(group,h)]),min_cell)
             for h,tol in WINDOWS}}
         for group in GROUPS])
    save(out/"quality_flags.csv",["check","rows_or_keys"],[
         {"check":name,"rows_or_keys":safe(count,min_cell)}
         for name,count in sorted(issues.items())])
    report={
        "status":"strict_motor_month_pair_feasibility_not_training_ready",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "windows":[{"horizon_months":h,"tolerance_months":tol} for h,tol in WINDOWS],
        "rules":[
            "Baseline motor requires EVENT_ID=BL and exam month matching BL Visit Start.",
            "Followup motor requires a distinct later event and exam month > BL month.",
            "Only exact single numeric score per patient/month/medication state/event accepted.",
            "If two candidate followup months are equidistant to target, reject tie.",
            "Do not combine ON, OFF, or unspecified medication states.",
        ],
        "limitations":[
            "UNSPECIFIED is missing medication metadata, not proof that treatment is unchanged.",
            "No future diagnosis or future motor data may enter baseline Doctor features.",
            "No patient-wise split exists yet for the Prodromal longitudinal task.",
            "Irregular visit and attrition mechanisms may bias estimated score changes.",
            "Score clinical meaning and date semantics need source-codebook validation.",
            "Only aggregate reports are exported. Keep raw data on authorized cluster.",
        ]}
    (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,
                   default=Path("outputs/phase22_strict_motor_pairs"))
    p.add_argument("--min-cell-count",type=int,default=10)
    args=p.parse_args()
    print(json.dumps(audit(args.data_root,args.output_dir,args.min_cell_count),
                     indent=2,ensure_ascii=False))
    print("Only upload aggregate Phase22 reports, never patient source files.")

if __name__=="__main__":
    main()
