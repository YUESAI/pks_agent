#!/usr/bin/env python3
"""Aggregate same-patient baseline feature coverage; no patient-level exports.

python scripts/episode_audit.py --data-root /scratch/ylong66/data/pks/data \
  --output-dir ./outputs/episode_audit
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

SPEC = {
 "PPMI": {
   "anchor":("Participant_Status_","COHORT"),
   "diagnosis":("Primary_Research_Diagnosis_","PRIMDIAG"),
   "age":("Age_at_visit_","AGE_AT_VISIT"),
   "motor":("MDS-UPDRS_Part_III_","NP3TOT"),
   "moca":("Montreal_Cognitive_Assessment__MoCA__","MCATOT"),
   "upsit":("University_of_Pennsylvania_Smell_Identification_Test_UPSIT_","TOTAL_CORRECT"),
   "rbd":("REM_Sleep_Behavior_Disorder_Screening_Questionnaire_","DRMVIVID"),
   "scopa":("SCOPA-AUT_","SCAU1")
 },
 "BioFIND": {
   "anchor":("Screening_Demographics_","APPRDX"),
   "diagnosis":("Primary_Diagnosis_","PRIMDIAG"),
   "motor_post_dose":("MDS_UPDRS_Part_III__Post_Dose__","NP3GAIT"),
   "moca":("Montreal_Cognitive_Assessment__MoCA__","MCATOT"),
   "rbd":("REM_Sleep_Disorder_Questionnaire_","DRMVIVID")
 }
}
COMBOS = {
 "PPMI":[("motor","moca"),("motor","moca","rbd"),("motor","moca","rbd","scopa"),
         ("motor","moca","rbd","upsit"),("motor","moca","rbd","scopa","upsit")],
 "BioFIND":[("motor_post_dose","moca"),("motor_post_dose","moca","rbd")]
}
MISSING = {"","na","nan","n/a","null","none","missing","."}
CODE_FIELDS = {"COHORT","COHORT_DEFINITION","PRIMDIAG","APPRDX","PDSTATE","IMPUTED_DATA"}

def clean(value):
    return str(value).strip() if value is not None else ""

def present(value):
    return clean(value).lower() not in MISSING

def match_file(folder, prefix):
    pat = re.compile(re.escape(prefix) + r"\d{2}[A-Za-z]{3}\d{4}\.csv$")
    matches = [p for p in folder.glob(prefix+"*.csv") if pat.fullmatch(p.name)]
    if len(matches) != 1:
        raise ValueError("Expected exactly one versioned CSV: "+prefix)
    return matches[0]

def as_date(value):
    s = clean(value)
    for fmt in ("%Y-%m-%d","%m/%d/%Y","%d-%b-%Y","%Y/%m/%d",
                "%Y-%m-%d %H:%M:%S","%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s,fmt).date()
        except ValueError:
            continue
    return None

def read_records(path, field):
    """Read selected visits and reconcile duplicate PATNO/EVENT_ID headers.

    Rows with two conflicting populated keys are excluded rather than joined.
    """
    data=defaultdict(lambda:defaultdict(list))
    with path.open(encoding="utf-8-sig",errors="replace",newline="") as handle:
        reader=csv.reader(handle)
        header=[x.strip() for x in next(reader,[])]
        locations=defaultdict(list)
        for idx,name in enumerate(header):
            locations[name].append(idx)
        if "PATNO" not in locations or field not in locations:
            raise ValueError("Missing patient key or measurement field")
        def field_value(row,name):
            items=[row[i].strip() for i in locations.get(name,[]) if i<len(row)]
            nonblank=[item for item in items if present(item)]
            if len(set(nonblank))>1:
                return None
            return nonblank[0] if nonblank else ""
        for row in reader:
            pid=field_value(row,"PATNO")
            event=field_value(row,"EVENT_ID")
            measurement=field_value(row,field)
            if not present(pid) or event is None or measurement is None:
                continue
            if event not in ("","CONSENT","SC","BL"):
                continue
            data[pid][event].append({
                "value":measurement,
                "date":as_date(field_value(row,"INFODT")),
                "state":field_value(row,"PDSTATE") or ""
            })
    return data

def label_and_date(diagnoses):
    labels, dates, conflicting_label, conflicting_date = {}, {}, 0, 0
    for pid, events in diagnoses.items():
        rows=events.get("BL",[])
        vals={r["value"] for r in rows if present(r["value"])}
        if len(vals)==1:
            labels[pid]=next(iter(vals))
        elif len(vals)>1:
            conflicting_label+=1
        values={r["date"] for r in rows if r["date"] is not None}
        if len(values)==1:
            dates[pid]=next(iter(values))
        elif len(values)>1:
            conflicting_date+=1
    return labels, dates, conflicting_label, conflicting_date

def feature_views(records, index_dates, window_days):
    bl, candidate, dated = set(), set(), set()
    duplicates=Counter()
    for pid, events in records.items():
        for visit in ("BL","SC"):
            rows=events.get(visit,[])
            measured=[r for r in rows if present(r["value"])]
            if len(rows)>1:
                duplicates[visit+"_repeated_subject_visit"]+=1
                duplicates[visit+"_extra_rows"]+=len(rows)-1
                if len({r["value"] for r in measured})>1:
                    duplicates[visit+"_different_measurements"]+=1
                if len({r["state"] for r in rows if present(r["state"])})>1:
                    duplicates[visit+"_multiple_medication_states"]+=1
            if not measured:
                continue
            if visit=="BL":
                bl.add(pid)
                candidate.add(pid)
                dated.add(pid)
            else:
                candidate.add(pid)
                index=index_dates.get(pid)
                if index is None:
                    duplicates["SC_without_BL_diagnosis_date"]+=1
                elif any(r["date"] is not None
                         and timedelta(0)<=index-r["date"]<=timedelta(days=window_days)
                         for r in measured):
                    dated.add(pid)
                else:
                    duplicates["SC_outside_window_or_undated"]+=1
    return {"strict_BL":bl,"SC_BL_candidate":candidate,
            "SC_BL_date_checked":dated},duplicates

def write_csv(path,fields,records):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(records)

def get_codes(root):
    result=[]
    for cohort,prefix in (("PPMI","Code_List_-_Harmonized_"),("BioFIND","Code_List_")):
        try:
            file=match_file(root/cohort,prefix)
            with file.open(encoding="utf-8-sig",errors="replace",newline="") as f:
                for row in csv.DictReader(f):
                    field=clean(row.get("ITM_NAME")).upper()
                    if field in CODE_FIELDS:
                        result.append({"cohort":cohort,"field":field,
                          "module_or_view":clean(row.get("MOD_NAME") or row.get("VIEWNAME")),
                          "raw_code":clean(row.get("CODE")),"official_decode":clean(row.get("DECODE"))})
        except (OSError,ValueError,csv.Error):
            continue
    return result

def analyze(root,out,cohorts,window_days,min_cell):
    out.mkdir(parents=True,exist_ok=True)
    overview,features,combos,strata,duplicate_rows,label_counts,crosstab,issues=([],[],[],[],[],[],[],[])
    for cohort in cohorts:
        data={}
        for name,(prefix,col) in SPEC[cohort].items():
            try:
                data[name]=read_records(match_file(root/cohort,prefix),col)
            except (OSError,ValueError,csv.Error,UnicodeError) as error:
                issues.append({"cohort":cohort,"table":name,"error":type(error).__name__})
        if "anchor" not in data or "diagnosis" not in data:
            continue
        labels,idx_dates,bad_labels,bad_dates=label_and_date(data["diagnosis"])
        anchor={}
        for pid,events in data["anchor"].items():
            codes={r["value"] for rows in events.values() for r in rows if present(r["value"])}
            if len(codes)==1:
                anchor[pid]=next(iter(codes))
        eligible=set(labels)&set(anchor)
        overview.append({"cohort":cohort,"anchor_patients":len(anchor),
            "BL_diagnosis_patients":len(labels),
            "anchor_intersect_BL_diagnosis":len(eligible),
            "conflicting_BL_labels":bad_labels,
            "BL_diagnosis_date_available":len(set(idx_dates)&eligible),
            "conflicting_BL_diagnosis_dates":bad_dates})
        for code,n in sorted(Counter(labels[p] for p in eligible).items()):
            if n>=min_cell:
                label_counts.append({"cohort":cohort,"raw_diagnosis_code":code,"patients":n})
        for (enroll,diag),n in sorted(Counter((anchor[p],labels[p]) for p in eligible).items()):
            if n>=min_cell:
                crosstab.append({"cohort":cohort,"anchor_code":enroll,
                                 "BL_diagnosis_code":diag,"patients":n})
        views={}
        for name,records in data.items():
            if name in ("anchor","diagnosis"):
                continue
            possible,dups=feature_views(records,idx_dates,window_days)
            views[name]=possible
            for metric,n in dups.items():
                if n:
                    duplicate_rows.append({"cohort":cohort,"table":name,
                                           "metric":metric,"count":n if n>=min_cell else f"<{min_cell}"})
            for key,members in possible.items():
                n=len(members&eligible)
                features.append({"cohort":cohort,"view":key,"feature":name,
                                 "BL_labeled_patients":n,
                                 "pct_labeled":round(100*n/len(eligible),2) if eligible else ""})
        for combo in COMBOS[cohort]:
            if not all(name in views for name in combo):
                continue
            for view in ("strict_BL","SC_BL_candidate","SC_BL_date_checked"):
                members=set(eligible)
                for name in combo:
                    members &= views[name][view]
                combos.append({"cohort":cohort,"view":view,"combination":"+".join(combo),
                               "BL_labeled_patients":len(members),
                               "pct_labeled":round(100*len(members)/len(eligible),2) if eligible else ""})
                for code,n in sorted(Counter(labels[p] for p in members).items()):
                    if n>=min_cell:
                        strata.append({"cohort":cohort,"view":view,"combination":"+".join(combo),
                                       "diagnosis_raw_code":code,"patients":n})
    write_csv(out/"episode_overview.csv",["cohort","anchor_patients","BL_diagnosis_patients",
              "anchor_intersect_BL_diagnosis","conflicting_BL_labels",
              "BL_diagnosis_date_available","conflicting_BL_diagnosis_dates"],overview)
    write_csv(out/"feature_by_view.csv",["cohort","view","feature","BL_labeled_patients",
              "pct_labeled"],features)
    write_csv(out/"joint_combinations.csv",["cohort","view","combination",
              "BL_labeled_patients","pct_labeled"],combos)
    write_csv(out/"joint_by_raw_diagnosis.csv",["cohort","view","combination",
              "diagnosis_raw_code","patients"],strata)
    write_csv(out/"duplicate_conditions.csv",["cohort","table","metric","count"],duplicate_rows)
    write_csv(out/"BL_diagnosis_codes.csv",["cohort","raw_diagnosis_code","patients"],label_counts)
    write_csv(out/"cohort_vs_diagnosis.csv",["cohort","anchor_code","BL_diagnosis_code",
              "patients"],crosstab)
    write_csv(out/"official_codebook_subset.csv",["cohort","field","module_or_view",
              "raw_code","official_decode"],get_codes(root))
    write_csv(out/"issues.csv",["cohort","table","error"],issues)
    summary={"generated_utc":datetime.now(timezone.utc).isoformat(),
             "data_root":str(root),"screening_window_days":window_days,"overview":overview,
             "tables_failed":len(issues),
             "warnings":[
               "Raw codes are not yet mapped to clinically verified PD/HC labels.",
               "SC_BL_candidate is an undated availability upper bound; NOT for training.",
               "SC_BL_date_checked verifies SC before BL diagnosis INFODT only, not all BL test timestamps.",
               "Blank-string coverage may include missing sentinel codes or imputed UPSIT.",
               "Repeated exams and medication states have not been reconciled.",
               "No patient IDs or measurement values are written to outputs."]}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    lines=["# Same-patient diagnostic episode audit","",
           "Unvalidated exploratory aggregates. Do NOT train on SC_BL_candidate.","",
           "| Cohort | Anchor subjects | Unique BL diagnoses | Intersection |",
           "|---|---:|---:|---:|"]
    for row in overview:
        lines.append("| {cohort} | {anchor_patients} | {BL_diagnosis_patients} | {anchor_intersect_BL_diagnosis} |".format(**row))
    lines += ["","| Cohort | View | Tests | Patients |",
              "|---|---|---|---:|"]
    for r in combos:
        lines.append(f"| {r['cohort']} | {r['view']} | {r['combination']} | {r['BL_labeled_patients']} |")
    lines += ["","SC_BL_candidate is not time checked; SC_BL_date_checked uses SC measurement INFODT",
              f"within {window_days} days before BL diagnosis INFODT (still NOT leakage-free).",
              "Decode cohort/diagnosis codes from official_codebook_subset.csv before PD vs HC labeling.",
              "Review medication state, missing sentinel values and UPSIT imputation.",""]
    (out/"report.md").write_text("\n".join(lines),encoding="utf-8")
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,default=Path("./outputs/episode_audit"))
    p.add_argument("--cohorts",nargs="+",choices=list(SPEC),default=list(SPEC))
    p.add_argument("--screening-window-days",type=int,default=180)
    p.add_argument("--min-cell-count",type=int,default=10)
    a=p.parse_args()
    root,out=a.data_root.expanduser().resolve(),a.output_dir.expanduser().resolve()
    if not root.is_dir() or out==root or root in out.parents or a.screening_window_days<1 or a.min_cell_count<10:
        p.error("Invalid data root, output location, time window or min-cell-count")
    print(json.dumps(analyze(root,out,a.cohorts,a.screening_window_days,a.min_cell_count),
                     ensure_ascii=False,indent=2))
    print("Reports:",out)

if __name__=="__main__":
    main()
