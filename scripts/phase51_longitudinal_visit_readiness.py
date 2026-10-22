#!/usr/bin/env python3
"""Phase51: frozen TRAIN longitudinal visit-count feasibility audit.

Read-only PPMI MDS-UPDRS Part III motor totals, keyed by distinct
patient/month/medication-state. Do NOT turn non-BL events into baseline.
Conservatively reject ambiguously duplicated patient-month exams,
non-numeric scores, unknown state codes, and inconsistent BL month.
Only small-cell-suppressed AGGREGATES are written (no individual data).

This is an audit of *observed usable visits*, not a validated temporal WM.
"""
from __future__ import annotations
import argparse,csv,json,math,sqlite3,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
from statistics import median
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase22_strict_motor_pairs import (
    one_file,parse_month,state_name,present
)

def anchors_train(manifest):
    con=sqlite3.connect("file:"+str(Path(manifest).resolve())+"?mode=ro",uri=True)
    try:
        rows=con.execute(
            "SELECT pid,baseline_month FROM patients WHERE split='train'").fetchall()
    finally:con.close()
    if not rows or len(set(pid for pid,_ in rows))!=len(rows):
        raise ValueError("Missing or duplicated frozen TRAIN registry")
    return dict(rows)

def band_visits(n):
    if n==0:return "0"
    if n==1:return "1"
    if n==2:return "2"
    if n==3:return "3"
    if n==4:return "4"
    return "5plus"

def suppress(n,minimum=10):
    return str(n) if n==0 or n>=minimum else "<"+str(minimum)

def audit_records(rows,anchors,minimum=10):
    """Rows iterable may be synthesized; patient identifiers never exported."""
    if minimum<10:raise ValueError("Minimum suppressed cell size is 10")
    quality=Counter()
    buckets=defaultdict(lambda:{"events":set(),"values":set(),"invalid":False,
                                 "rows":0})
    for row in rows:
        pid=str(row.get("PATNO","")).strip()
        if pid not in anchors:continue
        anchor=anchors[pid]
        if anchor is None:
            quality["no_anchored_baseline_month"]+=1
            continue
        event=str(row.get("EVENT_ID","")).strip()
        if not event or event in ("SC","CONSENT"):
            quality["excluded_nonclinical_event"]+=1;continue
        month=parse_month(row.get("EXAMDT") or row.get("INFODT"))
        if month is None:
            quality["missing_or_invalid_exam_month"]+=1;continue
        if event=="BL":
            if month!=anchor:
                quality["BL_event_anchor_mismatch"]+=1;continue
        elif month<=anchor:
            quality["non_BL_event_before_or_at_anchor"]+=1;continue
        state=state_name(row.get("PDSTATE",""))
        if state=="OTHER":
            quality["unknown_medication_state"]+=1;continue
        key=(pid,month,state)
        b=buckets[key]
        b["events"].add(event)
        b["rows"]+=1
        raw=str(row.get("NP3TOT","")).strip()
        try:
            val=float(raw)
            if not math.isfinite(val) or not 0<=val<=132:
                raise ValueError("Invalid score")
        except (ValueError,TypeError):
            b["invalid"]=True
            quality["unusable_motor_score_rows"]+=1
            continue
        b["values"].add(val)
    good=defaultdict(lambda:defaultdict(dict))
    valid_keys=0
    for (pid,month,state),b in buckets.items():
        if b["invalid"] or len(b["values"])!=1 or len(b["events"])!=1:
            quality["conflicted_patient_month_state_keys"]+=1;continue
        good[pid][state][month]=next(iter(b["values"]))
        valid_keys+=1
    # Only UNSPECIFIED is used for primary feasibility to preserve
    # comparability to frozen Phase23. ON/OFF are shown as separate audits.
    # A missing drug-state code is unknown medication state, not OFF.
    freq=Counter()
    separate=Counter()
    intervals=Counter()
    transitions=Counter()
    longer=Counter()
    n_primary_anchored=0
    for pid,anchor in anchors.items():
        states=good.get(pid,{})
        for state in ("UNSPECIFIED","ON","OFF"):
            months=sorted(states.get(state,{}))
            if state=="UNSPECIFIED":
                freq[band_visits(len(months))]+=1
                if anchor in months:
                    n_primary_anchored+=1
                    longer[band_visits(len(months))]+=1
            separate[(state,band_visits(len(months)))]+=1
            for a,b in zip(months,months[1:]):
                d=b-a
                if d<=0:raise AssertionError("Nonpositive visit interval")
                if state=="UNSPECIFIED":
                    intervals[d if d<=36 else "37plus"]+=1
                    transitions["primary_consecutive_pairs"]+=1
                else:
                    transitions[state+"_consecutive_pairs"]+=1
    # Because counts of consecutive intervals can be 1-9, suppress
    # individual month-gap frequency and only share coarse ranges.
    gap_bands=Counter()
    for gap,count in intervals.items():
        cat=("1_to_3" if isinstance(gap,int) and gap<=3
             else "4_to_8" if isinstance(gap,int) and gap<=8
             else "9_to_15" if isinstance(gap,int) and gap<=15
             else "16_to_24" if isinstance(gap,int) and gap<=24
             else "25_to_36" if isinstance(gap,int) and gap<=36
             else "37plus")
        gap_bands[cat]+=count
    counts={
        "frozen_train_patients":len(anchors),
        "anchored_train_patients":sum(x is not None for x in anchors.values()),
        "usable_unique_patient_month_state_exams":valid_keys,
        "primary_state_UNSPECIFIED_with_baseline":n_primary_anchored,
        "primary_state_consecutive_visit_pairs":transitions["primary_consecutive_pairs"],
    }
    reports={
        "primary_visit_distribution.csv":[
            {"unique_exam_visits":b,
             "train_patients":suppress(freq[b],minimum)}
            for b in ("0","1","2","3","4","5plus")],
        "primary_baseline_anchored_distribution.csv":[
            {"unique_exam_visits":b,
             "train_patients":suppress(longer[b],minimum)}
            for b in ("0","1","2","3","4","5plus")],
        "medication_state_visit_distribution.csv":[
            {"medication_state":state,"unique_exam_visits":b,
             "train_patients":suppress(separate[state,b],minimum)}
            for state in ("UNSPECIFIED","ON","OFF")
            for b in ("0","1","2","3","4","5plus")],
        "primary_followup_gap_bins.csv":[
            {"consecutive_visit_month_gap":cat,
             "patient_pairs":suppress(gap_bands[cat],minimum)}
            for cat in ("1_to_3","4_to_8","9_to_15",
                        "16_to_24","25_to_36","37plus")],
        "data_quality_counts.csv":[
            {"issue":k,"observations":suppress(v,minimum)}
            for k,v in sorted(quality.items())],
    }
    return counts,reports

def source_records(path):
    """Preserve duplicate header conflict checks; raw rows never exported."""
    with path.open("r",encoding="utf-8-sig",errors="replace",newline="") as f:
        reader=csv.DictReader(f)
        names=reader.fieldnames or []
        required={"PATNO","EVENT_ID","NP3TOT"}
        if not required.issubset(set(names)) or len(names)!=len(set(names)):
            raise ValueError("Missing or duplicate motor CSV columns")
        for row in reader:yield row

def execute(data_root,manifest,out,minimum=10):
    data_root=Path(data_root).expanduser().resolve()
    manifest=Path(manifest).expanduser().resolve()
    out=Path(out).expanduser().resolve()
    if not (data_root/"PPMI").is_dir() or not manifest.is_file():
        raise FileNotFoundError("Missing data or frozen manifest")
    if (out==data_root or data_root in out.parents or out==manifest or
        out==manifest.parent or minimum<10):
        raise ValueError("Unsafe report location / suppression")
    anchors=anchors_train(manifest)
    source=one_file(data_root/"PPMI","MDS-UPDRS_Part_III_")
    counts,reports=audit_records(source_records(source),anchors,minimum)
    out.mkdir(parents=True,exist_ok=True)
    for filename,rows in reports.items():
        with (out/filename).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    summary={
        "status":"TRAIN_only_longitudinal_raw_motor_multi_visit_readiness",
        "source_file":source.name,**counts,
        "medication_state_primary":"UNSPECIFIED_ONLY",
        "minimum_reportable_cell":minimum,
        "caveats":[
            "Visit count refers to unique nonambiguous motor patient-month/state keys.",
            "A blank PDSTATE means unspecified, not untreated or OFF.",
            "Distinct months and events do not guarantee comparable tests.",
            "ON/OFF not mixed with UNSPECIFIED in the primary trajectory count.",
            "No cross-modality harmonization or treatment intervention inferred.",
            "Only frozen TRAIN registry queried. Original TEST consumed.",
            "Only suppressed aggregate reports; no patient-level data."
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--min-cell",type=int,default=10)
    a=p.parse_args()
    print(json.dumps(execute(a.data_root,a.manifest_db,a.report_dir,a.min_cell),indent=2))
    print("Share ONLY aggregate CSV/JSON; no raw data or manifest.")
if __name__=="__main__":main()
