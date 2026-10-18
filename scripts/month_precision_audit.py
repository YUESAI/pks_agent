#!/usr/bin/env python3
"""Phase 8: month-precision chronology/ON-OFF audit from local database.

Only aggregate counts written. IDs and individual dates remain on cluster.
Do not treat BL visit start as verified diagnostic decision time.
"""
import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

def find_start(root):
    paths=list((root/"PPMI").glob("Baseline_Visit_Start_*.csv"))
    pat=re.compile(r"Baseline_Visit_Start_\d{2}[A-Za-z]{3}\d{4}\.csv$")
    paths=[p for p in paths if pat.fullmatch(p.name)]
    if len(paths)!=1:raise ValueError("Expected one PPMI Baseline Visit Start CSV")
    return paths[0]

def as_month(raw):
    s=(raw or "").strip()
    found=re.fullmatch(r"(\d{2})/(\d{4})",s)
    if found:
        month,year=map(int,found.groups())
        if 1<=month<=12 and 1900<=year<=2100:
            return f"{year:04d}-{month:02d}"
    return None

def start_months(root):
    anchors=defaultdict(set)
    with find_start(root).open(newline="",encoding="utf-8-sig",errors="replace") as f:
        for row in csv.DictReader(f):
            if (row.get("EVENT_ID") or "").strip()!="BL":continue
            pid=(row.get("PATNO") or "").strip()
            if pid:anchors[pid].add(as_month(row.get("INFODT")))
    result={}
    stats=Counter()
    for pid,raw in anchors.items():
        months={m for m in raw if m is not None}
        if len(months)==1:
            result[pid]=next(iter(months))
            stats["resolved_BL_month_anchor"]+=1
            if None in raw:stats["partly_undated_repeated_anchor"]+=1
        elif len(months)>1:
            stats["conflicting_BL_months"]+=1
        else:
            stats["missing_or_invalid_BL_month"]+=1
    return result,stats

def conn(path,engine):
    if engine=="duckdb":
        import duckdb
        return duckdb.connect(str(path),read_only=True)
    return sqlite3.connect("file:"+str(path.resolve())+"?mode=ro",uri=True)

def export(path,fields,rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def audit(root,dbpath,engine,out,minimum):
    out.mkdir(parents=True,exist_ok=True)
    anchors,anchor_stats=start_months(root)
    db=conn(dbpath,engine)
    chronology=Counter()
    formats=Counter()
    month_states=Counter()
    med_states=Counter()
    repeat_counts=Counter()
    patient_groups=defaultdict(list)
    try:
        # New schema needs both exam_month and date_precision. Refuse legacy DB.
        cols=[x[1] for x in db.execute("PRAGMA table_info('observations')").fetchall()]
        required={"exam_month","date_precision","date_source","medication_state"}
        if not required.issubset(cols):
            raise ValueError("Schema v2 required; build a NEW DB with build_patient_db.py")
        cursor=db.execute("""SELECT cohort,patient_id,event_id,concept,value_raw,exam_month,
                                  date_precision,date_source,medication_state,quality_status
                           FROM observations WHERE event_id IN ('BL','SC')""")
        while True:
            rows=cursor.fetchmany(5000)
            if not rows:break
            for cohort,pid,event,concept,value,month,precision,source,med,quality in rows:
                formats[(cohort,concept,precision or "unknown",source or "unknown")]+=1
                if cohort=="PPMI":
                    anchor=anchors.get(pid)
                    if not anchor:kind="no_BL_month_anchor"
                    elif not month:kind="unknown_exam_month"
                    elif month<anchor:kind="earlier_month"
                    elif month==anchor:kind="same_month_order_unknown"
                    else:kind="later_month"
                else:kind="no_verified_anchor"
                chronology[(cohort,concept,event,kind)]+=1
                if "updrs" in concept:
                    med_states[(cohort,event,concept,med or "EMPTY")]+=1
                    patient_groups[(cohort,pid,event,concept)].append((med or "",value or "",month or "",quality or ""))
    finally:
        db.close()
    for (cohort,pid,event,concept),rows in patient_groups.items():
        if len(rows)<2:continue
        label=(cohort,event,concept)
        repeat_counts[(*label,"repeated_patient_visit_concepts")]+=1
        meds={r[0] for r in rows if r[0]}
        values={r[1] for r in rows if r[1]}
        months={r[2] for r in rows if r[2]}
        if len(meds)>1:repeat_counts[(*label,"different_medication_states")]+=1
        if len(values)>1:repeat_counts[(*label,"different_scores")]+=1
        if len(months)>1:repeat_counts[(*label,"different_exam_months")]+=1
        if len(meds)>1 and len(values)>1:
            repeat_counts[(*label,"score_and_medication_both_different")]+=1
    # Do not export small groups or rare medication codes.
    export(out/"date_precision_counts.csv",["cohort","concept","precision","date_source","rows"],
           [dict(cohort=c,concept=k,precision=p,date_source=s,rows=n)
            for (c,k,p,s),n in sorted(formats.items()) if n>=minimum])
    export(out/"month_chronology.csv",["cohort","concept","visit","temporal_relation","rows"],
           [dict(cohort=c,concept=k,visit=v,temporal_relation=t,rows=n)
            for (c,k,v,t),n in sorted(chronology.items()) if n>=minimum])
    export(out/"motor_medication_frequency.csv",["cohort","visit","concept","raw_medication_state","rows"],
           [dict(cohort=c,visit=v,concept=k,raw_medication_state=m,rows=n)
            for (c,v,k,m),n in sorted(med_states.items()) if n>=minimum])
    export(out/"motor_repeat_summary.csv",["cohort","visit","concept","finding","patient_visit_concepts"],
           [dict(cohort=c,visit=v,concept=k,finding=f,patient_visit_concepts=n)
            for (c,v,k,f),n in sorted(repeat_counts.items()) if n>=minimum])
    summary={
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "database_schema":"v2_month_precision",
        "BL_anchor_statistics":dict(anchor_stats),
        "n_unique_BL_month_anchors":len(anchors),
        "temporal_semantics":"Month-level; same-month order unknown",
        "interpretation":[
            "MM/YYYY is treated as candidate month-year pending codebook verification.",
            "Earlier month means earlier than baseline visit START, not proven before diagnosis.",
            "Same month DOES NOT establish which event occurred first.",
            "No patient IDs or individual clinical scores or dates in outputs.",
            "Medication raw codes must be decoded before selecting ON/OFF exams.",
        ],
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--db-path",type=Path,required=True)
    p.add_argument("--engine",choices=("duckdb","sqlite"),default="duckdb")
    p.add_argument("--output-dir",type=Path,default=Path("outputs/month_audit"))
    p.add_argument("--min-cell-count",type=int,default=10)
    a=p.parse_args()
    root,db,out=(a.data_root.resolve(),a.db_path.resolve(),a.output_dir.resolve())
    if not root.is_dir() or not db.is_file() or out==root or root in out.parents or a.min_cell_count<10:
        p.error("Invalid root, db, output or minimum count")
    print(json.dumps(audit(root,db,a.engine,out,a.min_cell_count),indent=2,ensure_ascii=False))
    print("Upload only aggregated report files, not the database.")
if __name__=="__main__":main()
