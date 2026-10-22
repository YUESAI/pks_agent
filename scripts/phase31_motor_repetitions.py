#!/usr/bin/env python3
"""Phase31: read-only TRAIN-only duplicate Motor measurement mechanism audit.

Revisit PPMI raw MDS-UPDRS Part III rows that Phase23 excluded as repeated
patient-month keys. Identify whether they are exact repeats, distinct visits
in the same month, or genuinely discordant numeric scores.

This is a RECORD AUDIT, not an estimate of test-retest measurement error:
distinct EVENT_IDs may be distinct clinical exams, and dates are MM/YYYY.
Never reconcile collisions or overwrite the frozen Phase23 manifest here.
Only aggregated reports are exported; no patient IDs or scores leave cluster.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
from statistics import median

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase22_strict_motor_pairs import (
    one_file,records,parse_month,state_name,present,safe,save
)

PREFIX="MDS-UPDRS_Part_III_"
CONTEXTS=("baseline","6m_window","12m_window","24m_window","other_followup")
CATEGORIES=(
    "single_valid_row",
    "exact_repeat_same_event",
    "different_events_same_score",
    "different_events_different_scores",
    "same_event_different_scores",
    "invalid_value_in_key",
)
RANGES=("0","1","2","3_to_5","6_to_10","11plus")


def train_anchors(manifest):
    conn=sqlite3.connect("file:"+str(manifest.resolve())+"?mode=ro",uri=True)
    try:
        rows=conn.execute(
            "SELECT pid,baseline_month FROM patients WHERE split='train'"
        ).fetchall()
    finally:
        conn.close()
    if not rows or any(not pid for pid,month in rows):
        raise ValueError("Missing frozen PPMI TRAIN registry")
    if len(set(pid for pid,month in rows))!=len(rows):
        raise ValueError("Duplicate patient identifiers in TRAIN registry")
    return {pid:month for pid,month in rows}


def context_for(event,exam_month,anchor):
    """Respect EVENT_ID=BL; never reclassify late BL as follow-up."""
    if event=="BL":
        return "baseline" if exam_month==anchor else "bad_BL_month"
    if event in ("","SC","CONSENT") or exam_month<=anchor:
        return "excluded_event_or_time"
    gap=exam_month-anchor
    if 4<=gap<=8:return "6m_window"
    if 9<=gap<=15:return "12m_window"
    if 20<=gap<=28:return "24m_window"
    return "other_followup"


def score_spread_bin(spread):
    if spread==0:return "0"
    if spread<=1:return "1"
    if spread<=2:return "2"
    if spread<=5:return "3_to_5"
    if spread<=10:return "6_to_10"
    return "11plus"


def categorize(entry):
    """Exclusive mechanism categories ordered by data-quality severity."""
    if entry["invalid"]:
        return "invalid_value_in_key"
    event_values=entry["event_values"]
    if any(len(vals)>1 for vals in event_values.values()):
        return "same_event_different_scores"
    distinct={score for vals in event_values.values() for score in vals}
    if len(event_values)>1:
        return ("different_events_different_scores"
                if len(distinct)>1 else "different_events_same_score")
    if entry["rows"]>1:
        return "exact_repeat_same_event"
    return "single_valid_row"


def summarize(root,manifest,out,minimum=10):
    root,manifest,out=(Path(p).expanduser().resolve()
                       for p in (root,manifest,out))
    if not (root/"PPMI").is_dir() or not manifest.is_file():
        raise FileNotFoundError("Missing original PPMI folder or private manifest")
    if root==out or root in out.parents or out==manifest or minimum<10:
        raise ValueError("Unsafe output or small-cell threshold")
    anchors=train_anchors(manifest)
    source=one_file(root/"PPMI",PREFIX)
    quality=Counter()
    # Never store or export non-TRAIN rows. Patient keys remain in local RAM.
    grouped=defaultdict(lambda:{
        "rows":0,"event_values":defaultdict(set),
        "invalid":False,"scores":set(),
    })
    eligible_rows=Counter()
    observed_patients=defaultdict(set)
    for record in records(source,("PATNO","EVENT_ID","NP3TOT"),quality):
        pid=record.get("PATNO","")
        if pid not in anchors:
            continue
        anchor=anchors[pid]
        if anchor is None:
            quality["TRAIN_motor_rows_without_BL_anchor"]+=1
            continue
        event=record.get("EVENT_ID","")
        exam_month=parse_month(record.get("EXAMDT") or record.get("INFODT"))
        if exam_month is None:
            quality["TRAIN_motor_rows_without_exam_month"]+=1
            continue
        category=context_for(event,exam_month,anchor)
        if category not in CONTEXTS:
            quality["TRAIN_motor_rows_"+category]+=1
            continue
        state=state_name(record.get("PDSTATE"))
        if state!="UNSPECIFIED":
            quality["TRAIN_rows_other_medication_state"]+=1
            continue
        key=(pid,category,exam_month)
        item=grouped[key]
        item["rows"]+=1
        eligible_rows[category]+=1
        observed_patients[category].add(pid)
        try:
            value=float(record.get("NP3TOT",""))
            if not math.isfinite(value) or not 0<=value<=132:
                raise ValueError("Not an admissible motor score")
        except (TypeError,ValueError):
            item["invalid"]=True
            continue
        item["event_values"][event].add(value)
        item["scores"].add(value)

    key_counts=Counter()
    patient_affected=defaultdict(set)
    score_disagreement=Counter()
    kinds_with_uncertainty=("same_event_different_scores",
                            "different_events_different_scores")
    for (pid,context,month),item in grouped.items():
        category=categorize(item)
        key_counts[(context,category)]+=1
        patient_affected[(context,category)].add(pid)
        if category in kinds_with_uncertainty:
            spread=max(item["scores"])-min(item["scores"])
            score_disagreement[(context,category,score_spread_bin(spread))]+=1
    rows=[]
    affected=[]
    for context in CONTEXTS:
        for category in CATEGORIES:
            n=key_counts[(context,category)]
            rows.append({
                "visit_context":context,
                "mechanism":category,
                "patient_month_keys":safe(n,minimum),
            })
            n_people=len(patient_affected[(context,category)])
            affected.append({
                "visit_context":context,
                "mechanism":category,
                "train_patients_affected":safe(n_people,minimum),
            })
    spreads=[]
    for context in CONTEXTS:
        for category in kinds_with_uncertainty:
            for spread_bin in RANGES:
                n=score_disagreement[(context,category,spread_bin)]
                spreads.append({
                    "visit_context":context,
                    "mechanism":category,
                    "absolute_score_spread_bin":spread_bin,
                    "patient_month_keys":safe(n,minimum),
                })
    overview=[]
    for context in CONTEXTS:
        n_keys=sum(key_counts[(context,kind)] for kind in CATEGORIES)
        excluded=sum(key_counts[(context,kind)] for kind in (
            "different_events_same_score",
            "different_events_different_scores",
            "same_event_different_scores","invalid_value_in_key"))
        overview.append({
            "visit_context":context,
            "train_patients_with_rows":safe(
                len(observed_patients[context]),minimum),
            "eligible_source_rows":safe(eligible_rows[context],minimum),
            "patient_month_keys":safe(n_keys,minimum),
            "conservatively_rejected_keys":safe(excluded,minimum),
            "different_events_same_score_keys":safe(
                key_counts[(context,"different_events_same_score")],minimum),
        })

    out.mkdir(parents=True,exist_ok=True)
    save(out/"collision_mechanisms.csv",
         ["visit_context","mechanism","patient_month_keys"],rows)
    save(out/"collision_affected_patients.csv",
         ["visit_context","mechanism","train_patients_affected"],affected)
    save(out/"score_disagreement_bins.csv",
         ["visit_context","mechanism","absolute_score_spread_bin",
          "patient_month_keys"],spreads)
    save(out/"window_collision_overview.csv",
         ["visit_context","train_patients_with_rows","eligible_source_rows",
          "patient_month_keys","conservatively_rejected_keys",
          "different_events_same_score_keys"],overview)
    save(out/"source_quality_flags.csv",["issue","TRAIN_rows"],[
        {"issue":flag,"TRAIN_rows":safe(count,minimum)}
        for flag,count in sorted(quality.items())])
    summary={
        "status":"TRAIN_ONLY_raw_motor_repetition_audit_no_reconciliation",
        "frozen_train_population":len(anchors),
        "source_file":source.name,
        "medication_code_filter":"UNSPECIFIED_ONLY",
        "key_definition":"patient ID / baseline-vs-followup window / exam calendar month",
        "source_columns":["PATNO","EVENT_ID","NP3TOT","EXAMDT_or_INFODT","PDSTATE"],
        "warnings":[
            "Different EVENT_IDs may mean genuinely different examinations in one month.",
            "Monthly date precision prevents matching specific examination days.",
            "Score disagreement does NOT estimate clinical measurement noise.",
            "UNSPECIFIED medication code does not establish identical treatment state.",
            "No cohort or target scores were modified and no patient data exported.",
            "Only frozen TRAIN patient records examined; val/test untouched.",
            "Conservative rejected keys must NOT be recovered automatically.",
            "Counts below privacy threshold are suppressed independently.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--manifest-db",type=Path,required=True)
    parser.add_argument("--report-dir",type=Path,
                        default=Path("outputs/phase31_motor_repetitions"))
    parser.add_argument("--min-cell-count",type=int,default=10)
    options=parser.parse_args()
    info=summarize(options.data_root,options.manifest_db,
                   options.report_dir,options.min_cell_count)
    print(json.dumps(info,indent=2,ensure_ascii=False))
    print("Share aggregate Phase31 reports only; NEVER upload private manifest.")


if __name__=="__main__":
    main()
