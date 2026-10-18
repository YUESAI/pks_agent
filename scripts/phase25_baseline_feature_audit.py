#!/usr/bin/env python3
"""Phase 25: audit baseline multimodal feature readiness for prodromal Motor WM.

Uses ONLY the Phase23 TRAIN patient IDs. Reads no future numeric target values
and no validation/test clinical observations. Source clinical DB and private
longitudinal manifest are opened read-only.

Clinical inputs: BL-only Age/MoCA/RBD/SCOPA and strictly-pre-BL SC UPSIT
fallback (1..6 months), with quality, numeric range, and conflicting-repeat
filters. The Phase23 BL motor score is recorded only for aggregate coverage.
Future 12-month target readiness is used solely for *selection-bias auditing*
in aggregate reports; it is NEVER a model input.

Only aggregate files are written. Patient identifiers and scores stay local.
These rules are provisional pending verification of original PPMI codebooks.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase10_leakage_splits import open_source

FEATURES = {
    "age_at_visit": (0., 120.),
    "moca_total": (0., 30.),
    "rbd_vivid_dreams": (0., 20.),
    "scopa_aut_item1": (0., 20.),
    "upsit_total": (0., 40.),
}
PROVISIONAL_ITEM_RANGE = ("rbd_vivid_dreams", "scopa_aut_item1")
BAD_MISSING_CODES = {"-1", "-7", "-8", "-9", "-99", "999", "9999"}
FALSE_FLAGS = {"", "0", "false", "no", "n", "f", "none"}
STATUS = ("ready_BL", "ready_SC", "not_recorded", "date_rejected",
          "quality_rejected", "range_rejected", "ambiguous_repeat")
STRATA = ("has_12m_target", "missing_12m_target")


def month_number(v):
    if not v or not isinstance(v, str):
        return None
    try:
        year, month = v.split("-")
        y, m = int(year), int(month)
        if len(year) == 4 and len(month) == 2 and 1 <= m <= 12:
            return y*12+m
    except (ValueError, TypeError):
        pass
    return None


def clean_numeric(concept, value, quality, imputed):
    if quality != "observed_unverified" or str(imputed or "").strip().lower() not in FALSE_FLAGS:
        return "quality_rejected", None
    if value is None or str(value).strip() in BAD_MISSING_CODES:
        return "quality_rejected", None
    try:
        x = float(str(value).strip())
    except (TypeError, ValueError):
        return "quality_rejected", None
    lower, upper = FEATURES[concept]
    if not math.isfinite(x) or not lower <= x <= upper:
        return "range_rejected", None
    return "valid", x


def evaluate_visit(candidates, concept, anchor, event):
    """Select unique value from a strictly time-eligible observed visit."""
    if not candidates:
        return "not_recorded", None
    eligible = []
    failures = Counter()
    for value, month, quality, imputed in candidates:
        measured = month_number(month)
        if measured is None or anchor is None:
            failures["date_rejected"] += 1
            continue
        if event == "BL" and measured != anchor:
            failures["date_rejected"] += 1
            continue
        if event == "SC" and not (1 <= anchor-measured <= 6):
            failures["date_rejected"] += 1
            continue
        verdict, numeric = clean_numeric(concept,value,quality,imputed)
        if verdict != "valid":
            failures[verdict] += 1
        else:
            eligible.append(numeric)
    if eligible:
        # Conservative: do not silently average multiple genuinely different
        # exam results, even when from the same event and calendar month.
        if len(set(eligible)) > 1:
            return "ambiguous_repeat", None
        return ("ready_BL" if event == "BL" else "ready_SC"), eligible[0]
    priority = ("range_rejected", "quality_rejected", "date_rejected")
    return next((name for name in priority if failures[name]), "not_recorded"), None


def choose_feature(concept, entries, anchor):
    bl = entries.get("BL", [])
    status,value = evaluate_visit(bl, concept, anchor, "BL")
    # A BL row exists but fails QC/date: do not silently cherry-pick a
    # screening value to overwrite a questionable baseline exam.
    if concept == "upsit_total" and status == "not_recorded":
        return evaluate_visit(entries.get("SC", []), concept, anchor, "SC")
    return status,value


def load_train_people(manifest):
    with sqlite3.connect("file:"+str(manifest.resolve())+"?mode=ro",uri=True) as db:
        rows=db.execute(
            "SELECT pid,baseline_month,baseline_motor FROM patients "
            "WHERE split='train' ORDER BY pid").fetchall()
        if not rows:
            raise ValueError("No Phase23 TRAIN patient IDs")
        ids=[pid for pid,month,motor in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate training subject ID")
        out={}
        for pid,month,motor in rows:
            if not pid: raise ValueError("Empty TRAIN patient ID")
            if month is not None and not isinstance(month,int):
                month=int(month)
            out[pid] = (month,motor)
        readiness={}
        for start in range(0,len(ids),200):
            part=ids[start:start+200]
            marks=",".join("?" for _ in part)
            for pid,status in db.execute(
                "SELECT pid,status FROM readiness WHERE horizon=12 AND pid IN ("+
                marks+")",part).fetchall():
                if pid in readiness:
                    raise ValueError("Duplicate Phase23 target availability")
                readiness[pid]=status
        if set(readiness)!=set(ids):
            raise ValueError("Incomplete TRAIN readiness registry")
    return out,readiness


def source_rows(source_db,engine,train_ids):
    """Read only BL/SC selected concepts belonging to TRAIN patients."""
    source=open_source(source_db,engine)
    observations=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    try:
        fields={col[1] for col in source.execute("PRAGMA table_info('observations')").fetchall()}
        required={"cohort","patient_id","event_id","concept","value_raw",
                  "exam_month","quality_status","imputed_flag"}
        if not required.issubset(fields):
            raise ValueError("Requires Phase8 v2 observations database")
        for start in range(0,len(train_ids),200):
            part=train_ids[start:start+200]
            marks=",".join("?" for _ in part)
            query=(
                "SELECT patient_id,event_id,concept,value_raw,exam_month,"
                "quality_status,imputed_flag FROM observations "
                "WHERE cohort='PPMI' AND event_id IN ('BL','SC') "
                "AND concept IN ("+",".join("?" for _ in FEATURES)+") "
                "AND patient_id IN ("+marks+")"
            )
            params=list(FEATURES)+part
            for pid,event,concept,value,month,quality,imputed in source.execute(
                    query,params).fetchall():
                if pid not in train_ids:
                    raise ValueError("Non-TRAIN patient returned by query")
                observations[pid][concept][event].append((value,month,quality,imputed))
    finally:
        source.close()
    return observations


def show(n,minimum):
    return str(n) if n == 0 or n >= minimum else f"<{minimum}"


def save(path,fields,rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(source_db,engine,manifest_db,report_dir,minimum=10):
    source_db,manifest_db,report_dir=(Path(p).expanduser().resolve()
        for p in (source_db,manifest_db,report_dir))
    if not source_db.is_file() or not manifest_db.is_file():
        raise FileNotFoundError("Missing source clinical DB or Phase23 manifest")
    if report_dir in (source_db,manifest_db) or minimum<10:
        raise ValueError("Unsafe report path or min-cell-count < 10")
    people,readiness=load_train_people(manifest_db)
    ids=sorted(people)
    all_obs=source_rows(source_db,engine,ids)
    counts=Counter()
    group_totals=Counter()
    joint=Counter()
    numeric_by_group=defaultdict(list)
    ready_by_group=defaultdict(set)
    for pid in ids:
        anchor,baseline_motor=people[pid]
        group="has_12m_target" if readiness[pid]=="paired" else "missing_12m_target"
        group_totals[group]+=1
        chosen=set()
        for concept in FEATURES:
            status,value=choose_feature(concept,all_obs[pid].get(concept,{}),anchor)
            if status not in STATUS:
                raise ValueError("Unexpected feature status")
            counts[(group,concept,status)]+=1
            if status in ("ready_BL","ready_SC"):
                chosen.add(concept)
                numeric_by_group[(group,concept)].append(value)
        ready_by_group[group].add(pid)
        if baseline_motor is not None:
            joint[(group,"BL_motor_present")]+=1
        if "moca_total" in chosen and "rbd_vivid_dreams" in chosen:
            joint[(group,"MoCA_and_RBD")]+=1
        if "moca_total" in chosen and "upsit_total" in chosen:
            joint[(group,"MoCA_and_UPSIT")]+=1
        if len(chosen)>=3:
            joint[(group,"three_or_more_nonmotor_features")]+=1
        if len(chosen)==len(FEATURES):
            joint[(group,"all_five_nonmotor_features")]+=1

    report_dir.mkdir(parents=True,exist_ok=True)
    save(report_dir/"train_feature_status_by_followup.csv",
        ["train_followup_observability","feature","status","patients"],[
        dict(train_followup_observability=g,feature=feature,status=status,
             patients=show(counts[(g,feature,status)],minimum))
        for g in STRATA for feature in FEATURES for status in STATUS])
    save(report_dir/"train_joint_coverage.csv",
         ["train_followup_observability","feature_bundle","patients"],[
         dict(train_followup_observability=g,feature_bundle=bundle,
              patients=show(joint[(g,bundle)],minimum))
         for g in STRATA for bundle in (
             "BL_motor_present","MoCA_and_RBD","MoCA_and_UPSIT",
             "three_or_more_nonmotor_features","all_five_nonmotor_features")])
    summaries=[]
    for g in STRATA:
        for feature in FEATURES:
            values=numeric_by_group[(g,feature)]
            summaries.append({
                "train_followup_observability":g,"feature":feature,
                "numeric_count":show(len(values),minimum),
                "numeric_mean":round(sum(values)/len(values),3) if len(values)>=minimum else "",
                "numeric_min":round(min(values),3) if len(values)>=minimum else "",
                "numeric_max":round(max(values),3) if len(values)>=minimum else "",
            })
    save(report_dir/"train_feature_value_summaries.csv",
         ["train_followup_observability","feature","numeric_count",
          "numeric_mean","numeric_min","numeric_max"],summaries)
    summary={
        "status":"TRAIN_ONLY_baseline_multimodal_feasibility_preflight",
        "population":"PPMI Phase23 prodromal TRAIN registry",
        "train_patients":len(ids),
        "train_12m_target_observability":dict(group_totals),
        "candidate_features":list(FEATURES),
        "temporal_rules":"BL same month as anchor; UPSIT SC strictly 1..6 months before anchor",
        "source_scope":"Only TRAIN BL/SC observations; no future numeric values queried",
        "review_required_for_item_numeric_ranges":list(PROVISIONAL_ITEM_RANGE),
        "methodology_caveats":[
            "The group has_12m_target is used for aggregated selection-bias QC ONLY.",
            "Never give readiness or any future target as a model input.",
            "Within-month BL measurement order is not recoverable.",
            "MoCA/UPSIT/age and questionnaire code semantics must be verified.",
            "Presence of a number is not a clinically validated observation.",
            "Patients lacking 12-month follow-up are retained in coverage denominators.",
            "A strict BL-month filter may drop legitimate prior baseline tests.",
            "No validation/test records, outcomes or patient-level report exported.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (report_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-db",type=Path,required=True)
    p.add_argument("--source-engine",choices=("duckdb","sqlite"),default="duckdb")
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,
                   default=Path("outputs/phase25_baseline_features"))
    p.add_argument("--min-cell-count",type=int,default=10)
    a=p.parse_args()
    print(json.dumps(audit(a.source_db,a.source_engine,a.manifest_db,
                           a.report_dir,a.min_cell_count),indent=2,ensure_ascii=False))
    print("Upload aggregate Phase25 report only; private databases stay on cluster.")


if __name__=="__main__":
    main()
