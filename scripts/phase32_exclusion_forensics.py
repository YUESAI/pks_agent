#!/usr/bin/env python3
"""Phase32: explain Phase23 Motor-key exclusions, frozen TRAIN only.

Reproduce Phase23's patient/BL-or-FU/calendar-month key exactly and distinguish
absent NP3TOT, numeric sentinel/out-of-range, malformed, mixed valid+invalid
and true repeated-score disagreements. Verify the resulting TRAIN readiness
and target pairs against the frozen private Phase23 SQLite manifest.

NO manifest rewrite. No individual patient IDs, scores, dates or keys exported.
The raw CSV is streamed, but only frozen TRAIN subjects are retained.
"""
from __future__ import annotations
import argparse,csv,json,math,re,sqlite3,sys
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase22_strict_motor_pairs import (
    one_file,records,parse_month,state_name,target_for_window,save,safe
)
from scripts.phase31_motor_repetitions import train_anchors

WINDOWS=((6,2),(12,3),(24,4))
REASONS=("valid_single","valid_duplicate_same_event","missing_only",
         "negative_or_sentinel_only","above_max_only","non_numeric_only",
         "mixed_invalid_categories","mixed_valid_and_invalid",
         "same_event_score_disagreement","multiple_events_same_score",
         "multiple_events_score_disagreement")
CONTEXTS=("BL","6m","12m","24m","other_FU")
CODED={"-1","-7","-8","-9","-99","888","999","9999"}

def score_kind(text):
    text=str(text or "").strip()
    if not text or text.lower() in {"nan","na","n/a","null","none","missing","."}:
        return "missing",None
    if text in CODED:return "negative_or_sentinel",None
    try:value=float(text)
    except (TypeError,ValueError):return "non_numeric",None
    if not math.isfinite(value):return "non_numeric",None
    if value<0:return "negative_or_sentinel",None
    if value>132:return "above_max",None
    return "valid",value

def visit_context(kind,month,anchor):
    if kind=="BL":return "BL"
    gap=month-anchor
    if 4<=gap<=8:return "6m"
    if 9<=gap<=15:return "12m"
    if 20<=gap<=28:return "24m"
    return "other_FU"

def visit_code(raw):
    raw=str(raw or "").strip().upper()
    return raw if re.fullmatch(r"(?:BL|V\d{2})",raw) else "OTHER_EVENT"

def reason(entry):
    values=entry["values"]
    events=entry["events"]
    errors=set(entry["errors"])
    if values and errors:return "mixed_valid_and_invalid"
    if not values:
        if len(errors)>1:return "mixed_invalid_categories"
        return ({"missing":"missing_only",
                 "negative_or_sentinel":"negative_or_sentinel_only",
                 "above_max":"above_max_only",
                 "non_numeric":"non_numeric_only"}).get(
                 next(iter(errors),"non_numeric"),"non_numeric_only")
    if len(values)>1:
        return ("multiple_events_score_disagreement" if len(events)>1
                else "same_event_score_disagreement")
    if len(events)>1:return "multiple_events_same_score"
    return "valid_duplicate_same_event" if entry["rows"]>1 else "valid_single"

def accepted(entry):
    # EXACT Phase23 acceptance predicate (invalid flag + score/event sets).
    return not entry["errors"] and len(entry["values"])==1 and len(entry["events"])==1

def manifest_train(manifest):
    db=sqlite3.connect("file:"+str(manifest.resolve())+"?mode=ro",uri=True)
    try:
        people=dict(db.execute(
            "SELECT pid,baseline_motor FROM patients WHERE split='train'").fetchall())
        ready={(pid,h):status for pid,h,status in db.execute(
            "SELECT r.pid,r.horizon,r.status FROM readiness r "
            "JOIN patients p ON p.pid=r.pid WHERE p.split='train'").fetchall()}
        targets={(pid,h):(float(score),int(month)) for pid,h,score,month in db.execute(
            "SELECT t.pid,t.horizon,t.target_motor,t.target_month FROM targets t "
            "JOIN patients p ON p.pid=t.pid WHERE p.split='train'").fetchall()}
    finally:db.close()
    if not people or len(ready)!=3*len(people):
        raise ValueError("Incomplete frozen TRAIN manifest")
    return people,ready,targets

def audit(root,manifest,out,minimum=10):
    root,manifest,out=(Path(x).expanduser().resolve() for x in (root,manifest,out))
    if not (root/"PPMI").is_dir() or not manifest.is_file() or minimum<10:
        raise ValueError("Missing source/manifest or privacy threshold too small")
    if out==manifest or out==root or root in out.parents:
        raise ValueError("Reports may not overwrite private data")
    anchors=train_anchors(manifest)
    people,readiness,targets=manifest_train(manifest)
    if set(anchors)!=set(people):
        raise ValueError("TRAIN registry differs between audit and manifest")
    source=one_file(root/"PPMI","MDS-UPDRS_Part_III_")
    quality=Counter()
    raw=defaultdict(lambda:{"rows":0,"values":set(),"events":set(),"errors":Counter()})
    row_kinds=Counter()
    for row in records(source,("PATNO","EVENT_ID","NP3TOT"),quality):
        pid=row.get("PATNO","")
        if pid not in people:continue
        anchor=anchors[pid]
        if anchor is None:
            quality["no_strict_BL_month_anchor_rows"]+=1
            continue
        event=row.get("EVENT_ID","")
        month=parse_month(row.get("EXAMDT") or row.get("INFODT"))
        if month is None:
            quality["unparseable_exam_month_rows"]+=1
            continue
        if event=="BL":
            if month!=anchor:
                quality["BL_event_month_not_anchor_rows"]+=1
                continue
            kind="BL"
        elif event in ("","SC","CONSENT") or month<=anchor:
            quality["not_eligible_followup_event_rows"]+=1
            continue
        else:kind="FU"
        if state_name(row.get("PDSTATE"))!="UNSPECIFIED":
            quality["excluded_other_medication_state_rows"]+=1
            continue
        key=(pid,kind,month) # EXACT Phase23 key, not a window-scoped key
        item=raw[key]
        item["rows"]+=1
        item["events"].add(event)
        value_type,value=score_kind(row.get("NP3TOT"))
        context=visit_context(kind,month,anchor)
        row_kinds[(context,visit_code(event),value_type)]+=1
        if value_type=="valid":item["values"].add(value)
        else:item["errors"][value_type]+=1
    mechanisms=Counter();rejected=Counter();patient_counts=defaultdict(set)
    byvisit=Counter()
    bl={};fu=defaultdict(dict);bl_conflict=set()
    for (pid,kind,month),entry in raw.items():
        context=visit_context(kind,month,anchors[pid])
        result=reason(entry)
        mechanisms[(context,result)]+=1
        patient_counts[(context,result)].add(pid)
        byvisit[context]+=1
        if not accepted(entry):
            rejected[context]+=1
            if kind=="BL":bl_conflict.add(pid)
            continue
        value=next(iter(entry["values"]))
        if kind=="BL":bl[pid]=value
        else:fu[pid][month]=value

    mismatches=Counter()
    replay_status=Counter()
    matched_pairs=Counter()
    for pid,anchor in anchors.items():
        recorded=people[pid]
        if (recorded is None)!=(pid not in bl):
            mismatches["BL_value_presence"]+=1
        if recorded is not None and pid in bl and abs(float(recorded)-bl[pid])>1e-6:
            mismatches["BL_value_different"]+=1
        for horizon,tol in WINDOWS:
            if anchor is None:status,chosen="no_anchor",None
            elif pid not in bl:
                status,chosen=("conflicting_BL" if pid in bl_conflict
                               else "no_BL_motor"),None
            else:
                status,chosen=target_for_window(fu.get(pid,{}),anchor,horizon,tol)
            replay_status[(horizon,status)]+=1
            if readiness[(pid,horizon)]!=status:
                mismatches["readiness_status"]+=1
            recorded_target=targets.get((pid,horizon))
            if status=="paired":
                if chosen is None or recorded_target is None:
                    mismatches["target_presence"]+=1
                else:
                    m,value=chosen
                    if m!=recorded_target[1] or abs(value-recorded_target[0])>1e-6:
                        mismatches["target_value_or_month"]+=1
                    else:matched_pairs[horizon]+=1
            elif recorded_target is not None:
                mismatches["unexpected_manifest_target"]+=1
    out.mkdir(parents=True,exist_ok=True)
    save(out/"key_exclusion_reasons.csv",
         ["window","reason","patient_month_keys"],[
         {"window":c,"reason":r,"patient_month_keys":safe(mechanisms[c,r],minimum)}
         for c in CONTEXTS for r in REASONS])
    save(out/"source_value_types_by_event.csv",
         ["window","event_category","value_type","source_rows"],[
         {"window":c,"event_category":e,"value_type":t,
          "source_rows":safe(n,minimum)}
         for (c,e,t),n in sorted(row_kinds.items()) if n>=minimum])
    save(out/"exclusion_reconciliation.csv",
         ["window","all_keys","excluded_keys","accepted_keys"],[
         {"window":c,"all_keys":safe(byvisit[c],minimum),
          "excluded_keys":safe(rejected[c],minimum),
          "accepted_keys":safe(byvisit[c]-rejected[c],minimum)}
         for c in CONTEXTS])
    save(out/"frozen_manifest_check.csv",
         ["check","count"],[
         {"check":k,"count":safe(mismatches[k],minimum)}
         for k in ("BL_value_presence","BL_value_different","readiness_status",
                   "target_presence","target_value_or_month",
                   "unexpected_manifest_target")]+[
         {"check":"matched_target_horizon_"+str(h),
          "count":safe(matched_pairs[h],minimum)} for h,_ in WINDOWS])
    save(out/"train_replay_readiness.csv",["horizon","status","patients"],[
         {"horizon":h,"status":status,"patients":safe(n,minimum)}
         for (h,status),n in sorted(replay_status.items())])
    save(out/"row_quality_flags.csv",["flag","rows"],[
         {"flag":k,"rows":safe(n,minimum)} for k,n in sorted(quality.items())])
    total_mismatches=sum(mismatches.values())
    info={
        "status":"TRAIN_ONLY_Phase23_exact_key_definition_forensic_audit",
        "train_registry":len(people),
        "phase23_replayed_excluded_keys_train":sum(rejected.values()),
        "phase23_replayed_accepted_keys_train":
            sum(byvisit.values())-sum(rejected.values()),
        "manifest_comparison_total_mismatches":total_mismatches,
        "matched_manifest_targets_by_horizon":
            {str(h):matched_pairs[h] for h,_ in WINDOWS},
        "source_file":source.name,
        "cautions":[
            "Phase23's 583 excluded keys were counted over all cohort splits; this script is TRAIN only.",
            "Invalid includes empty NP3TOT rows; invalid is NOT synonymous with duplicate scores.",
            "No previously frozen patient manifest or source records were changed.",
            "Never automatically salvage a valid score from a mixed valid/invalid key.",
            "Different EVENT_IDs at a month may reflect distinct exams, not test-retest replicates.",
            "No validation/test patient measurements processed or exported.",
            "No patient identifiers, scores, raw dates, or visit-level rows exported.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (out/"summary.json").write_text(
        json.dumps(info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return info

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,
                   default=Path("outputs/phase32_exclusion_audit"))
    p.add_argument("--min-cell-count",type=int,default=10)
    a=p.parse_args()
    print(json.dumps(audit(a.data_root,a.manifest_db,
                           a.report_dir,a.min_cell_count),
                     indent=2,ensure_ascii=False))
    print("Share only aggregated Phase32 reports; never upload patient databases.")
if __name__=="__main__":
    main()
