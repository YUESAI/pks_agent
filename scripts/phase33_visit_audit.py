#!/usr/bin/env python3
"""Phase33: aggregate PPMI TRAIN 6-month Motor visit-code completeness.

Read-only. Never export identifiers, individual scores, or raw dates.
A missing NP3TOT is not evidence of a completed Motor examination.
"""
import argparse,csv,json,math,re,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase22_strict_motor_pairs import one_file,parse_month,state_name,safe,save
from scripts.phase31_motor_repetitions import train_anchors,context_for

EMPTY={"","na","n/a","nan","null","none","missing","."}
BADCODES={"-1","-7","-8","-9","-99","888","999","9999"}

def numeric_present(v):
    token=str(v or "").strip().lower()
    if token in EMPTY or token in BADCODES:return False
    try:return math.isfinite(float(token)) and float(token)>=0
    except ValueError:return False

def total_kind(v):
    if str(v or "").strip().lower() in EMPTY:return "empty"
    try:
        x=float(v)
        return "valid" if math.isfinite(x) and 0<=x<=132 else "invalid"
    except (TypeError,ValueError):return "invalid"

def item_bucket(n):
    return "none" if n==0 else "1_to_4" if n<=4 else "5_to_9" if n<=9 else "10plus"

def analyze(root,manifest,out,minimum=10):
    root,manifest,out=(Path(v).expanduser().resolve() for v in (root,manifest,out))
    if not (root/"PPMI").is_dir() or not manifest.is_file():
        raise FileNotFoundError("Missing original Motor CSV or local manifest")
    if out==manifest or root==out or root in out.parents or minimum<10:
        raise ValueError("Invalid report location or small-cell threshold")
    anchors=train_anchors(manifest)
    source=one_file(root/"PPMI","MDS-UPDRS_Part_III_")
    qc=Counter();records=[];patients=defaultdict(set)
    with source.open(encoding="utf-8-sig",errors="replace",newline="") as f:
        reader=csv.reader(f)
        pos=defaultdict(list)
        for i,name in enumerate(next(reader,[])):pos[name.strip()].append(i)
        for name in ("PATNO","EVENT_ID","NP3TOT"):
            if name not in pos:raise ValueError("Missing required field "+name)
        items=[key for key in pos if key.upper().startswith("NP3")
               and key.upper()!="NP3TOT"]
        for row in reader:
            def field(name):
                vals=[row[i].strip() for i in pos.get(name,[]) if i<len(row)]
                vals=[x for x in vals if x]
                if len(set(vals))>1:raise ValueError("Duplicate header conflict")
                return vals[0] if vals else ""
            try:pid=field("PATNO")
            except ValueError:
                qc["conflicting_PATNO_column"]+=1
                continue
            if pid not in anchors:continue
            if anchors[pid] is None:
                qc["TRAIN_rows_no_anchor"]+=1
                continue
            try:
                event=field("EVENT_ID").upper()
                month=parse_month(field("EXAMDT") or field("INFODT"))
                state=state_name(field("PDSTATE"))
                total=total_kind(field("NP3TOT"))
            except ValueError:
                qc["TRAIN_conflicting_scalar_columns"]+=1
                continue
            if month is None:
                qc["TRAIN_rows_undated"]+=1
                continue
            window=context_for(event,month,anchors[pid])
            if window not in ("baseline","6m_window","12m_window",
                             "24m_window","other_followup"):
                qc["TRAIN_excluded_visit_or_date"]+=1
                continue
            if state!="UNSPECIFIED":
                qc["TRAIN_other_drug_state"]+=1
                continue
            if not re.fullmatch(r"[A-Z][A-Z0-9_-]{0,11}",event):
                event="UNUSUAL_CODE"
            item_count=sum(any(i<len(row) and numeric_present(row[i])
                               for i in pos[name]) for name in items)
            records.append((window,event,total,item_bucket(item_count),pid))
            patients[event].add(pid)
    allowed={e for e,ps in patients.items()
             if len(ps)>=minimum and e!="UNUSUAL_CODE"}
    counts=Counter();subject=defaultdict(set)
    for window,event,total,items,pid in records:
        group=event if event in allowed else "RARE_EVENT_CODES"
        counts[(window,group,"rows")]+=1
        counts[(window,group,total)]+=1
        if total!="valid":counts[(window,group,"items_"+items)]+=1
        subject[(window,group)].add(pid)
    out.mkdir(parents=True,exist_ok=True)
    events=[];item_counts=[]
    for window,event in sorted(subject):
        events.append(dict(window=window,event_code=event,
          patients=safe(len(subject[window,event]),minimum),
          rows=safe(counts[window,event,"rows"],minimum),
          total_valid=safe(counts[window,event,"valid"],minimum),
          total_empty=safe(counts[window,event,"empty"],minimum),
          total_invalid=safe(counts[window,event,"invalid"],minimum)))
        for bucket in ("none","1_to_4","5_to_9","10plus"):
            item_counts.append(dict(window=window,event_code=event,
              populated_NP3_item_fields=bucket,
              rows_without_valid_total=safe(
                  counts[window,event,"items_"+bucket],minimum)))
    save(out/"event_code_totals.csv",
         ["window","event_code","patients","rows","total_valid",
          "total_empty","total_invalid"],events)
    save(out/"missing_total_item_evidence.csv",
         ["window","event_code","populated_NP3_item_fields",
          "rows_without_valid_total"],item_counts)
    save(out/"quality_flags.csv",["flag","rows"],
         [{"flag":k,"rows":safe(v,minimum)} for k,v in sorted(qc.items())])
    summary={"status":"TRAIN_ONLY_6m_event_source_preflight",
       "source_file":source.name,"train_population":len(anchors),
       "NP3_item_columns":len(items),
       "cautions":["Events only displayed if at least ten TRAIN patients use code.",
         "Blank NP3TOT does not prove whether an exam was attempted.",
         "Other Part III items cannot be converted to a valid total without codebook review.",
         "No patient IDs, individual scores, dates or raw rows exported.",
         "No val/test patients or prior manifest modified."],
       "generated_utc":datetime.now(timezone.utc).isoformat()}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,default=Path("outputs/phase33_visit_audit"))
    p.add_argument("--min-cell-count",type=int,default=10)
    a=p.parse_args()
    print(json.dumps(analyze(a.data_root,a.manifest_db,a.report_dir,
                             a.min_cell_count),indent=2))
if __name__=="__main__":main()
