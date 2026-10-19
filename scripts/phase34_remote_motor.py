#!/usr/bin/env python3
"""Phase34: TRAIN-only remote vs onsite Motor score-item missingness audit.

Read original PPMI MDS-UPDRS III CSV, retain only frozen TRAIN subjects.
Explicitly distinguish UR (unable to rate), blank, valid numeric and other
codes for six in-person-only rigidity/postural stability subitems.
No score imputation or manifest modification. Export aggregates only.
"""
import argparse,csv,json,math,sqlite3,sys
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase22_strict_motor_pairs import one_file,parse_month,state_name,safe,save
from scripts.phase31_motor_repetitions import train_anchors,context_for

SPECIAL=("NP3RIGN","NP3RIGRU","NP3RIGLU","NP3RIGRL","NP3RIGLL","NP3PSTBL")
WINDOWS=("6m_window","12m_window","24m_window")
STATUS=("numeric_0_to_4","UR","blank","other_code","not_in_source")
EMPTY={"","na","n/a","nan","null","none","missing","."}

def item_status(value):
    s=str(value or "").strip().upper()
    if s=="UR":return "UR"
    if s.lower() in EMPTY:return "blank"
    try:
        f=float(s)
        return "numeric_0_to_4" if math.isfinite(f) and f in (0.,1.,2.,3.,4.) else "other_code"
    except (ValueError,TypeError):
        return "other_code"

def audit(root,manifest,out,min_cell=10):
    root,manifest,out=[Path(x).expanduser().resolve() for x in (root,manifest,out)]
    if not (root/"PPMI").is_dir() or not manifest.is_file() or min_cell<10:
        raise ValueError("Missing source/manifest or unsafe cell threshold")
    if root==out or root in out.parents or manifest==out:
        raise ValueError("Unsafe output location")
    anchors=train_anchors(manifest)
    source=one_file(root/"PPMI","MDS-UPDRS_Part_III_")
    totals=Counter();counts=Counter();missing_headers=[]
    with source.open(encoding="utf-8-sig",errors="replace",newline="") as f:
        reader=csv.reader(f)
        cols=defaultdict(list)
        for i,name in enumerate(next(reader,[])):
            cols[name.strip().upper()].append(i)
        for required in ("PATNO","EVENT_ID","NP3TOT"):
            if required not in cols:raise ValueError("Missing required "+required)
        missing_headers=[name for name in SPECIAL if name not in cols]
        for row in reader:
            def value(name):
                return [row[i].strip() for i in cols.get(name,[]) if i<len(row)]
            pid=value("PATNO")
            if len(set(pid))>1 or not pid or pid[0] not in anchors:continue
            anchor=anchors[pid[0]]
            if anchor is None:continue
            events=value("EVENT_ID")
            if not events or len(set(events))>1:continue
            event=events[0].upper()
            dates=value("EXAMDT") or value("INFODT")
            if not dates or len(set(dates))>1:continue
            month=parse_month(dates[0])
            if month is None:continue
            window=context_for(event,month,anchor)
            if window not in WINDOWS:continue
            state=value("PDSTATE")
            if state and state_name(state[0])!="UNSPECIFIED":continue
            if event not in ("R01","V04","V06"):continue
            raw_total=value("NP3TOT")
            if not raw_total or len(set(raw_total))>1:continue
            if raw_total[0].strip():continue
            totals[(window,event)]+=1
            for item in SPECIAL:
                values=value(item)
                if item in missing_headers:
                    label="not_in_source"
                elif len(set(values))>1:
                    label="other_code"
                else:label=item_status(values[0] if values else "")
                counts[(window,event,item,label)]+=1
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for window in WINDOWS:
        for event in ("R01","V04","V06"):
            for item in SPECIAL:
                n=totals[window,event]
                if not n:continue
                rows.append(dict(window=window,event_code=event,item=item,
                    missing_total_rows=safe(n,min_cell),
                    **{label:safe(counts[window,event,item,label],min_cell)
                       for label in STATUS}))
    save(out/"six_item_missing_total_status.csv",
         ["window","event_code","item","missing_total_rows",*STATUS],rows)
    save(out/"event_missing_total_counts.csv",
         ["window","event_code","missing_total_rows"],
         [dict(window=w,event_code=e,missing_total_rows=safe(n,min_cell))
          for (w,e),n in sorted(totals.items())])
    summary={
        "status":"TRAIN_ONLY_remote_motor_UR_missingness_forensics",
        "source":source.name,"frozen_train_patients":len(anchors),
        "missing_six_item_headers":missing_headers,
        "scope":"Only R01 V04 V06 6/12/24 month visits with empty NP3TOT",
        "cautions":[
            "UR means unable to rate; confirm exact data dictionary semantics.",
            "Missing total may be due to several independent incomplete items.",
            "Only aggregated item-status frequencies exported.",
            "This is NOT a reconstructed total or validation of remotely assessed Motor.",
            "No source or private manifest modifications; val/test patients excluded."
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat()
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",required=True,type=Path)
    p.add_argument("--manifest-db",required=True,type=Path)
    p.add_argument("--report-dir",type=Path,default=Path("outputs/phase34_remote_motor"))
    p.add_argument("--min-cell-count",type=int,default=10)
    a=p.parse_args()
    print(json.dumps(audit(a.data_root,a.manifest_db,a.report_dir,a.min_cell_count),
                     indent=2))
if __name__=="__main__":main()
