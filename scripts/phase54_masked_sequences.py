"""Build masked longitudinal TRAIN patient sequences from verified source exams."""
from collections import Counter,defaultdict
from pathlib import Path
from scripts.phase22_strict_motor_pairs import one_file,parse_month,state_name
from scripts.phase31_motor_repetitions import train_anchors
from scripts.phase36_masked_site_feasibility import ALL,score,total
from scripts.phase53_mask_aware_motor_readiness import iter_source,categorize_episode

ITEMS=tuple(ALL)

def sequences_from_rows(rows,anchors):
    grouped=defaultdict(list)
    qc=Counter()
    for row in rows:
        pid=str(row.get("PATNO","")).strip()
        if pid not in anchors or anchors[pid] is None:continue
        event=str(row.get("EVENT_ID","")).strip().upper()
        month=parse_month(row.get("EXAMDT") or row.get("INFODT"))
        if month is None or event in ("","SC","CONSENT"):continue
        diff=month-anchors[pid]
        if (event=="BL" and diff!=0) or (event!="BL" and diff<=0):continue
        if event=="R01" and not 4<=diff<=8:continue
        if event=="V04" and not 9<=diff<=15:continue
        if state_name(row.get("PDSTATE",""))!="UNSPECIFIED":continue
        values={name:score(row.get(name,"")) for name in ITEMS}
        kind,nvalid,_=categorize_episode(values,total(row.get("NP3TOT","")))
        if kind in ("invalid_full_total_mismatch","unusable_below10"):continue
        # 101/UR => status 2; other absent => 0; measured => 1.
        statuses=tuple(1 if values[k][0]=="valid" else
                       2 if values[k][0]=="unable" else 0 for k in ITEMS)
        scores=tuple(values[k][1] if values[k][0]=="valid" else 0 for k in ITEMS)
        entry={"month":month,"event":event,"score":scores,"status":statuses}
        grouped[(pid,month)].append(entry)
    unique=defaultdict(dict)
    for (pid,month),candidates in grouped.items():
        identity={(r["event"],r["score"],r["status"]) for r in candidates}
        if len(identity)!=1:
            qc["conflicting_patient_month_keys"]+=1
            continue
        unique[pid][month]=candidates[0]
        if len(candidates)>1:qc["identical_duplicate_groups"]+=1
    result=[];anchored=0;total_pairs=0
    for pid,anchor in anchors.items():
        timeline=unique.get(pid,{})
        if anchor not in timeline or timeline[anchor]["event"]!="BL":continue
        anchored+=1
        visits=sorted(timeline.values(),key=lambda x:x["month"])
        total_pairs+=len(visits)-1
        if len(visits)>=2:result.append({"pid":pid,"visits":visits})
    meta={"frozen_train_patients":len(anchors),"anchored_patients":anchored,
          "patients_with_2plus_visits":len(result),
          "anchored_transition_pairs":total_pairs,
          "qc":dict(qc)}
    if not result:raise ValueError("No eligible sequences")
    return result,meta

def load_train(data_root,manifest):
    root=Path(data_root).expanduser().resolve()
    db=Path(manifest).expanduser().resolve()
    if not (root/"PPMI").is_dir() or not db.is_file():
        raise FileNotFoundError("Missing local PPMI data or frozen manifest")
    anchors=train_anchors(db)
    source=one_file(root/"PPMI","MDS-UPDRS_Part_III_")
    return sequences_from_rows(iter_source(source,anchors),anchors)
