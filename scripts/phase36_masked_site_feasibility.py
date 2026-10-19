#!/usr/bin/env python3
"""Phase36: TRAIN-only same-visit masked Motor benchmark feasibility.

A V04 onsite exam with 33 scored items can be masked to its 27 remotely
observable components while retaining same-visit six-item ground truth.
A true R01 (6-month remote) sample is used ONLY for feature-shift audit,
never mislabeled using a months-later V04 as contemporaneous truth.
No patient records exported; original manifest never changed.
"""
import argparse,csv,json,math,sys
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
from statistics import mean
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase22_strict_motor_pairs import one_file,parse_month,state_name,safe,save
from scripts.phase31_motor_repetitions import train_anchors

ALL=("NP3SPCH NP3FACXP NP3RIGN NP3RIGRU NP3RIGLU NP3RIGRL NP3RIGLL "
     "NP3FTAPR NP3FTAPL NP3HMOVR NP3HMOVL NP3PRSPR NP3PRSPL "
     "NP3TTAPR NP3TTAPL NP3LGAGR NP3LGAGL NP3RISNG NP3GAIT "
     "NP3FRZGT NP3PSTBL NP3POSTR NP3BRADY NP3PTRMR NP3PTRML "
     "NP3KTRMR NP3KTRML NP3RTARU NP3RTALU NP3RTARL NP3RTALL "
     "NP3RTALJ NP3RTCON").split()
HIDDEN="NP3RIGN NP3RIGRU NP3RIGLU NP3RIGRL NP3RIGLL NP3PSTBL".split()
VISIBLE=tuple(x for x in ALL if x not in HIDDEN)
assert len(ALL)==33 and len(VISIBLE)==27 and len(HIDDEN)==6

def score(v):
    t=str(v or "").strip().upper()
    if t=="101" or t=="UR":return "unable",None
    if t in ("","NA","N/A","NAN","NULL","NONE","."):return "empty",None
    try:x=float(t)
    except ValueError:return "invalid",None
    if math.isfinite(x) and x in (0.,1.,2.,3.,4.):
        return "valid",int(x)
    return "invalid",None

def total(v):
    try:
        x=float(str(v).strip())
        return int(x) if x.is_integer() and 0<=x<=132 else None
    except (ValueError,TypeError,AttributeError):return None

def group_gap(v):
    return "1_3" if v<=3 else "4_6" if v<=6 else "7_10" if v<=10 else "11plus"

def band(v):
    return "0" if v==0 else "1_2" if v<=2 else "3_5" if v<=5 else "6plus"

def parse_rows(source,anchors,qc):
    """Yield at most eligible TRAIN R01/V04 visit records in RAM.

    Repeated scalars with discordant content are excluded. The same source
    exam month fallback as Phase23 is maintained.
    """
    with source.open(encoding="utf-8-sig",errors="replace",newline="") as f:
        reader=csv.reader(f)
        columns=defaultdict(list)
        for i,name in enumerate(next(reader,[])):
            columns[name.strip().upper()].append(i)
        required=set(ALL)|{"PATNO","EVENT_ID","NP3TOT"}
        if not required.issubset(columns):
            raise ValueError("Missing explicitly defined MDS-UPDRS III item headers: "
                             +",".join(sorted(required-set(columns))))
        for row in reader:
            def get(name):
                vals=[row[i].strip() for i in columns.get(name,[]) if i<len(row)]
                unique={s for s in vals if s}
                if len(unique)>1:raise ValueError("Contradictory repeated header")
                return next(iter(unique)) if unique else ""
            try:
                pid=get("PATNO")
                if pid not in anchors or anchors[pid] is None:continue
                event=get("EVENT_ID").upper()
                if event not in ("R01","V04"):continue
                m=parse_month(get("EXAMDT") or get("INFODT"))
                if m is None:
                    qc["selected_exam_month_missing"]+=1;continue
                gap=m-anchors[pid]
                if (event=="R01" and not 4<=gap<=8) or (
                        event=="V04" and not 9<=gap<=15):continue
                if state_name(get("PDSTATE"))!="UNSPECIFIED":continue
                fields={name:score(get(name)) for name in ALL}
                outcome=total(get("NP3TOT"))
            except ValueError:
                qc["selected_duplicate_scalar_or_item_conflict"]+=1
                continue
            yield pid,event,m,fields,outcome

def unique_by_patient(rows,qc):
    """Quarantine repeated or conflicting same-event exams."""
    candidates=defaultdict(list)
    for pid,event,m,fields,outcome in rows:
        candidates[(pid,event)].append((m,fields,outcome))
    selected={}
    for (pid,event),values in candidates.items():
        if len(values)!=1:
            qc["ambiguous_"+event+"_records_patients"]+=1
            continue
        selected[(pid,event)]=values[0]
    return selected

def audit(root,manifest,out,min_cell=10):
    root,manifest,out=[Path(p).expanduser().resolve() for p in (root,manifest,out)]
    if not (root/"PPMI").is_dir() or not manifest.is_file() or min_cell<10:
        raise ValueError("Missing input or unsafe suppression level")
    if out==manifest or out==root or root in out.parents:
        raise ValueError("Report must be outside source and private DB")
    source=one_file(root/"PPMI","MDS-UPDRS_Part_III_")
    anchors=train_anchors(manifest)
    qc=Counter()
    selected=unique_by_patient(parse_rows(source,anchors,qc),qc)
    remote={};site={}
    stage=Counter();category=Counter()
    for (pid,event),(month,fields,outcome) in selected.items():
        visible=[fields[name] for name in VISIBLE]
        hidden=[fields[name] for name in HIDDEN]
        n_visible=sum(t=="valid" for t,v in visible)
        all_six_unable=all(t=="unable" for t,v in hidden)
        complete_visible=n_visible==27
        if event=="R01":
            stage["R01_unique_patient"]+=1
            if outcome is None and all_six_unable:
                stage["R01_no_total_six_unable"]+=1
                if complete_visible:stage["R01_no_total_six_unable_27valid"]+=1
            remote[pid]=(month,fields,outcome,complete_visible,all_six_unable)
        else:
            stage["V04_unique_patient"]+=1
            if outcome is None:stage["V04_no_total"]+=1
            if outcome is not None and complete_visible:
                stage["V04_valid_total_27valid"]+=1
            all33=all(t=="valid" for t,v in fields.values())
            if outcome is not None and all33:
                stage["V04_all33_numeric_and_total"]+=1
                if sum(v for t,v in fields.values())==outcome:
                    stage["V04_verified_33item_sum"]+=1
                    six_total=sum(v for t,v in hidden)
                    category[("site_hidden_six_score",band(six_total))]+=1
                    category[("site_six_nonzero",str(sum(v>0 for t,v in hidden)))]+=1
                    site[pid]=(month,fields,outcome)
                else:qc["V04_total_not_equal_sum33"]+=1
            elif outcome is not None:qc["V04_total_with_some_items_non_numeric"]+=1
    source_pairs=Counter()
    shift_values=defaultdict(list)
    for pid,(rmonth,rfields,rtotal,vis_ok,unable) in remote.items():
        if unable and vis_ok and rtotal is None:
            for name in VISIBLE:shift_values[(name,"real_R01")].append(rfields[name][1])
        if pid not in site:continue
        vmonth,vfields,vtotal=site[pid]
        if vmonth<=rmonth:
            qc["V04_not_later_than_R01"]+=1
            continue
        source_pairs["R01_any_to_same_patient_V04_verified"]+=1
        if unable and vis_ok and rtotal is None:
            source_pairs["R01_27valid_6unable_to_later_V04_verified"]+=1
            category[("observed_R01_to_V04_gap",group_gap(vmonth-rmonth))]+=1
    for pid,(month,fields,vtotal) in site.items():
        for name in VISIBLE:shift_values[(name,"masked_V04")].append(fields[name][1])
    out.mkdir(parents=True,exist_ok=True)
    save(out/"record_readiness.csv",["criterion","TRAIN_patients"],[
        {"criterion":label,"TRAIN_patients":safe(n,min_cell)}
        for label,n in sorted({**stage,**source_pairs}.items())])
    save(out/"onsite_hidden_six_profile.csv",
         ["measure","category","TRAIN_patients"],[
         {"measure":measure,"category":value,
          "TRAIN_patients":safe(n,min_cell)}
         for (measure,value),n in sorted(category.items())])
    distribution=[]
    for item in VISIBLE:
        for dataset in ("real_R01","masked_V04"):
            values=shift_values[item,dataset]
            distribution.append({
                "item":item,"sample":dataset,
                "patients":safe(len(values),min_cell),
                "mean_item_score":round(mean(values),3) if len(values)>=min_cell else "",
                "nonzero_pct":round(100*sum(v>0 for v in values)/len(values),1)
                              if len(values)>=min_cell else "",
            })
    save(out/"visible_item_domain_shift.csv",[
        "item","sample","patients","mean_item_score","nonzero_pct"],distribution)
    save(out/"quality_flags.csv",["check","patients_or_records"],[
        {"check":k,"patients_or_records":safe(v,min_cell)}
        for k,v in sorted(qc.items())])
    summary={
        "status":"TRAIN_only_same_visit_masked_onsite_feasibility_NO_predictions",
        "train_registry":len(anchors),"source_file":source.name,
        "clinically_defined_visible_items":len(VISIBLE),
        "clinically_defined_unobservable_remote_items":len(HIDDEN),
        "eligible_V04_same_visit_labels":safe(stage["V04_verified_33item_sum"],min_cell),
        "real_R01_six_unable_27valid":safe(
            stage["R01_no_total_six_unable_27valid"],min_cell),
        "R01_to_later_V04_verified_pairs":safe(
            source_pairs["R01_27valid_6unable_to_later_V04_verified"],min_cell),
        "cautions":[
            "At V04 the same-visit six hidden item labels exist; masking is a counterfactual measurement simulation, not real randomized exams.",
            "Real R01 visible items are months earlier than V04, not same-time labels.",
            "R01/V04 between-visit changes and selection bias invalidate naive same-day supervision.",
            "A valid 33-item sum is only a strict provenance cross-check, not approval to reconstruct any R01 total.",
            "Medication state UNSPECIFIED remains unverified.",
            "No patient-level observations, labels, dates or predictions exported.",
            "No patient split changed; no val/test queried.",
        ],"generated_utc":datetime.now(timezone.utc).isoformat()}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,default=Path("outputs/phase36_masked_site"))
    p.add_argument("--min-cell-count",type=int,default=10)
    a=p.parse_args()
    print(json.dumps(audit(a.data_root,a.manifest_db,a.report_dir,a.min_cell_count),
                     ensure_ascii=False,indent=2))
    print("Upload ONLY aggregate Phase36 reports. Keep patient DB private.")
if __name__=="__main__":main()
