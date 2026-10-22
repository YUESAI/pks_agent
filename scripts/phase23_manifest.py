#!/usr/bin/env python3
"""Private prodromal longitudinal manifest. Share aggregate reports only."""
import argparse,csv,hashlib,json,sqlite3,sys
from pathlib import Path
from collections import defaultdict,Counter
from statistics import mean
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase22_strict_motor_pairs import one_file,records,parse_month,state_name,target_for_window,present,safe,save

P={"status":"Participant_Status_","diag":"Primary_Research_Diagnosis_","anchor":"Baseline_Visit_Start_","motor":"MDS-UPDRS_Part_III_"}
WINDOWS=((6,2),(12,3),(24,4))

def split_all(ids,seed):
    ranked=sorted(ids,key=lambda p:hashlib.sha256(f"{seed}:PPMI:prodromal:{p}".encode()).hexdigest())
    n=len(ranked);t=round(n*.7);v=round(n*.15)
    return {p:"train" if i<t else "val" if i<t+v else "test" for i,p in enumerate(ranked)}

def assemble(root):
    folder=root/"PPMI"
    files={k:one_file(folder,v) for k,v in P.items()}
    qc=Counter()
    cohorts=defaultdict(set);diags=defaultdict(set);anchors=defaultdict(set)
    for r in records(files["status"],("PATNO","COHORT"),qc):
        if present(r["PATNO"]) and present(r["COHORT"]):
            cohorts[r["PATNO"]].add(r["COHORT"])
    for r in records(files["diag"],("PATNO","EVENT_ID","PRIMDIAG"),qc):
        if r["EVENT_ID"]=="BL" and present(r["PATNO"]):
            diags[r["PATNO"]].add(r["PRIMDIAG"] or None)
    for r in records(files["anchor"],("PATNO","EVENT_ID","INFODT"),qc):
        if r["EVENT_ID"]=="BL" and present(r["PATNO"]):
            anchors[r["PATNO"]].add(parse_month(r["INFODT"]))
    people={}
    for pid in cohorts.keys()|diags.keys():
        if cohorts[pid]=={"4"} and diags[pid]=={"25"}:
            a=anchors[pid]
            people[pid]=next(iter(a)) if len(a)==1 and None not in a else None
    if not people:
        raise ValueError("No strict prodromal baseline candidates")
    raw=defaultdict(lambda:{"values":set(),"events":set(),"invalid":False})
    for r in records(files["motor"],("PATNO","EVENT_ID","NP3TOT"),qc):
        pid,e=r["PATNO"],r["EVENT_ID"]
        if pid not in people or people[pid] is None:
            continue
        m=parse_month(r.get("EXAMDT") or r.get("INFODT"))
        if m is None:
            continue
        if e=="BL":
            if m!=people[pid]:
                qc["baseline_month_mismatch"]+=1
                continue
            kind="BL"
        elif e in ("","SC","CONSENT") or m<=people[pid]:
            continue
        else:
            kind="FU"
        if state_name(r.get("PDSTATE"))!="UNSPECIFIED":
            continue
        v=raw[(pid,kind,m)]
        v["events"].add(e)
        try:
            x=float(r["NP3TOT"])
            if not 0<=x<=132:
                raise ValueError("out of range")
            v["values"].add(x)
        except (ValueError,TypeError):
            v["invalid"]=True
    bl={};fu=defaultdict(dict);bad_bl=set()
    for (pid,kind,m),v in raw.items():
        if v["invalid"] or len(v["values"])!=1 or len(v["events"])!=1:
            qc["conflicting_patient_month_exam"]+=1
            if kind=="BL":
                bad_bl.add(pid)
            continue
        score=next(iter(v["values"]))
        if kind=="BL":
            bl[pid]=score
        else:
            fu[pid][m]=score
    readiness={};pairs=[]
    for pid,a in people.items():
        for h,tol in WINDOWS:
            if a is None:
                status,chosen="no_anchor",None
            elif pid not in bl:
                status,chosen=("conflicting_BL" if pid in bad_bl else "no_BL_motor"),None
            else:
                status,chosen=target_for_window(fu.get(pid,{}),a,h,tol)
            readiness[(pid,h)]=status
            if status=="paired":
                m,score=chosen
                pairs.append((pid,h,a,m,m-a,bl[pid],score,score-bl[pid]))
    return people,bl,readiness,pairs,qc

def build(root,dbpath,reports,seed=20261018):
    root,dbpath,reports=[Path(x).expanduser().resolve() for x in (root,dbpath,reports)]
    if not (root/"PPMI").is_dir() or dbpath.exists() or root in dbpath.parents or root in reports.parents:
        raise ValueError("Missing raw data or unsafe existing/output database path")
    people,bl,ready,pairs,qc=assemble(root)
    groups=split_all(people,seed)
    dbpath.parent.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(dbpath)
    try:
        conn.executescript("""CREATE TABLE patients(pid TEXT PRIMARY KEY, split TEXT, baseline_month INTEGER, baseline_motor REAL);
        CREATE TABLE readiness(pid TEXT,horizon INTEGER,status TEXT,PRIMARY KEY(pid,horizon));
        CREATE TABLE targets(pid TEXT,horizon INTEGER,baseline_month INTEGER,target_month INTEGER,months_apart INTEGER,baseline_motor REAL,target_motor REAL,delta_motor REAL,PRIMARY KEY(pid,horizon));""")
        conn.executemany("INSERT INTO patients VALUES (?,?,?,?)",[(pid,groups[pid],a,bl.get(pid)) for pid,a in people.items()])
        conn.executemany("INSERT INTO readiness VALUES (?,?,?)",[(pid,h,status) for (pid,h),status in ready.items()])
        conn.executemany("INSERT INTO targets VALUES (?,?,?,?,?,?,?,?)",pairs)
        conn.commit()
    finally:
        conn.close()
    reports.mkdir(parents=True,exist_ok=True)
    counts=Counter((groups[pid],h,status) for (pid,h),status in ready.items())
    save(reports/"readiness_by_split.csv",["split","horizon","status","patients"],
         [dict(split=s,horizon=h,status=k,patients=safe(n,10)) for (s,h,k),n in sorted(counts.items())])
    pair_counts=Counter((groups[pid],h) for pid,h,*_ in pairs)
    pop=Counter(groups.values())
    save(reports/"pairs_by_split.csv",["split","total","ready_6m","ready_12m","ready_24m"],
         [dict(split=s,total=pop[s],ready_6m=safe(pair_counts[s,6],10),ready_12m=safe(pair_counts[s,12],10),ready_24m=safe(pair_counts[s,24],10)) for s in ("train","val","test")])
    q=[]
    for s in ("train","val","test"):
        for available in (True,False):
            values=[v for pid,v in bl.items() if groups[pid]==s and (ready[pid,12]=="paired")==available]
            q.append(dict(split=s,ready_12m=available,n_baseline_motor=safe(len(values),10),mean_baseline_score=round(mean(values),2) if len(values)>=10 else ""))
    save(reports/"followup_selection_bias.csv",["split","ready_12m","n_baseline_motor","mean_baseline_score"],q)
    save(reports/"quality_flags.csv",["flag","count"],[dict(flag=k,count=safe(v,10)) for k,v in sorted(qc.items())])
    summary=dict(status="local_private_manifest_not_training_validated",baseline_candidates=len(people),
       ready_12m=sum(h==12 for _,h,*_ in pairs),population_splits=dict(pop),seed=seed,
       primary_target="BL to 12+/-3 months, UNSPECIFIED state NP3TOT",
       warnings=["No clinical validation of missing drug-state equivalence",
       "Future target and availability must not be baseline model features",
       "Patient splits assigned before filtering by future availability",
       "Never upload SQLite file; only aggregate report directory"],
       generated_utc=datetime.now(timezone.utc).isoformat())
    (reports/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261018)
    a=p.parse_args()
    print(json.dumps(build(a.data_root,a.manifest_db,a.report_dir,a.seed),indent=2))
if __name__=="__main__":
    main()
