#!/usr/bin/env python3
"""Phase43 locked VAL-only remote-to-future onsite risk check; TEST unopened."""
import argparse,csv,hashlib,json,math,sqlite3,sys
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase36_masked_site_feasibility import VISIBLE,HIDDEN,parse_rows,unique_by_patient
from scripts.phase22_strict_motor_pairs import one_file
VERSION="phase43-val-v1-frozen-20261018"
SEED=20261018
MIN_CELL=10
BUDGETS=(.20,.10,.30,.40,.50)

def validation_anchors(manifest):
    with sqlite3.connect("file:"+str(Path(manifest).resolve())+"?mode=ro",uri=True) as db:
        rows=db.execute("SELECT pid,baseline_month FROM patients WHERE split='val'").fetchall()
    if not rows or len({pid for pid,m in rows})!=len(rows):
        raise ValueError("Invalid frozen VAL patient registry")
    return dict(rows)

def load_val(root,manifest):
    anchors=validation_anchors(manifest)
    source=one_file(root/"PPMI","MDS-UPDRS_Part_III_")
    quality=Counter()
    selected=unique_by_patient(parse_rows(source,anchors,quality),quality)
    remote={};site={}
    for (pid,event),(month,fields,total) in selected.items():
        if any(fields[k][0]!="valid" for k in VISIBLE):
            quality[event+"_visible_incomplete"]+=1
            continue
        visible=sum(fields[k][1] for k in VISIBLE)
        if event=="R01":
            if total is None and all(fields[k][0]=="unable" for k in HIDDEN):
                remote[pid]=(month,float(visible))
            else:quality["R01_not_six_unrateable"]+=1
        elif event=="V04":
            if total is not None and all(fields[k][0]=="valid" for k in HIDDEN):
                hidden=sum(fields[k][1] for k in HIDDEN)
                if hidden+visible==total:site[pid]=(month,hidden)
                else:quality["V04_inconsistent_total"]+=1
            else:quality["V04_incomplete"]+=1
    registry={}
    for pid,(month,visible) in remote.items():
        label=None
        if pid in site:
            later,hidden=site[pid]
            if later>month:label=bool(hidden>0)
            else:quality["V04_not_after_R01"]+=1
        registry[pid]=(visible,label)
    return registry,quality,len(anchors)

def count_budget(n,fraction):
    return min(n,int(math.floor(n*fraction+.5)))

def policy_hash(pid):
    return hashlib.sha256(("Phase41|"+str(SEED)+"|"+pid).encode()).hexdigest()

def select_ids(registry,budget):
    order=sorted(registry,key=lambda pid:(-registry[pid][0],policy_hash(pid)))
    return set(order[:count_budget(len(order),budget)])

def bounds(registry,chosen):
    n=len(registry);k=len(chosen)
    if not n or not k or not chosen.issubset(registry):
        raise ValueError("Insufficient VAL cases or invalid selection")
    positives=sum(label is True for score,label in registry.values())
    negatives=sum(label is False for score,label in registry.values())
    unknown=n-positives-negatives
    a=sum(registry[pid][1] is True for pid in chosen)
    b=sum(registry[pid][1] is False for pid in chosen)
    u=k-a-b;v=unknown-u
    if positives<1 or min(a,b,u,v)<0:
        raise ValueError("No observed VAL positives or inconsistent labels")
    return {"n":n,"k":k,"known_positive":positives,"known_negative":negatives,
            "unknown":unknown,"selected_positive":a,"selected_unknown":u,
            "precision_lower":a/k,"precision_upper":(a+u)/k,
            "recall_lower":a/(positives+v),
            "recall_upper":(a+u)/(positives+u),
            "random_expectation":k/n,
            "lower_lift":a/(positives+v)-k/n}

def auc_ap(registry):
    rows=[(score,label) for score,label in registry.values() if label is not None]
    pos=[score for score,label in rows if label]
    neg=[score for score,label in rows if not label]
    if len(pos)<MIN_CELL or len(neg)<MIN_CELL:return None,None
    auc=sum(1 if p>n else .5 if p==n else 0 for p in pos for n in neg)/(len(pos)*len(neg))
    ordered=sorted(rows,key=lambda pair:-pair[0])
    j=0;found=0;tp=0;ap=0
    while j<len(ordered):
        k=j+1
        while k<len(ordered) and ordered[k][0]==ordered[j][0]:k+=1
        extra=sum(label for _,label in ordered[j:k])
        tp+=extra;found=k
        ap+=(extra/len(pos))*(tp/found)
        j=k
    return auc,ap

def protected(n):
    return str(n) if n==0 or n>=MIN_CELL else "<10"

def save(path,fields,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader();w.writerows(rows)

def summarize(registry,quality,cohort_n,out):
    if len(registry)<MIN_CELL:raise ValueError("Too few eligible VAL R01 cases")
    primary=bounds(registry,select_ids(registry,.20))
    auc,ap=auc_ap(registry)
    rows=[]
    for budget in BUDGETS:
        x=bounds(registry,select_ids(registry,budget))
        rows.append({
            "budget_pct":int(round(100*budget)),
            "exact_topk":protected(x["k"]),
            "selected_known_future_positive":protected(x["selected_positive"]),
            "selected_without_future_label":protected(x["selected_unknown"]),
            "precision_lower_pct":round(100*x["precision_lower"],2),
            "precision_upper_pct":round(100*x["precision_upper"],2),
            "recall_lower_pct":round(100*x["recall_lower"],2),
            "recall_upper_pct":round(100*x["recall_upper"],2),
            "random_expected_recall_pct":round(100*x["random_expectation"],2),
            "worst_case_recall_lift_pp":round(100*x["lower_lift"],2),
            "precommitted_primary":budget==.20})
    out.mkdir(parents=True,exist_ok=True)
    save(out/"locked_validation_policy.csv",list(rows[0]),rows)
    save(out/"quality_flags.csv",["issue","VAL_records_or_patients"],
         [{"issue":key,"VAL_records_or_patients":protected(n)}
          for key,n in sorted(quality.items())])
    summary={
        "protocol_version":VERSION,"status":"LOCKED_VALIDATION_ONLY",
        "split":"val","TEST_unopened":True,
        "frozen_val_registry":cohort_n,"eligible_val_R01":len(registry),
        "eligible_future_V04":len(registry)-primary["unknown"],
        "missing_future_V04":protected(primary["unknown"]),
        "known_future_six_positive":protected(primary["known_positive"]),
        "precommitted_budget_pct":20,"tie_seed":SEED,
        "precommitted_statistic":"sharp recall lower bound minus random exact-k expected recall",
        "primary_lift_pp":round(100*primary["lower_lift"],2),
        "primary_lift_is_positive":primary["lower_lift"]>0,
        "labeled_subset_AUROC":round(auc,5) if auc is not None else None,
        "labeled_subset_AP":round(ap,5) if ap is not None else None,
        "no_model_training":True,
        "cautions":[
            "Validation policy fixed before examining VAL outcomes; do not retune.",
            "Only split=val patients read; TEST remains unopened.",
            "Future V04 score is months later, NOT R01 same-day hidden truth.",
            "Worst-case identification bounds are NOT confidence intervals.",
            "A positive lower-bound lift is NOT a hypothesis significance test.",
            "Referral cost, causal clinical benefit and patient safety unmeasured.",
            "Only aggregated numbers released; small counts suppressed.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat()}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+chr(10))
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,default=Path("outputs/phase43_locked_validation"))
    a=p.parse_args()
    root,db,out=(Path(x).expanduser().resolve()
                 for x in (a.data_root,a.manifest_db,a.report_dir))
    if not (root/"PPMI").is_dir() or not db.is_file() or out==db or out==root or root in out.parents:
        raise ValueError("Missing inputs or unsafe output")
    registry,qc,n=load_val(root,db)
    print(json.dumps(summarize(registry,qc,n,out),indent=2))
    print("Do not tune after VAL; TEST remains unopened.")
if __name__=="__main__":main()
