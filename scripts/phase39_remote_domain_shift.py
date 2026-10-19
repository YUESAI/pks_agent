#!/usr/bin/env python3
"""Phase39: audit observable-feature transport from V04 masking to real R01.

TRAIN-only read-only PPMI Prodromal audit. Reuse Phase36's strict same-visit
V04 eligibility and R01 remote six-unrateable + 27-valid eligibility.
Compare observable-item distributions, matched-patient R01-to-later-V04
changes, and transfer of V04-derived visible-total rank thresholds.

This CANNOT measure performance on actual R01 missing-six targets: those
labels DO NOT EXIST. A later V04 exam is NOT R01 same-day ground truth.
No patient IDs, individual scores, dates, or decision lists are exported.
"""
import argparse,csv,json,math,random,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean,median

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase36_masked_site_feasibility import (
    VISIBLE,HIDDEN,parse_rows,unique_by_patient
)
from scripts.phase31_motor_repetitions import train_anchors
from scripts.phase22_strict_motor_pairs import one_file,safe,save

BUDGETS=(.10,.20,.30,.40,.50)
SAMPLES=("remote_R01","onsite_V04","paired_R01","paired_V04")


def quantile(data,q):
    if not data:raise ValueError("Empty percentile")
    values=sorted(data)
    pos=(len(values)-1)*q
    lower=int(pos);upper=min(lower+1,len(values)-1)
    return values[lower]*(upper-pos)+values[upper]*(pos-lower)


def statistics(values):
    if not values:return {"n":0}
    m=mean(values)
    sd=math.sqrt(mean((x-m)**2 for x in values))
    return {"n":len(values),"mean":m,"median":median(values),
            "q25":quantile(values,.25),"q75":quantile(values,.75),
            "sd":sd}


def ks_distance(a,b):
    """Descriptive empirical two-sample KS distance, no p-value."""
    if not a or not b:return None
    aa=sorted(a);bb=sorted(b)
    ia=ib=0;result=0.
    for value in sorted(set(aa)|set(bb)):
        while ia<len(aa) and aa[ia]<=value:ia+=1
        while ib<len(bb) and bb[ib]<=value:ib+=1
        result=max(result,abs(ia/len(aa)-ib/len(bb)))
    return result


def standardized_difference(a,b):
    """SMD = (remote mean - onsite mean) / pooled population SD."""
    if not a or not b:return None
    m1,m2=mean(a),mean(b)
    v1=mean((x-m1)**2 for x in a)
    v2=mean((x-m2)**2 for x in b)
    denominator=math.sqrt((v1+v2)/2)
    return (m1-m2)/denominator if denominator>0 else None


def load_populations(root,manifest):
    anchors=train_anchors(manifest)
    source=one_file(root/"PPMI","MDS-UPDRS_Part_III_")
    qc=Counter()
    selected=unique_by_patient(parse_rows(source,anchors,qc),qc)
    remote={};onsite={}
    for (pid,event),(month,fields,total) in selected.items():
        if any(fields[name][0]!="valid" for name in VISIBLE):
            qc[event+"_27_visible_incomplete"]+=1
            continue
        values={name:fields[name][1] for name in VISIBLE}
        visible_total=sum(values.values())
        if event=="R01":
            if total is None and all(fields[name][0]=="unable" for name in HIDDEN):
                remote[pid]={"month":month,"values":values,"sum":visible_total}
            else:qc["R01_other_total_or_six_status"]+=1
        elif event=="V04":
            if (total is not None and
                all(fields[name][0]=="valid" for name in HIDDEN) and
                sum(fields[name][1] for name in HIDDEN)+visible_total==total):
                onsite[pid]={"month":month,"values":values,"sum":visible_total}
            else:qc["V04_invalid_full_total_or_hidden_items"]+=1
    common={}
    for pid in remote.keys() & onsite.keys():
        r=remote[pid];v=onsite[pid]
        if v["month"]<=r["month"]:
            qc["paired_V04_not_later_than_R01"]+=1
            continue
        common[pid]=(r,v)
    return remote,onsite,common,qc,source,len(anchors)


def threshold_benchmark(onsite,remote,common,budget):
    """Transport V04 score cutoff to R01 WITHOUT hidden R01 ground truth.

    Include all ties at the V04 cutoff. Thus the realized site proportion
    need not equal nominal budget; this is explicitly recorded.
    """
    site_scores=sorted((r["sum"] for r in onsite.values()),reverse=True)
    count=max(1,int(math.floor(len(site_scores)*budget+.5)))
    cutoff=site_scores[count-1]
    selected_site={pid for pid,r in onsite.items() if r["sum"]>=cutoff}
    selected_remote={pid for pid,r in remote.items() if r["sum"]>=cutoff}
    changed=sum((r["sum"]>=cutoff)!=(v["sum"]>=cutoff)
                for pid,(r,v) in common.items())
    return {
        "nominal_budget":budget,
        "site_selected_n":len(selected_site),
        "site_selected_rate":len(selected_site)/len(onsite),
        "remote_selected_n":len(selected_remote),
        "remote_selected_rate":len(selected_remote)/len(remote),
        "paired_remote_selected_n":sum(r["sum"]>=cutoff for r,v in common.values()),
        "paired_onsite_selected_n":sum(v["sum"]>=cutoff for r,v in common.values()),
        "paired_selection_switch_n":changed,
    }


def bootstrap_paired_delta(common,reps,seed):
    if len(common)<10 or reps<100:raise ValueError("Insufficient paired audit samples")
    values=[v["sum"]-r["sum"] for r,v in common.values()]
    rng=random.Random(seed)
    draws=[]
    for i in range(reps):
        draws.append(mean(values[rng.randrange(len(values))]
                          for _ in values))
    return mean(values),quantile(draws,.025),quantile(draws,.975)


def audit(root,manifest,out,bootstraps=500,minimum=10,seed=20261018):
    root,manifest,out=[Path(p).expanduser().resolve()
                       for p in (root,manifest,out)]
    if not (root/"PPMI").is_dir() or not manifest.is_file():
        raise FileNotFoundError("Missing PPMI source or frozen private manifest")
    if out==root or root in out.parents or out==manifest or minimum<10:
        raise ValueError("Unsafe output directory or min-cell-count")
    remote,site,paired,qc,source,train_total=load_populations(root,manifest)
    if min(len(remote),len(site),len(paired))<minimum:
        raise ValueError("Not enough eligible TRAIN patients for domain audit")
    population={
        "remote_R01":list(remote.values()),
        "onsite_V04":list(site.values()),
        "paired_R01":[r for r,v in paired.values()],
        "paired_V04":[v for r,v in paired.values()],
    }
    counts=[{"source":name,"patients":len(values)}
            for name,values in population.items()]
    observed=[]
    for item in ("VISIBLE_27_TOTAL",*VISIBLE):
        for name in SAMPLES:
            v=([x["sum"] for x in population[name]] if item=="VISIBLE_27_TOTAL"
               else [x["values"][item] for x in population[name]])
            st=statistics(v)
            observed.append({
                "feature":item,"sample":name,"n":st["n"],
                "mean":round(st["mean"],3),
                "median":round(st["median"],3),
                "nonzero_pct":round(100*sum(x>0 for x in v)/len(v),2),
            })
    differences=[]
    for item in ("VISIBLE_27_TOTAL",*VISIBLE):
        for comparison,a_name,b_name in (
            ("all_R01_minus_V04","remote_R01","onsite_V04"),
            ("matched_R01_minus_V04","paired_R01","paired_V04"),
        ):
            a=[r["sum"] if item=="VISIBLE_27_TOTAL" else r["values"][item]
               for r in population[a_name]]
            b=[r["sum"] if item=="VISIBLE_27_TOTAL" else r["values"][item]
               for r in population[b_name]]
            smd=standardized_difference(a,b)
            differences.append({
                "feature":item,"comparison":comparison,
                "mean_R01_minus_V04":round(mean(a)-mean(b),3),
                "standardized_mean_difference":round(smd,3) if smd is not None else "",
                "empirical_KS_distance":round(ks_distance(a,b),3),
            })
    matched_changes=[v["sum"]-r["sum"] for r,v in paired.values()]
    gap=[v["month"]-r["month"] for r,v in paired.values()]
    change=statistics(matched_changes)
    point,lo,hi=bootstrap_paired_delta(paired,bootstraps,seed)
    paired_report={
        "paired_patients":len(paired),
        "mean_onsite_minus_remote_visible_total":round(point,4),
        "bootstrap_mean_lower_95":round(lo,4),
        "bootstrap_mean_upper_95":round(hi,4),
        "median_delta":round(change["median"],3),
        "q25_delta":round(change["q25"],3),
        "q75_delta":round(change["q75"],3),
        "onsite_higher":safe(sum(x>0 for x in matched_changes),minimum),
        "same_score":safe(sum(x==0 for x in matched_changes),minimum),
        "onsite_lower":safe(sum(x<0 for x in matched_changes),minimum),
        "median_month_gap":round(median(gap),2),
    }
    threshold_rows=[]
    for budget in BUDGETS:
        result=threshold_benchmark(site,remote,paired,budget)
        threshold_rows.append({
            "nominal_site_budget":budget,
            "realized_site_selected":safe(result["site_selected_n"],minimum),
            "realized_site_selected_pct":round(100*result["site_selected_rate"],2),
            "same_cutoff_R01_selected":safe(result["remote_selected_n"],minimum),
            "same_cutoff_R01_selected_pct":round(100*result["remote_selected_rate"],2),
            "paired_R01_selected":safe(result["paired_remote_selected_n"],minimum),
            "paired_V04_selected":safe(result["paired_onsite_selected_n"],minimum),
            "paired_patient_selection_changed":safe(result["paired_selection_switch_n"],minimum),
        })
    out.mkdir(parents=True,exist_ok=True)
    save(out/"eligible_domain_counts.csv",
         ["source","patients"],counts)
    save(out/"visible_item_distributions.csv",
         ["feature","sample","n","mean","median","nonzero_pct"],observed)
    save(out/"visible_item_shift.csv",
         ["feature","comparison","mean_R01_minus_V04",
          "standardized_mean_difference","empirical_KS_distance"],differences)
    save(out/"matched_patient_visible_change.csv",
         list(paired_report),[paired_report])
    save(out/"visible_sum_threshold_transport.csv",
         ["nominal_site_budget","realized_site_selected",
          "realized_site_selected_pct","same_cutoff_R01_selected",
          "same_cutoff_R01_selected_pct","paired_R01_selected",
          "paired_V04_selected","paired_patient_selection_changed"],threshold_rows)
    save(out/"quality_flags.csv",
         ["check","TRAIN_records_or_patients"],
         [{"check":name,"TRAIN_records_or_patients":safe(n,minimum)}
          for name,n in sorted(qc.items())])
    summary={
        "status":"TRAIN_only_real_R01_to_masked_V04_visible_feature_shift_audit",
        "train_registry":train_total,
        "remote_R01_eligible":len(remote),
        "onsite_V04_eligible":len(site),
        "same_patient_R01_then_V04":len(paired),
        "feature_count":len(VISIBLE),
        "bootstrap_replicates":bootstraps,"seed":seed,
        "threshold_policy":"V04 visible-total top-budget cutoff, ALL ties included; transfer unchanged to R01",
        "interpretation_limits":[
            "No hidden six scores are observed at R01, so R01 policy utility cannot be evaluated.",
            "V04 is months AFTER R01; paired changes include actual clinical and assessment change.",
            "Both domain populations can contain the same patients; sample comparisons are descriptive.",
            "Site cutoff uses site TRAIN observable scores only and has tie inflation.",
            "Domain shift may reflect visit modality, time, cohort composition or clinical change.",
            "No patient-specific decision or score, raw month, or ID exported.",
            "Only frozen TRAIN patients; PPMI validation/test remain unopened.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (out/"summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)+chr(10))
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,default=Path("outputs/phase39_domain_shift"))
    p.add_argument("--bootstraps",type=int,default=500)
    p.add_argument("--seed",type=int,default=20261018)
    p.add_argument("--min-cell-count",type=int,default=10)
    a=p.parse_args()
    print(json.dumps(audit(a.data_root,a.manifest_db,a.report_dir,
                           a.bootstraps,a.min_cell_count,a.seed),
                     indent=2,ensure_ascii=False))
    print("Upload aggregate Phase39 reports only. No patient-level data.")
if __name__=="__main__":main()
