#!/usr/bin/env python3
"""Phase41: observational R01 to later V04 future hidden-six risk ranking.

Only frozen PPMI Prodromal TRAIN participants. At 6-month R01, 27 Motor
items are visible while six items are unrateable (101). Target: whether
those SIX items at the patient's LATER V04 in-person examination are >0.
This is a future risk-stratification proxy, NOT true R01 hidden scores,
contemporaneous value-of-information, referral effectiveness, or a
causal policy evaluation.

Compare R01 observable sum with random exact-quota ranking. Later V04
observable sum and perfect future labels are marked RETROSPECTIVE references
and must NEVER be advertised as R01-available policy inputs. Include
patient-level repeated-random and bootstrap uncertainty, and tie-break
sensitivity. No individual clinical data, IDs or decisions are exported.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase36_masked_site_feasibility import (
    VISIBLE,HIDDEN,parse_rows,unique_by_patient
)
from scripts.phase31_motor_repetitions import train_anchors
from scripts.phase22_strict_motor_pairs import one_file,safe,save

BUDGETS=(.10,.20,.30,.40,.50)
BIN_LABELS=("0","1_3","4_7","8plus")


def score_bin(value):
    return "0" if value==0 else "1_3" if value<=3 else "4_7" if value<=7 else "8plus"


def quantile(values,frac):
    if not values:raise ValueError("Empty values")
    values=sorted(values)
    pos=(len(values)-1)*frac
    lo=int(pos);hi=min(lo+1,len(values)-1)
    return values[lo]*(hi-pos)+values[hi]*(pos-lo)


def tie_hash(pid,seed):
    return hashlib.sha256(
        (f"Phase41|{seed}|{pid}").encode("utf-8")).hexdigest()


def quota(n,budget):
    return max(0,min(n,int(math.floor(n*budget+.5))))


def rank_rows(rows,field,seed):
    return sorted(rows,key=lambda r:(-r[field],tie_hash(r["pid"],seed)))


def rank_select(rows,field,budget,seed):
    return rank_rows(rows,field,seed)[:quota(len(rows),budget)]


def roc_auc(rows,field):
    """Tie-aware AUROC for future hidden-six positive classification."""
    ordered=sorted(rows,key=lambda r:r[field])
    pos=sum(r["future_six"]>0 for r in rows)
    neg=len(rows)-pos
    if not pos or not neg:return None
    total=0.;j=0
    while j<len(ordered):
        end=j+1
        while end<len(ordered) and ordered[end][field]==ordered[j][field]:
            end+=1
        rank=((j+1)+end)/2
        total+=rank*sum(ordered[k]["future_six"]>0 for k in range(j,end))
        j=end
    return (total-pos*(pos+1)/2)/(pos*neg)


def average_precision(rows,field):
    """Tie-aware threshold AP; ties enter together, no arbitrary within-tie order."""
    ordered=sorted(rows,key=lambda r:r[field],reverse=True)
    npos=sum(r["future_six"]>0 for r in rows)
    if not npos:return None
    tp=0;found=0;result=0.;j=0
    while j<len(ordered):
        end=j+1
        while end<len(ordered) and ordered[end][field]==ordered[j][field]:
            end+=1
        group_pos=sum(ordered[k]["future_six"]>0 for k in range(j,end))
        tp+=group_pos
        found=end
        result+=(group_pos/npos)*(tp/found)
        j=end
    return result


def load_train_pairs(root,manifest):
    anchors=train_anchors(manifest)
    source=one_file(root/"PPMI","MDS-UPDRS_Part_III_")
    qc=Counter()
    selected=unique_by_patient(parse_rows(source,anchors,qc),qc)
    remote={};onsite={}
    for (pid,event),(month,fields,total) in selected.items():
        if any(fields[name][0]!="valid" for name in VISIBLE):
            qc[event+"_visible_incomplete"]+=1
            continue
        visible=sum(fields[name][1] for name in VISIBLE)
        if event=="R01":
            if total is None and all(fields[name][0]=="unable" for name in HIDDEN):
                remote[pid]=(month,visible)
            else:qc["R01_not_six_unrateable"]+=1
        elif event=="V04":
            if (total is not None and
                all(fields[name][0]=="valid" for name in HIDDEN) and
                visible+sum(fields[name][1] for name in HIDDEN)==total):
                onsite[pid]=(month,visible,sum(fields[name][1] for name in HIDDEN))
            else:qc["V04_not_complete_same_visit"]+=1
    paired=[]
    for pid in sorted(remote.keys()&onsite.keys()):
        rmonth,rscore=remote[pid]
        vmonth,vscore,hidden=onsite[pid]
        if vmonth<=rmonth:
            qc["V04_not_after_R01"]+=1
            continue
        paired.append({
            "pid":pid,"remote_sum":rscore,"future_visible_sum":vscore,
            "future_six":hidden,"gap":vmonth-rmonth,
        })
    return paired,remote,onsite,qc,source,len(anchors)


def selection_stats(selected,all_rows):
    positives=sum(r["future_six"]>0 for r in all_rows)
    burden=sum(r["future_six"] for r in all_rows)
    found=sum(r["future_six"]>0 for r in selected)
    mass=sum(r["future_six"] for r in selected)
    return {
        "selected_n":len(selected),
        "captured_positives":found,
        "positive_recall":found/positives if positives else None,
        "captured_burden":mass,
        "burden_fraction":mass/burden if burden else None,
    }


def bootstrap_excess_recall(rows,field,budget,reps,seed):
    """Re-rank every patient-bootstrap sample; compare to random expectation.

    The estimand is future-positive recall minus selected-fraction, not
    same-visit clinical benefit. Bootstrap samples duplicate subjects,
    treated as independent sampled patient slots (not unique IDs).
    """
    rng=random.Random(seed)
    diffs=[]
    for b in range(reps):
        sample=[]
        for j in range(len(rows)):
            row=rows[rng.randrange(len(rows))]
            sample.append(dict(row,pid=f"{row['pid']}|replicate_slot_{j}"))
        if not any(r["future_six"]>0 for r in sample):
            continue
        chosen=rank_select(sample,field,budget,seed+b)
        result=selection_stats(chosen,sample)
        diffs.append(result["positive_recall"]-len(chosen)/len(sample))
    if len(diffs)<int(.9*reps):
        raise ValueError("Insufficient valid bootstrap samples")
    return quantile(diffs,.025),quantile(diffs,.975)


def run(root,manifest,out,seed=20261018,bootstraps=500,
        random_repeats=500,tie_repeats=100,minimum=10):
    root,manifest,out=(Path(x).expanduser().resolve()
                       for x in (root,manifest,out))
    if not (root/"PPMI").is_dir() or not manifest.is_file():
        raise FileNotFoundError("Missing PPMI source or frozen manifest")
    if root==out or root in out.parents or out==manifest:
        raise ValueError("Unsafe report location")
    if min(bootstraps,random_repeats)<100 or tie_repeats<20 or minimum<10:
        raise ValueError("Insufficient repeats or privacy threshold")
    rows,remote,site,qc,source,train_registry=load_train_pairs(root,manifest)
    if len(rows)<minimum:raise ValueError("Too few paired TRAIN patients")
    pos=sum(r["future_six"]>0 for r in rows)
    if min(pos,len(rows)-pos)<minimum:
        raise ValueError("Insufficient positive/negative future labels")
    burden=sum(r["future_six"] for r in rows)
    # Avoid a target from R01; V04 target always months after the decision.
    future_score={r["pid"]:r["future_six"] for r in rows}
    outcome_rows=[
        {"category":name,"patients":safe(n,minimum)}
        for name,n in sorted(Counter(
            "zero" if r["future_six"]==0 else
            "1_2" if r["future_six"]<=2 else
            "3_5" if r["future_six"]<=5 else "6plus"
            for r in rows).items())
    ]
    strata=[]
    for group in BIN_LABELS:
        part=[r for r in rows if score_bin(r["remote_sum"])==group]
        found=sum(r["future_six"]>0 for r in part)
        strata.append({
            "R01_visible_sum_bin":group,
            "patients":safe(len(part),minimum),
            "later_V04_hidden_positive":safe(found,minimum),
            "later_V04_positive_rate":round(found/len(part),4)
                if len(part)>=minimum and
                   (found==0 or found>=minimum) and
                   (len(part)-found==0 or len(part)-found>=minimum) else "",
        })
    discriminations=[]
    for field,label in (
        ("remote_sum","R01_visible_sum_DEPLOYABLE_INPUT"),
        ("future_visible_sum","V04_visible_sum_RETROSPECTIVE_ONLY"),
    ):
        auc=roc_auc(rows,field);ap=average_precision(rows,field)
        discriminations.append({
            "ranking":label,"patients":len(rows),
            "nonzero_prevalence":round(pos/len(rows),5),
            "AUROC":round(auc,5) if auc is not None else "",
            "average_precision":round(ap,5) if ap is not None else "",
        })
    budget_rows=[]
    tie_rows=[]
    for budget in BUDGETS:
        actual=rank_select(rows,"remote_sum",budget,seed)
        future=rank_select(rows,"future_visible_sum",budget,seed)
        oracle=rank_select(rows,"future_six",budget,seed)
        # Repeated random selection quantifies the variation due to
        # the random baseline's draw, unlike a single arbitrary seed.
        sampled=[]
        for j in range(random_repeats):
            rng=random.Random(seed+100000*int(100*budget)+j)
            pick=rng.sample(rows,quota(len(rows),budget))
            sampled.append(selection_stats(pick,rows))
        tie_results=[
            selection_stats(rank_select(rows,"remote_sum",budget,seed+j),rows)
            for j in range(tie_repeats)
        ]
        lo,hi=bootstrap_excess_recall(
            rows,"remote_sum",budget,bootstraps,
            seed+int(1000*budget))
        for name,stat,reference in (
            ("R01_visible_sum",selection_stats(actual,rows),False),
            ("random_repeated_mean",{
                "selected_n":quota(len(rows),budget),
                "captured_positives":mean(x["captured_positives"] for x in sampled),
                "positive_recall":mean(x["positive_recall"] for x in sampled),
                "captured_burden":mean(x["captured_burden"] for x in sampled),
                "burden_fraction":mean(x["burden_fraction"] for x in sampled)},False),
            ("V04_visible_sum_FUTURE_REFERENCE",selection_stats(future,rows),True),
            ("V04_hidden_six_ORACLE",selection_stats(oracle,rows),True),
        ):
            budget_rows.append({
                "nominal_budget_pct":int(budget*100),
                "ranking":name,"selected_n":stat["selected_n"],
                "captured_future_positive":(round(stat["captured_positives"],3) if stat["captured_positives"]>=minimum or stat["captured_positives"]==0 else f"<{minimum}"),
                "positive_recall":round(stat["positive_recall"],5),
                "future_hidden_burden_captured_pct":round(
                    100*stat["burden_fraction"],3),
                "not_available_at_R01":reference,
            })
        tie_rows.append({
            "budget_pct":int(budget*100),
            "R01_score_tie_positive_recall_low":round(
                min(x["positive_recall"] for x in tie_results),4),
            "R01_score_tie_positive_recall_high":round(
                max(x["positive_recall"] for x in tie_results),4),
            "random_positive_recall_q025":round(
                quantile([x["positive_recall"] for x in sampled],.025),4),
            "random_positive_recall_q975":round(
                quantile([x["positive_recall"] for x in sampled],.975),4),
            "R01_excess_recall_vs_random_expectation":round(
                selection_stats(actual,rows)["positive_recall"]-
                quota(len(rows),budget)/len(rows),4),
            "bootstrap_excess_recall_low":round(lo,4),
            "bootstrap_excess_recall_high":round(hi,4),
        })
    missing_site=[{"followup_V04_complete":bool(value),
                   "patients":safe(sum((pid in site)==value for pid in remote),minimum),
                   "mean_R01_visible_total":round(
                       mean(score for pid,(month,score) in remote.items()
                            if (pid in site)==value),3)
                   if sum((pid in site)==value for pid in remote)>=minimum else ""}
                  for value in (True,False)]
    out.mkdir(parents=True,exist_ok=True)
    save(out/"future_V04_hidden_outcomes.csv",
         ["category","patients"],outcome_rows)
    save(out/"R01_score_vs_future_hidden_prevalence.csv",
         ["R01_visible_sum_bin","patients","later_V04_hidden_positive",
          "later_V04_positive_rate"],strata)
    save(out/"prospective_rank_discrimination.csv",
         ["ranking","patients","nonzero_prevalence","AUROC",
          "average_precision"],discriminations)
    save(out/"future_outcome_budget_curves.csv",
         ["nominal_budget_pct","ranking","selected_n",
          "captured_future_positive","positive_recall",
          "future_hidden_burden_captured_pct",
          "not_available_at_R01"],budget_rows)
    save(out/"random_and_tie_sensitivity.csv",
         ["budget_pct","R01_score_tie_positive_recall_low",
          "R01_score_tie_positive_recall_high",
          "random_positive_recall_q025","random_positive_recall_q975",
          "R01_excess_recall_vs_random_expectation",
          "bootstrap_excess_recall_low","bootstrap_excess_recall_high"],tie_rows)
    save(out/"followup_selection_summary.csv",
         ["followup_V04_complete","patients","mean_R01_visible_total"],missing_site)
    save(out/"source_quality_flags.csv",
         ["check","records_or_patients"],[
             {"check":key,"records_or_patients":safe(value,minimum)}
             for key,value in sorted(qc.items())])
    result={
        "status":"TRAIN_only_observational_R01_to_future_V04_hidden_six_prognostic_proxy",
        "TRAIN_registry":train_registry,
        "R01_eligible_remote_people":len(remote),
        "paired_R01_then_V04_people":len(rows),
        "V04_future_hidden_six_positive_people":pos,
        "V04_future_hidden_total_burden":safe(burden,minimum),
        "mean_R01_to_V04_interval_months":round(mean(r["gap"] for r in rows),3),
        "budgets_percent":list(int(x*100) for x in BUDGETS),
        "random_repeats":random_repeats,"tie_repeats":tie_repeats,
        "patient_bootstrap_repeats":bootstraps,"seed":seed,
        "caveats":[
            "V04 six-item labels are LATER than R01; not true missing R01 labels.",
            "Future visible V04 and oracle rankings are references, never R01 inputs.",
            "Ranking is observational prediction, not referral treatment effect or VOI.",
            "Only patients observed at BOTH R01 and V04 have this outcome.",
            "No causal claim, no validated clinical examination cost model.",
            "Random repeats/tie sensitivity and conditional patient bootstrap differ.",
            "Only TRAIN clinical records processed. No val/test or per-person outputs.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (out/"summary.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False)+chr(10))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--manifest-db",type=Path,required=True)
    parser.add_argument("--report-dir",type=Path,
                        default=Path("outputs/phase41_future_risk"))
    parser.add_argument("--seed",type=int,default=20261018)
    parser.add_argument("--bootstraps",type=int,default=500)
    parser.add_argument("--random-repeats",type=int,default=500)
    parser.add_argument("--tie-repeats",type=int,default=100)
    parser.add_argument("--min-cell-count",type=int,default=10)
    args=parser.parse_args()
    print(json.dumps(run(args.data_root,args.manifest_db,args.report_dir,
        args.seed,args.bootstraps,args.random_repeats,
        args.tie_repeats,args.min_cell_count),indent=2,ensure_ascii=False))
    print("Upload only aggregated Phase41 reports; private patient data stays local.")


if __name__=="__main__":
    main()
