#!/usr/bin/env python3
"""Phase40: exact-budget, tie-aware policy calibration and rank stability.

PPMI Prodromal TRAIN only. Reuses strict Phase39 eligibility for real R01
partial remote exams and complete onsite V04 exams. No real R01 six-hidden
ground truth exists, so this is a selection-feasibility / transport audit,
NOT an observed value-of-information or clinical referral effectiveness trial.

Top-k selection in each domain uses ONLY 27 observable item total scores.
Deterministic seed-based hashes break exact-score ties. All counters and
rank analyses are aggregated; no private IDs/scores/dates/choices exported.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,sys
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase39_remote_domain_shift import (
    load_populations,threshold_benchmark,quantile
)
from scripts.phase22_strict_motor_pairs import safe,save

BUDGETS=(.10,.20,.30,.40,.50)


def quota(n,budget):
    return max(0,min(n,int(math.floor(n*budget+.5))))


def tie_hash(pid,seed):
    return hashlib.sha256(
        (f"phase40|{seed}|{pid}").encode("utf-8")).hexdigest()


def ranked_ids(population,seed):
    return sorted(population,key=lambda pid:(
        -population[pid]["sum"],tie_hash(pid,seed)))


def exact_select(population,budget,seed):
    rank=ranked_ids(population,seed)
    selected=set(rank[:quota(len(rank),budget)])
    if len(selected)!=quota(len(population),budget):
        raise ValueError("Exact-budget selection count mismatch")
    return selected


def boundary_ties(population,budget):
    rank=sorted((r["sum"] for r in population.values()),reverse=True)
    k=quota(len(rank),budget)
    if not rank or k==0 or k==len(rank):
        return (0,0,None)
    cutoff=rank[k-1]
    higher=sum(score>cutoff for score in rank)
    equal=sum(score==cutoff for score in rank)
    return higher,equal,cutoff


def midranks(values):
    """Midranks for tied clinical scores (ascending 1-based)."""
    ordered=sorted(range(len(values)),key=lambda i:values[i])
    ranks=[0.]*len(values)
    j=0
    while j<len(ordered):
        k=j+1
        while k<len(ordered) and values[ordered[k]]==values[ordered[j]]:
            k+=1
        avg=((j+1)+k)/2
        for idx in ordered[j:k]:ranks[idx]=avg
        j=k
    return ranks


def rank_correlation(a,b):
    """Spearman correlation with tied midranks; None if constant."""
    if len(a)!=len(b) or not a:return None
    ra=midranks(a);rb=midranks(b)
    ma=mean(ra);mb=mean(rb)
    va=sum((v-ma)**2 for v in ra)
    vb=sum((v-mb)**2 for v in rb)
    if va==0 or vb==0:return None
    return sum((x-ma)*(y-mb) for x,y in zip(ra,rb))/math.sqrt(va*vb)


def paired_spearman(paired,bootstraps,seed):
    values=[(r["sum"],v["sum"]) for r,v in paired.values()]
    corr=rank_correlation([x for x,y in values],[y for x,y in values])
    if corr is None:raise ValueError("Cannot compute rank stability")
    rng=random.Random(seed)
    draws=[]
    for replicate in range(bootstraps):
        sample=[values[rng.randrange(len(values))] for _ in values]
        result=rank_correlation(
            [x for x,y in sample],[y for x,y in sample])
        if result is not None:draws.append(result)
    if len(draws)<int(.9*bootstraps):
        raise ValueError("Too many invalid bootstrap samples")
    return corr,quantile(draws,.025),quantile(draws,.975)


def run(data_root,manifest,out,seed=20261018,bootstraps=500,minimum=10):
    root,manifest,out=[Path(p).expanduser().resolve()
                       for p in (data_root,manifest,out)]
    if not (root/"PPMI").is_dir() or not manifest.is_file():
        raise FileNotFoundError("Missing PPMI raw data or frozen manifest")
    if out==root or root in out.parents or out==manifest:
        raise ValueError("Unsafe report path")
    if minimum<10 or bootstraps<100:raise ValueError("Invalid privacy/bootstrap setting")
    remote,onsite,paired,qc,source,train_count=load_populations(root,manifest)
    if min(map(len,(remote,onsite,paired)))<minimum:
        raise ValueError("Not enough TRAIN subjects for rank calibration")
    score_pairs={pid:(r,v) for pid,(r,v) in paired.items()}
    pr={pid:r for pid,(r,v) in score_pairs.items()}
    pv={pid:v for pid,(r,v) in score_pairs.items()}
    rho,lower,upper=paired_spearman(score_pairs,bootstraps,seed)
    rows=[];matched=[];tie_impact=[]
    for budget in BUDGETS:
        site_choice=exact_select(onsite,budget,seed)
        remote_choice=exact_select(remote,budget,seed)
        remote_threshold=threshold_benchmark(onsite,remote,paired,budget)
        higher_site,equal_site,cutoff=boundary_ties(onsite,budget)
        higher_remote,equal_remote,_=boundary_ties(remote,budget)
        # R01 threshold transfer may produce fewer/more selections than
        # calibrated exact-quota rank selection in real R01.
        imported={pid for pid,r in remote.items() if r["sum"]>=cutoff}
        overlap=len(imported&remote_choice)
        rows.append({
            "nominal_budget_pct":round(100*budget),
            "V04_population":len(onsite),
            "V04_exact_k":len(site_choice),
            "V04_all_ties_selected":remote_threshold["site_selected_n"],
            "R01_population":len(remote),
            "R01_exact_k":len(remote_choice),
            "R01_V04_cutoff_selected":len(imported),
            "R01_exact_k_and_cutoff_overlap":overlap,
            "R01_actual_budget_pct":round(100*len(remote_choice)/len(remote),2),
            "R01_imported_cutoff_pct":round(100*len(imported)/len(remote),2),
        })
        tie_impact.append({
            "nominal_budget_pct":round(100*budget),
            "V04_above_boundary_count":higher_site,
            "V04_equal_boundary_count":equal_site,
            "R01_above_boundary_count":higher_remote,
            "R01_equal_boundary_count":equal_remote,
        })
        # Selection stability in SAME paired population, with exact
        # counts and same tie-break hash, but DIFFERENT visit months.
        choose_r=exact_select(pr,budget,seed)
        choose_v=exact_select(pv,budget,seed)
        common=len(choose_r&choose_v)
        union=len(choose_r|choose_v)
        matched.append({
            "nominal_budget_pct":round(100*budget),
            "paired_population":len(score_pairs),
            "R01_selected":len(choose_r),
            "V04_selected":len(choose_v),
            "both_selected":common,
            "switched_selected_status":len(choose_r^choose_v),
            "jaccard_selected_sets":round(common/union,4) if union else "",
            "retention_at_V04_pct":round(100*common/len(choose_r),2)
                                   if choose_r else "",
        })
    out.mkdir(parents=True,exist_ok=True)
    save(out/"exact_budget_calibration.csv",[
        "nominal_budget_pct","V04_population","V04_exact_k",
        "V04_all_ties_selected","R01_population","R01_exact_k",
        "R01_V04_cutoff_selected","R01_exact_k_and_cutoff_overlap",
        "R01_actual_budget_pct","R01_imported_cutoff_pct"],rows)
    save(out/"budget_boundary_ties.csv",[
        "nominal_budget_pct","V04_above_boundary_count",
        "V04_equal_boundary_count","R01_above_boundary_count",
        "R01_equal_boundary_count"],tie_impact)
    save(out/"matched_patient_topk_stability.csv",[
        "nominal_budget_pct","paired_population","R01_selected",
        "V04_selected","both_selected","switched_selected_status",
        "jaccard_selected_sets","retention_at_V04_pct"],matched)
    save(out/"paired_spearman.csv",[
        "paired_patients","spearman_midrank","bootstrap_lower_95",
        "bootstrap_upper_95","bootstrap_replicates"],[{
            "paired_patients":len(score_pairs),
            "spearman_midrank":round(rho,4),
            "bootstrap_lower_95":round(lower,4),
            "bootstrap_upper_95":round(upper,4),
            "bootstrap_replicates":bootstraps,
        }])
    save(out/"quality_flags.csv",["check","records_or_patients"],[
        {"check":key,"records_or_patients":safe(n,minimum)}
        for key,n in sorted(qc.items())])
    summary={
        "status":"TRAIN_ONLY_exact_budget_remote_rank_stability_no_outcome",
        "train_registry":train_count,
        "eligible_R01":len(remote),"eligible_V04":len(onsite),
        "paired_R01_to_later_V04":len(score_pairs),
        "budget_percentages":[int(100*b) for b in BUDGETS],
        "ranking_feature":"27 visible MDS-UPDRS Part III items sum",
        "ties":"deterministic SHA256 seeded patient-ID tie break, used only in memory",
        "limitations":[
            "Exact-budget selection is NOT proof of useful or safe real R01 referral.",
            "R01 six-hidden ground truth is structurally unobserved.",
            "V04 occurs months after R01; rank instability can include disease change.",
            "Rank correlation is descriptive and cannot isolate modality shift.",
            "Seeded hash resolves ties arbitrarily, not clinically or fairly.",
            "V04 imported threshold includes all integer score ties, possibly exceeding budget.",
            "No val/test observations, patient IDs or decisions exported.",
            "All aggregate top-k counts are distribution summaries, not clinical outcomes.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+chr(10))
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--manifest-db",type=Path,required=True)
    parser.add_argument("--report-dir",type=Path,
                        default=Path("outputs/phase40_budget_calibration"))
    parser.add_argument("--seed",type=int,default=20261018)
    parser.add_argument("--bootstraps",type=int,default=500)
    parser.add_argument("--min-cell-count",type=int,default=10)
    a=parser.parse_args()
    print(json.dumps(run(a.data_root,a.manifest_db,a.report_dir,a.seed,
                         a.bootstraps,a.min_cell_count),indent=2,
                     ensure_ascii=False))
    print("Upload aggregate Phase40 report only; keep all patient data private.")

if __name__=="__main__":
    main()
