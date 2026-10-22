#!/usr/bin/env python3
"""Phase57: Phase56-cohort frozen R01 visible-27 total-score benchmark.

Compare exactly the 197 (or actual audit count) real R01->future V04
patients against Phase56 model choices, WITHOUT training or reusing TEST.
Uses only R01's actually scored visible items for each patient's risk.
A complete R01 visible sum is an UNCALIBRATED ranking score, not a
future-site six-item estimate. Reports aggregate numbers only.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,random
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean
from scripts.phase36_masked_site_feasibility import ALL,HIDDEN,VISIBLE
from scripts.phase54_masked_sequences import load_train
from scripts.phase56_remote_to_site_wm import find_pairs,auc

HIDX=tuple(ALL.index(k) for k in HIDDEN)
VIDX=tuple(ALL.index(k) for k in VISIBLE)

def build_scores(pairs):
    records=[]
    for pid,(bl,r,v) in pairs.items():
        if not all(r["status"][j]==1 for j in VIDX):
            raise ValueError("R01 visible item actually missing")
        if not all(v["status"][j]==1 for j in HIDX):
            raise ValueError("True future six item score unavailable")
        score=sum(int(r["score"][j]) for j in VIDX)
        positive=int(any(v["score"][j]>0 for j in HIDX))
        records.append((pid,score,positive))
    return records

def summary_at_budget(records,seed):
    n=len(records);positive=sum(z for _,_,z in records)
    if n<20 or positive==0 or positive==n:
        raise ValueError("Insufficient mixed outcomes")
    k=min(n,max(1,int(n*.20+.5)))
    rank=sorted(records,key=lambda row:(-row[1],hashlib.sha256(
        f"Phase56|{seed}|{row[0]}".encode()).hexdigest()))
    captures=sum(row[2] for row in rank[:k])
    auroc=auc([{"risk":score,"positive":label}
               for _,score,label in records])
    # Exact-k random selection expectation is transparent and model-free.
    expected=positive*k/n
    return {"method":"R01_visible_27_sum_locked_simple_baseline",
            "eligible":n,"positive":positive,"selected":k,
            "captured_positive":captures,
            "topk_recall":round(captures/positive,6),
            "auroc":round(auroc,6),
            "expected_random_positive":round(expected,5),
            "random_expected_recall":round(k/n,6),
            "tie_break":"Phase56_SHA256_seed_patient",
            "risk_is_uncalibrated_visible_sum":True}

def run(patients,out,seed=20261018,bootstraps=500):
    pairs,_=find_pairs(patients)
    records=build_scores(pairs)
    stats=summary_at_budget(records,seed)
    rng=random.Random(seed+357)
    samples=[]
    for _ in range(bootstraps):
        drawn=[records[rng.randrange(len(records))] for j in records]
        try:
            met=summary_at_budget(drawn,seed)
            samples.append((met["topk_recall"],met["auroc"]))
        except ValueError:
            continue
    if len(samples)<100:raise ValueError("Insufficient valid bootstrap samples")
    def percentile(vals,p):
        z=sorted(vals);x=(len(z)-1)*p;j=int(x);k=min(j+1,len(z)-1)
        return round(z[j]*(k-x)+z[k]*(x-j),6)
    bootstrap=[{
        "method":stats["method"],"bootstrap_valid_draws":len(samples),
        "topk_recall_ci95_low":percentile([a for a,b in samples],.025),
        "topk_recall_ci95_high":percentile([a for a,b in samples],.975),
        "auroc_ci95_low":percentile([b for a,b in samples],.025),
        "auroc_ci95_high":percentile([b for a,b in samples],.975),
    }]
    out=Path(out).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    for name,rows in (("visible_sum_27_metrics.csv",[stats]),
                      ("visible_sum_27_bootstrap.csv",bootstrap)):
        with (out/name).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    summary={"status":"TRAIN_only_Phase56_exact_cohort_visible_sum_benchmark",
             "strict_real_R01_future_site_pairs":len(records),
             "seed":seed,
             "target":"future V04 hidden-six score >0",
             "caveats":[
                "Same cohort as Phase56 if same Git commit and eligibility.",
                "R01 27 score sum is an uncalibrated ranking baseline.",
                "Site visit occurs after R01; no same-visit hidden truth.",
                "Bootstrap is for this baseline alone; not paired WM vs baseline.",
                "No original consumed Phase44 TEST access.",
                "No IDs, scores, individual choices or patient rows exported."],
             "generated_utc":datetime.now(timezone.utc).isoformat()}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return stats

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    args=p.parse_args()
    patients,_=load_train(args.data_root,args.manifest_db)
    print(json.dumps(run(patients,args.report_dir),indent=2))
if __name__=="__main__":
    main()
