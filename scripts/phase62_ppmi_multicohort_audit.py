#!/usr/bin/env python3
"""Phase62: PPMI PD/HC/Prodromal baseline-anchored, mask-aware cohort audit.

Source: local PPMI records only. All raw identifiers/scores stay on cluster.
Prodromal IDs restricted to original frozen TRAIN manifest (not consumed TEST).
PD and HC cohort membership is established using the legacy strict BL mapping.
Medication state is retained and never conflated; 101/UR is not score zero.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sqlite3
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

from scripts.phase22_strict_motor_pairs import (
    one_file,records,parse_month,state_name,baseline_group,present,
    target_for_window,WINDOWS,safe
)
from scripts.phase31_motor_repetitions import train_anchors
from scripts.phase36_masked_site_feasibility import ALL,HIDDEN,VISIBLE,score,total
from scripts.phase53_mask_aware_motor_readiness import iter_source,categorize_episode

GROUPS=("prodromal","PD","HC")
STATES=("UNSPECIFIED","OFF","ON")
MIN_ITEMS=10
HIDX=tuple(ALL.index(x) for x in HIDDEN)
VIDX=tuple(ALL.index(x) for x in VISIBLE)


def clinical_registry(root,manifest):
    """A new, strict registry: never add a consumed Prodromal VAL/TEST ID."""
    folder=root/"PPMI"
    qc=Counter()
    cohorts=defaultdict(set);diagnoses=defaultdict(set);anchors=defaultdict(set)
    for row in records(one_file(folder,"Participant_Status_"),
                       ("PATNO","COHORT"),qc):
        if present(row["PATNO"]) and present(row["COHORT"]):
            cohorts[row["PATNO"]].add(row["COHORT"])
    for row in records(one_file(folder,"Primary_Research_Diagnosis_"),
                       ("PATNO","EVENT_ID","PRIMDIAG"),qc):
        if row["EVENT_ID"]=="BL" and present(row["PATNO"]):
            diagnoses[row["PATNO"]].add(
                row["PRIMDIAG"] if present(row["PRIMDIAG"]) else None)
    for row in records(one_file(folder,"Baseline_Visit_Start_"),
                       ("PATNO","EVENT_ID","INFODT"),qc):
        if row["EVENT_ID"]=="BL" and present(row["PATNO"]):
            anchors[row["PATNO"]].add(parse_month(row["INFODT"]))

    frozen=train_anchors(manifest)
    registry={}
    breakdown=Counter()
    for pid in cohorts.keys()|diagnoses.keys():
        codes=cohorts.get(pid,set());diag=diagnoses.get(pid,set())
        if len(codes)!=1 or len(diag)!=1 or None in diag:
            breakdown["ambiguous_cohort_or_baseline_diagnosis"]+=1
            continue
        group=baseline_group(next(iter(codes)),next(iter(diag)))
        if group not in GROUPS:
            breakdown["outside_strict_three_group_mapping"]+=1
            continue
        if group=="prodromal":
            if pid not in frozen:
                breakdown["prodromal_not_in_frozen_TRAIN"]+=1
                continue
        anchor=anchors.get(pid,set())
        if len(anchor)!=1 or None in anchor:
            breakdown["missing_or_ambiguous_BL_anchor"]+=1
            continue
        month=next(iter(anchor))
        if group=="prodromal" and frozen[pid]!=month:
            breakdown["prodromal_frozen_month_discordance"]+=1
            continue
        registry[pid]=(group,month)
        breakdown["strict_"+group]+=1
    if not registry:raise ValueError("No validated baseline participants")
    return registry,breakdown,qc


def classify_visit(row,anchor,quality):
    event=str(row["EVENT_ID"]).upper()
    month=parse_month(row.get("EXAMDT") or row.get("INFODT"))
    if month is None or event in ("","SC","CONSENT"):
        quality["not_a_dated_clinical_visit"]+=1
        return None
    gap=month-anchor
    if event=="BL" and gap!=0:
        quality["misdated_BL"]+=1
        return None
    if event!="BL" and gap<=0:
        quality["invalid_followup_chronology"]+=1
        return None
    if event=="R01" and not 4<=gap<=8:
        quality["outside_R01_window"]+=1
        return None
    if event=="V04" and not 9<=gap<=15:
        quality["outside_V04_window"]+=1
        return None
    state=state_name(row.get("PDSTATE",""))
    if state not in STATES:
        quality["other_medication_code"]+=1
        return None
    values={key:score(row.get(key,"")) for key in ALL}
    documented=total(row.get("NP3TOT",""))
    kind,nvalid,is_remote=categorize_episode(values,documented)
    if kind=="invalid_full_total_mismatch":
        quality["inconsistent_33item_total"]+=1
        return None
    if nvalid<MIN_ITEMS:
        quality["fewer_than_10_measured_items"]+=1
        return None
    scores=tuple(v if st=="valid" else 0 for st,v in
                 (values[k] for k in ALL))
    mask=tuple(1 if values[k][0]=="valid" else
               2 if values[k][0]=="unable" else 0 for k in ALL)
    visit={"month":month,"event":event,"state":state,
           "score":scores,"status":mask,
           "documented_total":documented,"mask_type":kind}
    return month,state,visit


def load_cohorts(root,manifest):
    """One eligible baseline state per person selected BEFORE future selection.

    UNSPECIFIED > OFF > ON preference is prespecified, not chosen using future
    outcome availability. Only followups with the same recorded state remain.
    """
    root=Path(root).resolve();manifest=Path(manifest).resolve()
    if not (root/"PPMI").is_dir() or not manifest.is_file():
        raise FileNotFoundError("Missing PPMI or frozen manifest")
    registry,counts,source_qc=clinical_registry(root,manifest)
    anchors={pid:month for pid,(group,month) in registry.items()}
    quality=Counter()
    buckets=defaultdict(list)
    source=one_file(root/"PPMI","MDS-UPDRS_Part_III_")
    for row in iter_source(source,anchors):
        pid=row["PATNO"]
        accepted=classify_visit(row,anchors[pid],quality)
        if accepted is None:continue
        month,state,v=accepted
        buckets[(pid,month,state)].append(v)

    entries=defaultdict(lambda:defaultdict(dict))
    for (pid,month,state),variants in buckets.items():
        fingerprints={(v["event"],v["score"],v["status"],
                       v["documented_total"]) for v in variants}
        if len(fingerprints)>1:
            quality["discordant_same_patient_month_state"]+=1
            continue
        if len(variants)>1:quality["identical_duplicate_collapsed"]+=1
        entries[pid][state][month]=variants[0]
    patients=[]
    for pid,(group,anchor) in registry.items():
        state_map=entries.get(pid,{})
        choice=next((state for state in STATES
                     if anchor in state_map.get(state,{}) and
                     state_map[state][anchor]["event"]=="BL"),None)
        if choice is None:
            quality["registered_without_usable_BL_motor"]+=1
            continue
        visits=sorted(state_map[choice].values(),key=lambda v:v["month"])
        if visits[0]["event"]!="BL" or visits[0]["month"]!=anchor:
            raise AssertionError("Unanchored selected BL")
        patients.append({"pid":pid,"cohort":group,"state":choice,
                         "baseline_month":anchor,"visits":visits})
    if len({p["pid"] for p in patients})!=len(patients):
        raise AssertionError("Duplicate person across cohorts")
    report={"registry_counts":dict(counts),
            "quality_flags":{k:safe(v,10) for k,v in
                             sorted((quality+source_qc).items())},
            "selected_motor_patients":len(patients),
            "selected_cohort_counts":dict(Counter(p["cohort"] for p in patients)),
            "source_motor_table":source.name}
    return patients,report


def future_site_pairs(patients,horizon):
    """One fixed-horizon BL -> verified future six-score target per person.

    Only records with 33 actually scored target items AND a consistent
    documented total are eligible; target month used to select outcome
    BUT never supplied to any baseline predictor. Target EVENT_ID not used
    as model input. Medication-state continuity is required.
    """
    specs={h:tol for h,tol in WINDOWS}
    if horizon not in specs:raise ValueError("Horizon must be 6,12,24")
    output=[]
    counts=Counter()
    for patient in patients:
        group=patient["cohort"]
        baseline=patient["visits"][0]
        # Do not condition on target burden during visit selection.
        candidates={v["month"]:v for v in patient["visits"][1:]
                    if abs(v["month"]-patient["baseline_month"]-horizon)
                    <=specs[horizon]
                    and v["mask_type"]=="complete_33_verified_total"}
        status,selection=target_for_window(
            {month:0 for month in candidates},
            patient["baseline_month"],horizon,specs[horizon])
        counts[(group,status)]+=1
        if status!="paired":continue
        month,_=selection
        target=candidates[month]
        if not all(target["status"][j]==1 for j in HIDX):
            raise AssertionError("Unmeasured target leakage")
        burden=sum(target["score"][j] for j in HIDX)
        output.append({"pid":patient["pid"],"cohort":group,
                       "state":patient["state"],"baseline":baseline,
                       "target_burden":burden,"target_positive":int(burden>0)})
    return output,counts


def audit_tables(patients,registry_meta):
    from collections import Counter
    rows=[];visits=[];pair_rows=[];status=[]
    for group in GROUPS:
        selected=[p for p in patients if p["cohort"]==group]
        counted=Counter(p["state"] for p in selected)
        by_visits=Counter(min(5,len(p["visits"])) for p in selected)
        for state in STATES:
            rows.append({"cohort":group,"baseline_medication_state":state,
                         "participants":safe(counted[state],10)})
        for n in (1,2,3,4,5):
            visits.append({"cohort":group,"n_valid_baseline_plus_visits":
                           str(n) if n<5 else "5plus",
                           "participants":safe(by_visits[n],10)})
        for horizon in (6,12,24):
            examples,counts=future_site_pairs(selected,horizon)
            positives=sum(e["target_positive"] for e in examples)
            pair_rows.append({
                "cohort":group,"horizon_months":horizon,
                "strict_pairs":safe(len(examples),10),
                "future_six_positive":safe(positives,10),
                "future_six_zero":safe(len(examples)-positives,10),
                "target_not_in_window":safe(counts[group,"no_in_window_target"],10),
                "target_equidistant":safe(counts[group,"equidistant_months"],10)})
        for mask_type in (
                "complete_33_verified_total","complete_33_total_absent",
                "remote_27_valid_six_unrateable","partial_27plus",
                "partial_10to26"):
            count=sum(v["mask_type"]==mask_type for p in selected
                      for v in p["visits"])
            status.append({"cohort":group,"mask_type":mask_type,
                           "recorded_visits":safe(count,10)})
    return {"cohort_medication_counts.csv":rows,
            "longitudinal_episode_counts.csv":visits,
            "strict_future_six_pair_capacity.csv":pair_rows,
            "masked_visit_coverage.csv":status}


def export_audit(patients,metadata,out):
    out=Path(out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=False)
    tables=audit_tables(patients,metadata)
    for name,rows in tables.items():
        with (out/name).open("w",newline="",encoding="utf8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
    summary={**metadata,
        "protocol":"Phase62_PPMI_BL_strict_PD_HC_ProdromalTRAIN_masked_state_aware",
        "valid_groups":list(GROUPS),
        "target_horizons_months":[6,12,24],
        "source_files_local_only":True,
        "original_consumed_Prodromal_TEST_accessed":False,
        "constraints":[
            "Prodromal whitelist is frozen original TRAIN only",
            "PD/HC are newly defined retrospective development groups, not external validation",
            "Group comes from single COHORT and single BL PRIMDIAG code",
            "No cross-group patient duplication",
            "Only one BL medication state chosen a priori for each person",
            "Medication-state continuity: ON/OFF/UNSPECIFIED never conflated",
            "101/UR preserved in masks; never converted into measured zero",
            "All target six items are actually recorded and NP3TOT verified",
            "Six-month target can be infeasible if remote hands-on scores are UR",
            "No row-level patient data or source clinical records exported",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat()}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    a=p.parse_args()
    patients,meta=load_cohorts(a.data_root,a.manifest_db)
    result=export_audit(patients,meta,a.report_dir)
    print(json.dumps({"status":"complete","selected_motor_patients":
                      result["selected_motor_patients"],
                      "report_dir":str(a.report_dir)},indent=2))
    print("Only aggregate results should leave cluster.")


if __name__=="__main__":main()
