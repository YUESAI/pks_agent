#!/usr/bin/env python3
"""Phase53: TRAIN-only item-mask-aware motor visit capacity audit.

Phase51 used documented NP3TOT totals. At genuine remote R01 visits,
NP3TOT is often empty because six hands-on items are coded 101/UR, while
all 27 remotely visible items may be measured. Excluding such visits
would silently erase genuine *partially observed* clinical information.

Here a clinical observation = (actual month, actual event, 33 item
values, 33 validity/UR masks, documented total provenance). No hidden
score is imputed or treated as zero. We keep only the frozen TRAIN
Prodromal patient registry, reject discordant same-month duplicates
and inappropriate visit times, and export AGGREGATE counts only.

This audit does NOT learn a new world model, infer intervention effects,
or access the consumed Phase44 TEST.
"""
from __future__ import annotations
import argparse,csv,json,math,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase22_strict_motor_pairs import one_file,parse_month,state_name,safe
from scripts.phase31_motor_repetitions import train_anchors
from scripts.phase36_masked_site_feasibility import ALL,VISIBLE,HIDDEN,score,total

EVENT_BINS=("BL","R01","V04","OTHER")
PATIENT_VISITS=("0","1","2","3","4","5plus")
MASK_TYPES=("complete_33_verified_total",
            "complete_33_total_absent",
            "remote_27_valid_six_unrateable",
            "partial_27plus",
            "partial_10to26",
            "unusable_below10")
INTERVALS=("1_3","4_8","9_15","16_24","25_36","37plus")


def event_bin(e):
    return e if e in ("BL","R01","V04") else "OTHER"


def visit_band(n):
    return str(n) if n<5 else "5plus"


def gap_band(dt):
    if dt<=3:return "1_3"
    if dt<=8:return "4_8"
    if dt<=15:return "9_15"
    if dt<=24:return "16_24"
    if dt<=36:return "25_36"
    return "37plus"


def categorize_episode(items,documented):
    """Only measurements 0..4 are known; 101/UR is never a zero score."""
    valid={key:int(v) for key,(st,v) in items.items() if st=="valid"}
    nvalid=len(valid)
    six_unknown=all(items[key][0]=="unable" for key in HIDDEN)
    all_visible=all(key in valid for key in VISIBLE)
    if nvalid==33:
        calculated=sum(valid.values())
        if documented is None:
            return "complete_33_total_absent",nvalid,False
        if documented==calculated:
            return "complete_33_verified_total",nvalid,False
        return "invalid_full_total_mismatch",nvalid,False
    if all_visible and six_unknown and nvalid==27:
        return "remote_27_valid_six_unrateable",nvalid,True
    if nvalid>=27:return "partial_27plus",nvalid,False
    if nvalid>=10:return "partial_10to26",nvalid,False
    return "unusable_below10",nvalid,False


def audit_items(rows,anchors,min_cell=10):
    """Accept pre-parsed row dictionaries; keep patient data RAM-only.

    Include only UNSPECIFIED medication state (not confirmed untreated).
    True BL must be at frozen month; followups strictly later. R01/V04
    must satisfy frozen study windows to avoid provenance drift.
    """
    if min_cell<10:raise ValueError("Minimum aggregate cell size >=10")
    quality=Counter()
    indexed=defaultdict(list)
    for row in rows:
        pid=str(row.get("PATNO","")).strip()
        if pid not in anchors:continue
        anchor=anchors[pid]
        if anchor is None:
            quality["missing_bl_anchor_source_rows"]+=1
            continue
        event=str(row.get("EVENT_ID","")).strip().upper()
        if not event or event in ("SC","CONSENT"):
            quality["excluded_non_exam_event"]+=1
            continue
        month=parse_month(row.get("EXAMDT") or row.get("INFODT"))
        if month is None:
            quality["invalid_exam_date"]+=1;continue
        delta=month-anchor
        if event=="BL" and delta!=0:
            quality["misdated_baseline"]+=1;continue
        if event!="BL" and delta<=0:
            quality["bad_followup_chronology"]+=1;continue
        if event=="R01" and not 4<=delta<=8:
            quality["off_window_R01"]+=1;continue
        if event=="V04" and not 9<=delta<=15:
            quality["off_window_V04"]+=1;continue
        if state_name(row.get("PDSTATE",""))!="UNSPECIFIED":
            quality["nonprimary_medication_state_excluded"]+=1;continue
        values={name:score(row.get(name,"")) for name in ALL}
        documented=total(row.get("NP3TOT",""))
        kind,nvalid,isremote=categorize_episode(values,documented)
        if kind=="invalid_full_total_mismatch":
            quality["complete_33_total_mismatch"]+=1
            continue
        if kind=="unusable_below10":
            quality["fewer_than_10_valid_items"]+=1
            continue
        # Identical duplicate source records collapse; conflicting
        # same patient/month records (including different events)
        # are conservatively quarantined, not silently overwritten.
        fingerprint=(event,tuple(values[name] for name in ALL),documented)
        indexed[(pid,month)].append((fingerprint,kind,nvalid,isremote))
    accepted=defaultdict(dict)
    for (pid,month),candidates in indexed.items():
        keys={v[0] for v in candidates}
        if len(keys)!=1:
            quality["conflicting_same_patient_month_item_records"]+=1
            continue
        token,kind,nvalid,isremote=candidates[0]
        if len(candidates)>1:
            quality["identical_source_duplicate_collapsed"]+=1
        accepted[pid][month]=(token[0],kind,nvalid,isremote)
    patterns=Counter()
    by_event=Counter()
    patient_counts=Counter()
    anchored_counts=Counter()
    gap_counts=Counter()
    all_pairs=0
    anchored_pairs=0
    remote_people=set()
    remote_to_site=set()
    rescued_rows=0
    for pid,anchor in anchors.items():
        items=accepted.get(pid,{})
        values=sorted(items.items())
        count=len(values)
        patient_counts[visit_band(count)]+=1
        has_bl=(anchor in items and items[anchor][0]=="BL")
        if has_bl:
            anchored_counts[visit_band(count)]+=1
        for month,info in values:
            event,kind,nvalid,isremote=info
            patterns[kind]+=1
            by_event[(event_bin(event),kind)]+=1
            if isremote and event=="R01":
                remote_people.add(pid)
            if isremote and info[0]=="R01":
                # Total may be absent or supplied despite masked items.
                # No value of a 101-coded item is ever inferred.
                rescued_rows+=1
        for (month_a,_),(month_b,_) in zip(values,values[1:]):
            if month_b<=month_a:raise AssertionError("Invalid visit order")
            if has_bl:
                gap_counts[gap_band(month_b-month_a)]+=1
                anchored_pairs+=1
            all_pairs+=1
        if has_bl:
            remotes=[m for m,info in values if
                     info[0]=="R01" and info[1]=="remote_27_valid_six_unrateable"]
            sites=[m for m,info in values if info[0]=="V04" and
                   info[1]=="complete_33_verified_total"]
            if any(site>remote for remote in remotes for site in sites):
                remote_to_site.add(pid)

    def sup(n):return safe(n,min_cell)
    anchored_n=sum(anchored_counts.values())
    count_at_least=lambda min_n:sum(
        num for cat,num in anchored_counts.items()
        if (5 if cat=="5plus" else int(cat))>=min_n)
    summary={
        "status":"TRAIN_only_PPMI_mask_aware_motor_episode_capacity",
        "frozen_train_patients":len(anchors),
        "anchored_valid_motor_episode_patients":anchored_n,
        "baseline_anchored_at_least_2_episodes":count_at_least(2),
        "baseline_anchored_at_least_3_episodes":count_at_least(3),
        "baseline_anchored_at_least_4_episodes":count_at_least(4),
        "baseline_anchored_at_least_5_episodes":count_at_least(5),
        "all_usable_partial_or_complete_episodes":sum(patterns.values()),
        "baseline_anchored_episode_transition_pairs":anchored_pairs,
        "all_episode_transition_pairs_including_unanchored":all_pairs,
        "remote_visible27_six_unrateable_episodes":patterns[
            "remote_27_valid_six_unrateable"],
        "unique_patients_with_real_R01_masked_episode":len(remote_people),
        "patients_with_masked_R01_and_later_verified_V04":len(remote_to_site),
        "total_missing_R01_source_rows_not_imputed":rescued_rows,
        "min_valid_motor_items_per_usable_episode":10,
        "study_event_windows":{"R01":[4,8],"V04":[9,15]},
        "original_consumed_final_test_accessed":False,
        "limits":[
            "Partial observation is retained with an explicit mask, not imputed.",
            "A 101/UR hands-on item is unknown, NOT a measured zero.",
            "Some usable episodes may have no recorded NP3TOT; not direct equivalents to Phase51.",
            "A single patient can contribute many transitions; counts are not independent sample sizes.",
            "Only frozen TRAIN participant IDs and local raw PPMI motor rows are consulted.",
            "UNSPECIFIED is not a verified drug or disease state.",
            "No inferred treatment actions or intervention counterfactuals.",
            "No patient IDs, visit dates, item scores or sequences exported."
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat()
    }
    tables={
       "episode_mask_patterns.csv":[{"mask_type":k,"episodes":sup(patterns[k])}
                                    for k in MASK_TYPES],
       "event_by_mask_pattern.csv":[{"event":event,"mask_type":kind,
                                      "episodes":sup(by_event[(event,kind)])}
                                    for event in EVENT_BINS for kind in MASK_TYPES],
       "all_patient_episode_distribution.csv":[
           {"usable_episodes":k,"TRAIN_patients":sup(patient_counts[k])}
           for k in PATIENT_VISITS],
       "baseline_anchored_episode_distribution.csv":[
           {"usable_episodes":k,"TRAIN_patients":sup(anchored_counts[k])}
           for k in PATIENT_VISITS],
       "anchored_adjacent_gap_bins.csv":[
           {"months_apart":k,"transition_pairs":sup(gap_counts[k])}
           for k in INTERVALS],
       "quality_flags.csv":[
           {"reason":name,"TRAIN_rows_or_keys":sup(n)}
           for name,n in sorted(quality.items())]
    }
    return summary,tables


def iter_source(source,anchors):
    """Defensive duplicate-header-aware scan, only recognized TRAIN IDs."""
    with source.open("r",encoding="utf-8-sig",errors="replace",newline="") as f:
        reader=csv.reader(f)
        columns=defaultdict(list)
        for index,name in enumerate(next(reader,[])):
            columns[name.strip().upper()].append(index)
        required=set(ALL)|{"PATNO","EVENT_ID","NP3TOT"}
        if not required.issubset(columns):
            raise ValueError("Missing required Motor item columns: "+str(
                             sorted(required-set(columns))))
        fields=list(ALL)+["PATNO","EVENT_ID","EXAMDT","INFODT",
                          "NP3TOT","PDSTATE"]
        for raw in reader:
            patient_fields=[raw[i].strip() for i in
                            columns["PATNO"] if i<len(raw)]
            pids={x for x in patient_fields if x}
            if len(pids)!=1 or next(iter(pids)) not in anchors:continue
            parsed={}
            inconsistent=False
            for field in fields:
                candidates=[raw[i].strip() for i in columns.get(field,[])
                            if i<len(raw) and raw[i].strip()]
                if len(set(candidates))>1:
                    inconsistent=True;break
                parsed[field]=candidates[0] if candidates else ""
            if not inconsistent:
                yield parsed


def run(root,manifest,out,min_cell=10):
    root,manifest,out=[Path(p).expanduser().resolve()
                       for p in (root,manifest,out)]
    if not (root/"PPMI").is_dir() or not manifest.is_file():
        raise FileNotFoundError("Missing frozen PPMI TRAIN sources")
    if (root==out or root in out.parents or
        manifest==out or out==manifest.parent or min_cell<10):
        raise ValueError("Unsafe output path or unsuppressed report")
    anchors=train_anchors(manifest)
    source=one_file(root/"PPMI","MDS-UPDRS_Part_III_")
    summary,tables=audit_items(iter_source(source,anchors),anchors,min_cell)
    summary["source_file"]=source.name
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():
        if not rows:continue
        with (out/name).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--min-cell",type=int,default=10)
    args=p.parse_args()
    print(json.dumps(run(args.data_root,args.manifest_db,
                         args.report_dir,args.min_cell),indent=2))
    print("Share only suppressed aggregate Phase53 reports.")


if __name__=="__main__":main()
