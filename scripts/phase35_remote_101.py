#!/usr/bin/env python3
"""Phase35: TRAIN-only remote Part III item codebook and observability audit.

Validate numeric 101 (Unable to Rate) in six hands-on MDS-UPDRS Part III
items of 6-month R01 remote visits. Assess remaining item completeness and
patient overlap with later V04 in-person visits. No total score imputation,
individual patient values, predictions, dates, or IDs are exported.

Later clinic visits are NOT contemporaneous ground truth for remote exams.
No val/test patients or clinical target scores are queried.
"""
import argparse,csv,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase22_strict_motor_pairs import one_file,parse_month,state_name,safe,save
from scripts.phase31_motor_repetitions import train_anchors
from scripts.phase34_remote_motor import SPECIAL

def status(value):
    token=str(value or "").strip().upper()
    if token in ("","NA","N/A","NAN","NULL","NONE"):return "blank"
    if token=="UR":return "literal_UR"
    try:
        n=float(token)
    except ValueError:return "other"
    if n==101:return "numeric_101"
    if math.isfinite(n) and n in (0.,1.,2.,3.,4.):return "valid_0to4"
    return "other"

def bucket(n,maximum):
    if n==0:return "none"
    if n==maximum:return "all"
    if n>=20:return "20plus"
    if n>=10:return "10to19"
    return "1to9"

def calculate(root,manifest,out,minimum=10):
    root,manifest,out=[Path(v).expanduser().resolve() for v in (root,manifest,out)]
    if not (root/"PPMI").is_dir() or not manifest.is_file() or minimum<10:
        raise ValueError("Missing source/manifest or invalid suppression threshold")
    if out==manifest or root==out or root in out.parents:
        raise ValueError("Unsafe output location")
    anchors=train_anchors(manifest)
    path=one_file(root/"PPMI","MDS-UPDRS_Part_III_")
    counts=Counter();profiles=Counter()
    remote_patients=set();remotes_with_6_101=set()
    partial_remote=set();clinic_patients=set()
    all_remote=set()
    qc=Counter()
    with path.open(encoding="utf-8-sig",errors="replace",newline="") as fh:
        reader=csv.reader(fh)
        columns=defaultdict(list)
        for index,name in enumerate(next(reader,[])):
            columns[name.strip().upper()].append(index)
        required={"PATNO","EVENT_ID","NP3TOT",*SPECIAL}
        if not required.issubset(columns):
            raise ValueError("Missing required Part III item headers: "+
                             ",".join(sorted(required-set(columns))))
        other_items=[name for name in columns if name.startswith("NP3") and
                     name!="NP3TOT" and name not in SPECIAL]
        if not other_items:
            raise ValueError("No other NP3 score columns")
        for row in reader:
            def field(name):
                vals=[row[i].strip() for i in columns.get(name,[]) if i<len(row)]
                distinct={x for x in vals if x}
                if len(distinct)>1:raise ValueError("Conflicting duplicate header")
                return next(iter(distinct)) if distinct else ""
            try:
                pid=field("PATNO")
                if pid not in anchors or anchors[pid] is None:
                    continue
                event=field("EVENT_ID").upper()
                month=parse_month(field("EXAMDT") or field("INFODT"))
                if month is None:continue
                state=state_name(field("PDSTATE"))
                total=field("NP3TOT")
            except ValueError:
                qc["TRAIN_conflicting_scalar_header_rows"]+=1
                continue
            if state!="UNSPECIFIED":
                continue
            gap=month-anchors[pid]
            if event=="V04" and 9<=gap<=15:
                # Verify the full total is a valid finite 0..132 number.
                try:valid_total=0<=float(total)<=132 and math.isfinite(float(total))
                except ValueError:valid_total=False
                if valid_total:clinic_patients.add(pid)
            if event!="R01" or not 4<=gap<=8:
                continue
            all_remote.add(pid)
            if status(total)!="blank":
                counts[("R01_6m","nonblank_total_rows")]+=1
                continue
            remote_patients.add(pid)
            statuses=[]
            try:
                for item in SPECIAL:
                    kind=status(field(item))
                    statuses.append(kind)
                    counts[(item,kind)]+=1
                score_count=sum(status(field(item))=="valid_0to4"
                                for item in other_items)
            except ValueError:
                qc["TRAIN_remote_conflicting_item_headers"]+=1
                continue
            n101=sum(x=="numeric_101" for x in statuses)
            profiles[("six_101_count",str(n101))]+=1
            profiles[("other_score_coverage",bucket(score_count,len(other_items)))]+=1
            if n101==6:
                remotes_with_6_101.add(pid)
                if score_count>=20:partial_remote.add(pid)
    nrows=sum(profiles["six_101_count",str(i)] for i in range(7))
    out.mkdir(parents=True,exist_ok=True)
    item_rows=[{"item":item,"status":kind,
                "R01_blank_total_rows":safe(counts[item,kind],minimum)}
               for item in SPECIAL
               for kind in ("numeric_101","literal_UR","valid_0to4","blank","other")]
    save(out/"remote_six_item_codes.csv",
         ["item","status","R01_blank_total_rows"],item_rows)
    profile_rows=[{"metric":category,"category":label,
                   "R01_blank_total_rows":safe(number,minimum)}
                  for (category,label),number in sorted(profiles.items())]
    save(out/"remote_item_completion.csv",
         ["metric","category","R01_blank_total_rows"],profile_rows)
    coverage=[
        ("R01_any_6m_patient",len(all_remote)),
        ("R01_blank_total_patient",len(remote_patients)),
        ("R01_all_six_numeric_101_patient",len(remotes_with_6_101)),
        ("R01_all_six_101_other20plus_patient",len(partial_remote)),
        ("V04_valid_total_patient",len(clinic_patients)),
        ("R01_blank_total_and_later_V04_valid_patient",
         len(remote_patients & clinic_patients)),
        ("R01_all_six_101_other20plus_and_later_V04_valid_patient",
         len(partial_remote & clinic_patients)),
    ]
    save(out/"remote_clinic_overlap.csv",
         ["measure","TRAIN_patients"],
         [{"measure":k,"TRAIN_patients":safe(n,minimum)} for k,n in coverage])
    save(out/"quality_flags.csv",["flag","rows"],
         [{"flag":name,"rows":safe(count,minimum)}
          for name,count in sorted(qc.items())])
    summary={
        "status":"TRAIN_ONLY_remote_101_numeric_code_and_partial_exam_audit",
        "source_file":path.name,
        "train_registry":len(anchors),
        "six_remote_hands_on_items":list(SPECIAL),
        "non_six_NP3_header_count":len(other_items),
        "R01_six_month_blank_total_rows_processed":nrows,
        "R01_six_month_all_six_numeric_101_patients":
            safe(len(remotes_with_6_101),minimum),
        "warnings":[
            "Numeric 101 means Unable to Rate according to the PPMI 2024 CRF; confirm export mapping.",
            "V04 checks are months later and NOT ground truth for R01 on the same day.",
            "Other NP3 field counts are header-based; validate clinical item meanings.",
            "NO remote NP3TOT reconstructed or imputations performed.",
            "Only frozen TRAIN records were retained; val/test remain inaccessible.",
            "No person-level records, scores, raw timestamps or IDs exported.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",required=True,type=Path)
    parser.add_argument("--manifest-db",required=True,type=Path)
    parser.add_argument("--report-dir",type=Path,default=Path("outputs/phase35_remote_101"))
    parser.add_argument("--min-cell-count",type=int,default=10)
    a=parser.parse_args()
    print(json.dumps(calculate(a.data_root,a.manifest_db,a.report_dir,
                               a.min_cell_count),indent=2,ensure_ascii=False))
    print("Upload aggregate Phase35 report only; no private data.")
if __name__=="__main__":main()
