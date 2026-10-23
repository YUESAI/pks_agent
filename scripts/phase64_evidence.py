"""Phase64 retrospective PPMI multiscale clinical evidence service.

Strictly limited to patients returned by Phase62 load_cohorts:
original frozen Prodromal TRAIN + independently defined PD/HC.
Only baseline BL and real R01 (4..8 months AFTER BL) item values;
never read V04 or other future tables as patient evidence.
"""
from __future__ import annotations
import csv
import math
import re
from collections import Counter,defaultdict
from scripts.phase22_strict_motor_pairs import one_file,parse_month
from scripts.phase36_masked_site_feasibility import ALL,HIDDEN,VISIBLE

SPECS=(
 ("PartI","MDS-UPDRS_Part_I_",r"^NP1[A-Z0-9]+$",0.,4.),
 ("PartII","MDS_UPDRS_Part_II__Patient_Questionnaire_",r"^NP2[A-Z0-9]+$",0.,4.),
 ("MoCA","Montreal_Cognitive_Assessment__MoCA__","MCATOT",0.,30.),
 ("UPSIT","University_of_Pennsylvania_Smell_Identification_Test_UPSIT_","TOTAL_CORRECT",0.,40.),
 ("RBD","REM_Sleep_Behavior_Disorder_Screening_Questionnaire_","DRMVIVID|SLPINJUR",0.,4.),
 ("SCOPA","SCOPA-AUT_",r"^SCAU\d+$",0.,4.),
)
# Do not silently interpret 101/UR, 99, 999, etc. as clinical numbers.
ABSENT={"","NA","N/A","NAN","NONE","NULL","MISSING","UR","101","99","999","-1","."}
MOTOR=tuple(VISIBLE)


def numeric(raw,low,high):
    s=str(raw or "").strip().upper()
    if s in ABSENT:return None
    try:x=float(s)
    except (ValueError,TypeError):return None
    if not math.isfinite(x) or x!=int(x) or not low<=x<=high:
        return None
    return int(x)


def bucket(key,value):
    """Fixed, label-independent 5-category code; actual value retained separately."""
    if key.startswith(("BL_","R01_")):
        return max(0,min(4,int(value)))
    if key=="MoCA_MCATOT":return min(4,int(value)//7)
    if key=="UPSIT_TOTAL_CORRECT":return min(4,int(value)//9)
    return max(0,min(4,int(value)))


def evidence_tables(root,people):
    """Read only predeclared clinical measurement files; conservative uniqueness.

    Fail-fast for present but malformed tables; missing optional table recorded.
    Baseline event BL required. If a dated value is present it must match
    true BL month. Conflicting records are quarantined, never chosen by target.
    """
    from pathlib import Path
    folder=Path(root)/"PPMI"
    registry={p["pid"]:p for p in people}
    collected=defaultdict(lambda:defaultdict(list))
    coverage=Counter()
    for label,prefix,selector,low,high in SPECS:
        try:path=one_file(folder,prefix)
        except ValueError:
            coverage["optional_table_missing_"+label]+=1
            continue
        with path.open(encoding="utf-8-sig",errors="replace",newline="") as stream:
            reader=csv.DictReader(stream)
            headers=reader.fieldnames or []
            if len(headers)!=len(set(headers)):
                raise ValueError("Duplicate columns in "+path.name)
            if not {"PATNO","EVENT_ID"}.issubset(headers):
                raise ValueError("Missing patient/event keys in "+path.name)
            matches=(["MCATOT"] if label=="MoCA" else
                     ["TOTAL_CORRECT"] if label=="UPSIT" else
                     ["DRMVIVID","SLPINJUR"] if label=="RBD" else
                     [h for h in headers if re.fullmatch(selector,h)])
            matches=[col for col in matches if col in headers and
                     (label in ("MoCA","UPSIT","RBD") or
                      not col.endswith(("TOT","TOTAL")))]
            coverage["features_found_"+label]=len(matches)
            for row in reader:
                pid=str(row.get("PATNO","")).strip()
                if pid not in registry or row.get("EVENT_ID")!="BL":
                    continue
                month_raw=row.get("EXAMDT") or row.get("INFODT")
                if month_raw and str(month_raw).strip():
                    measured_month=parse_month(month_raw)
                    if measured_month is None or measured_month!=registry[pid]["baseline_month"]:
                        coverage["off_month_"+label]+=1
                        continue
                for field in matches:
                    val=numeric(row.get(field,""),low,high)
                    if val is not None:collected[pid][label+"_"+field].append(val)
    output={}
    for person in people:
        pid=person["pid"]
        items={}
        baseline=person["visits"][0]
        for j,name in enumerate(ALL):
            if baseline["status"][j]==1:
                items["BL_"+name]=int(baseline["score"][j])
        for key,values in collected.get(pid,{}).items():
            unique=set(values)
            if len(unique)==1:items[key]=values[0]
            else:coverage["conflicting_scalar_value"]+=1
        remote=[v for v in person["visits"][1:]
                if v["event"]=="R01" and
                4<=v["month"]-person["baseline_month"]<=8]
        if len(remote)==1:
            for j,name in enumerate(VISIBLE):
                i=ALL.index(name)
                if remote[0]["status"][i]==1:
                    items["R01_"+name]=int(remote[0]["score"][i])
        output[pid]=items
        for k in items:coverage["available_"+k]+=1
    return output,{k:v for k,v in sorted(coverage.items())
                   if not k.startswith("available_")},{
                       k:v for k,v in coverage.items() if k.startswith("available_")}


def panel_items(panel,items):
    """Five declared information sets; uses true R01 only when before target."""
    if panel not in range(5):raise ValueError("Unknown observation panel")
    selected=[]
    for key in items:
        if key.startswith("R01_"):
            if panel in (2,4):selected.append(key)
        elif key.startswith("BL_"):
            if key[3:] in MOTOR or panel in (3,4):
                selected.append(key)
        elif panel in (1,2,3,4):
            selected.append(key)
    return tuple(sorted(selected))


def episode_rows(patients,truth,evidence,cohort,horizon):
    """Keep ALL eligible outcomes, even with sparse/missing baseline tests."""
    mapping={p["pid"]:p for p in patients if p["cohort"]==cohort}
    rows=[]
    for event in truth:
        pid=event["pid"]
        if pid not in mapping:continue
        observed=evidence[pid]
        # Domain guard: R01 only before earliest 12m target window.
        if horizon<12 and any(k.startswith("R01_") for k in observed):
            observed={k:v for k,v in observed.items()
                      if not k.startswith("R01_")}
        rows.append({"pid":pid,"cohort":cohort,"horizon":horizon,
                     "burden":int(event["target_burden"]),
                     "label":int(event["target_positive"]),
                     "measured":observed})
    return rows


class PatientTool:
    """Exactly recorded item-score lookup. No mock/imagination is releasable."""
    def __init__(self,measured,allowed,budget):
        self._values={k:measured[k] for k in allowed if k in measured}
        self.available=frozenset(self._values)
        self.observed={}
        self.budget=int(budget)

    def request(self,item):
        if (item not in self.available or item in self.observed or
            len(self.observed)>=self.budget):
            raise PermissionError("Forbidden, unmeasured, repeated or over-budget")
        value=self._values[item]
        self.observed[item]=value
        return {"item":item,"recorded_value":value,
                "provenance":"actually_measured_BL_or_pre_target_R01"}
