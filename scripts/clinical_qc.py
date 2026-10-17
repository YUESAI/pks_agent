#!/usr/bin/env python3
"""Read-only clinical QC: duplicate keys, coded frequencies, exam timing.
Exports only aggregate counts and official codebook entries. No patient IDs.
"""
import argparse,csv,json,re
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
TABLES={
"PPMI":{
 "diagnosis":("Primary_Research_Diagnosis_","PRIMDIAG"),
 "motor":("MDS-UPDRS_Part_III_","NP3TOT"),
 "upsit":("University_of_Pennsylvania_Smell_Identification_Test_UPSIT_","TOTAL_CORRECT"),
 "moca":("Montreal_Cognitive_Assessment__MoCA__","MCATOT"),
 "rbd":("REM_Sleep_Behavior_Disorder_Screening_Questionnaire_","DRMVIVID"),
 "scopa":("SCOPA-AUT_","SCAU1"),
 "enrollment":("Participant_Status_","COHORT")},
"BioFIND":{
 "diagnosis":("Primary_Diagnosis_","PRIMDIAG"),
 "motor_post_dose":("MDS_UPDRS_Part_III__Post_Dose__","NP3GAIT"),
 "moca":("Montreal_Cognitive_Assessment__MoCA__","MCATOT"),
 "rbd":("REM_Sleep_Disorder_Questionnaire_","DRMVIVID"),
 "enrollment":("Screening_Demographics_","APPRDX")}}
CODES={"COHORT","PRIMDIAG","APPRDX","PDSTATE","PDTRTMNT","IMPUTED_DATA","UPSIT_SOURCE"}
EMPTY={"","na","nan","n/a","null","none","."}
def ok(s):return s is not None and str(s).strip().lower() not in EMPTY
def find(folder,prefix):
 p=re.compile(re.escape(prefix)+r"\d{2}[A-Za-z]{3}\d{4}\.csv$")
 m=[f for f in folder.glob(prefix+"*.csv") if p.fullmatch(f.name)]
 if len(m)!=1:raise ValueError(f"Expected one file matching {prefix}; got {len(m)}")
 return m[0]
def date(s):
 for fmt in ("%Y-%m-%d","%m/%d/%Y","%d-%b-%Y","%Y/%m/%d"):
  try:return datetime.strptime(s,fmt).date()
  except (TypeError,ValueError):pass
 return None
def choose(row,positions,col):
 vals=[row[i].strip() for i in positions.get(col,[]) if i<len(row)]
 good=[v for v in vals if ok(v)]
 return (good[0] if good else ""),len(set(good))>1
def scan(path,field):
 stats=Counter();codes=defaultdict(Counter);patients=set();keys=Counter()
 visits=Counter();events=defaultdict(lambda:defaultdict(list));dups=[]
 with path.open(encoding="utf-8-sig",errors="replace",newline="") as h:
  rd=csv.reader(h);header=next(rd,[]);pos=defaultdict(list)
  for i,col in enumerate(header):pos[col.strip()].append(i)
  if "PATNO" not in pos or field not in pos:raise ValueError("missing PATNO or field")
  dups=[(k,len(v)) for k,v in pos.items() if len(v)>1]
  for row in rd:
   stats["rows"]+=1
   pid,bad=choose(row,pos,"PATNO");event,ebad=choose(row,pos,"EVENT_ID")
   if bad or ebad:stats["key_conflicts"]+=1;continue
   if not ok(pid):stats["missing_id"]+=1;continue
   patients.add(pid);keys[(pid,event)]+=1;visits[event]+=1
   raw,vbad=choose(row,pos,field)
   if vbad:stats["measurement_conflicts"]+=1;continue
   if ok(raw):
    stats["present_measurements"]+=1
    if raw in {"-1","-7","-8","-9","-99","999","9999"}:stats["possible_sentinel"]+=1
    if event in ("SC","BL"):
     rawdate,_=choose(row,pos,"INFODT")
     events[pid][event].append(date(rawdate))
   for c in CODES:
    if c in pos:
     v,bad=choose(row,pos,c)
     if not bad and ok(v) and len(v)<=48:codes[c][v]+=1
 stats["patients"]=len(patients);stats["repeated_patient_visit_keys"]=sum(n>1 for n in keys.values())
 stats["extra_rows_per_key"]=sum(n-1 for n in keys.values())
 return stats,visits,codes,events,dups
def dump(path,headers,rows):
 with path.open("w",newline="",encoding="utf-8") as f:
  writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader();writer.writerows(rows)
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--data-root",type=Path,required=True)
 ap.add_argument("--output-dir",type=Path,default=Path("outputs/clinical_qc"))
 ap.add_argument("--min-category-count",type=int,default=10)
 a=ap.parse_args();root=a.data_root.resolve();out=a.output_dir.resolve()
 if not root.is_dir() or out==root or root in out.parents or a.min_category_count<10:ap.error("invalid root/output/threshold")
 out.mkdir(parents=True,exist_ok=True)
 quality=[];header_dups=[];frequencies=[];timing=[];codebooks=[];issues=[]
 for cohort,spec in TABLES.items():
  baseline={};dates={}
  for name in ["diagnosis"]+[k for k in spec if k!="diagnosis"]:
   prefix,field=spec[name]
   try:
    path=find(root/cohort,prefix)
    stats,visits,codes,events,dup=scan(path,field)
   except (OSError,ValueError,csv.Error) as e:
    issues.append(dict(cohort=cohort,table=name,error=type(e).__name__));continue
   quality.append(dict(cohort=cohort,table=name,source_file=path.name,**stats))
   for col,n in dup:header_dups.append(dict(cohort=cohort,table=name,column=col,occurrences=n))
   for col,hist in codes.items():
    for code,n in hist.most_common():
     if n>=a.min_category_count:frequencies.append(dict(cohort=cohort,table=name,column=col,code=code,rows=n))
   if name=="diagnosis":
    baseline={pid:next(iter(d)) for pid,v in events.items()
              if len(d:={x for x in v.get("BL",[]) if x is not None})==1}
   else:dates[name]=events
  for name,people in dates.items():
   counts=Counter()
   for pid,evs in people.items():
    anchor=baseline.get(pid)
    for ev in ("BL","SC"):
     for measurement in evs.get(ev,[]):
      state=("no_baseline_date" if anchor is None else "no_measurement_date" if measurement is None
             else "before" if measurement<anchor else "after" if measurement>anchor else "same_day")
      counts[(ev,state)]+=1
   for (visit,state),count in counts.items():
    timing.append(dict(cohort=cohort,table=name,visit=visit,relative_to_BL_diagnosis_form=state,rows=count))
  try:
   p=find(root/cohort,"Code_List_-_Harmonized_" if cohort=="PPMI" else "Code_List_")
   with p.open(encoding="utf-8-sig",errors="replace",newline="") as f:
    for r in csv.DictReader(f):
     if str(r.get("ITM_NAME","")).upper() in CODES:
      codebooks.append(dict(cohort=cohort,module=r.get("MOD_NAME") or r.get("VIEWNAME") or "",
                            field=r["ITM_NAME"],code=r.get("CODE") or "",description=r.get("DECODE") or ""))
  except (OSError,ValueError,csv.Error) as e:issues.append(dict(cohort=cohort,table="codebook",error=type(e).__name__))
 dump(out/"table_quality.csv",["cohort","table","source_file"]+
      sorted(set(k for item in quality for k in item)-{"cohort","table","source_file"}),quality)
 dump(out/"duplicate_headers.csv",["cohort","table","column","occurrences"],header_dups)
 dump(out/"clinical_codes.csv",["cohort","table","column","code","rows"],frequencies)
 dump(out/"timing.csv",["cohort","table","visit","relative_to_BL_diagnosis_form","rows"],timing)
 dump(out/"official_codes.csv",["cohort","module","field","code","description"],codebooks)
 dump(out/"issues.csv",["cohort","table","error"],issues)
 summary=dict(tables=len(quality),issues=len(issues),duplicate_header_tables=len({(x["cohort"],x["table"]) for x in header_dups}),
 caveats=["Diagnostic codes require manual clinical validation.","BL diagnosis form dates are not verified decision times.",
 "Counts are aggregated; missing-sentinel codes and imputed measures need clinical decoding."])
 (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
 print(json.dumps(summary,indent=2));print("Reports:",out)
if __name__=="__main__":main()
