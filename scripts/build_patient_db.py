#!/usr/bin/env python3
"""Create a local, unvalidated Parkinson research DB (patient-level; DO NOT upload).

Read-only source CSVs; provenance, duplicate exams and quality flags are retained.
Labels and observations are separate. No automatic SC->BL temporal merge.
Run: python scripts/build_patient_db.py --data-root /scratch/ylong66/data/pks/data \
  --db-path outputs/patient_db/pks.duckdb
Dependency: pip install duckdb. For standard-library fallback use
--engine sqlite --db-path outputs/patient_db/pks.sqlite
"""
import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

TABLES = {
 "PPMI": {
  "anchor": ("Participant_Status_", "COHORT"),
  "diagnosis": ("Primary_Research_Diagnosis_", "PRIMDIAG"),
  "clinical": [
   ("Age_at_visit_", "demographic", {"AGE_AT_VISIT":"age_at_visit"}),
   ("MDS-UPDRS_Part_III_", "motor_exam",
    {"NP3TOT":"updrs3_total","NP3GAIT":"updrs3_gait","NP3BRADY":"updrs3_bradykinesia"}),
   ("Montreal_Cognitive_Assessment__MoCA__", "cognitive_exam", {"MCATOT":"moca_total"}),
   ("University_of_Pennsylvania_Smell_Identification_Test_UPSIT_", "smell_exam",
    {"TOTAL_CORRECT":"upsit_total"}),
   ("REM_Sleep_Behavior_Disorder_Screening_Questionnaire_", "sleep_questionnaire",
    {"DRMVIVID":"rbd_vivid_dreams"}),
   ("SCOPA-AUT_", "autonomic_questionnaire", {"SCAU1":"scopa_aut_item1"})
  ]
 },
 "BioFIND": {
  "anchor": ("Screening_Demographics_", "APPRDX"),
  "diagnosis": ("Primary_Diagnosis_", "PRIMDIAG"),
  "clinical": [
   ("MDS_UPDRS_Part_III__Post_Dose__", "motor_exam_post_dose",
    {"NP3GAIT":"updrs3_gait_post_dose","NP3BRADY":"updrs3_bradykinesia_post_dose"}),
   ("Montreal_Cognitive_Assessment__MoCA__", "cognitive_exam", {"MCATOT":"moca_total"}),
   ("REM_Sleep_Disorder_Questionnaire_", "sleep_questionnaire",
    {"DRMVIVID":"rbd_vivid_dreams"})
  ]
 }
}
MISSING={"","nan","na","n/a","null","none","missing","."}
SUSPECT_MISSING={"-1","-7","-8","-9","-99","999","9999"}
def good(x):return x is not None and str(x).strip().lower() not in MISSING

def find_file(folder,prefix):
 pattern=re.compile(re.escape(prefix)+r"\d{2}[A-Za-z]{3}\d{4}\.csv$")
 files=[p for p in folder.glob(prefix+"*.csv") if pattern.fullmatch(p.name)]
 if len(files)!=1:raise ValueError(f"Expected 1 file for {prefix}, got {len(files)}")
 return files[0]

def iter_rows(file):
 """Yield (line, getter, header) preserving duplicated raw CSV headers."""
 with file.open(encoding="utf-8-sig",errors="replace",newline="") as f:
  reader=csv.reader(f);headers=[x.strip() for x in next(reader,[])]
  positions=defaultdict(list)
  for i,col in enumerate(headers):positions[col].append(i)
  for num,row in enumerate(reader,start=2):
   def get(col):
    values=[row[i].strip() for i in positions.get(col,[]) if i<len(row)]
    nonblank=[v for v in values if good(v)]
    if len(set(nonblank))>1:raise ValueError("conflicting duplicate column")
    return nonblank[0] if nonblank else ""
   yield num,get,positions

def subject_codes(file,field,only_bl,errors):
 found=defaultdict(set)
 for _,get,columns in iter_rows(file):
  if "PATNO" not in columns or field not in columns:raise ValueError(f"Missing column in {file.name}")
  try:
   p,code=get("PATNO"),get(field)
   if only_bl and get("EVENT_ID")!="BL":continue
  except ValueError:
   errors["ambiguous_label_or_key"]+=1;continue
  if good(p) and good(code):found[p].add(code)
 return {p:next(iter(v)) if len(v)==1 else None for p,v in found.items()}

def candidate_label(cohort,enrollment,diagnosis):
 if cohort=="PPMI":
  if enrollment=="1" and diagnosis=="1":return "PD"
  if enrollment=="2" and diagnosis=="17":return "HC"
 if cohort=="BioFIND":
  if enrollment=="1" and diagnosis=="01":return "PD"
  if enrollment=="2" and diagnosis=="17":return "HC"
 return None

def parse_exam_time(text):
 """Return (ISO day, ISO month, precision); MM/YYYY is provisional only.

 Never assign a fictitious day of month. Calendar month strings still
 require verification against the original codebook/export conventions.
 """
 raw=str(text).strip() if text is not None else ""
 if not good(raw):return None,None,"missing"
 match=re.fullmatch(r"(\d{2})/(\d{4})",raw)
 if match:
  month,year=map(int,match.groups())
  if 1<=month<=12 and 1900<=year<=2100:
   return None,f"{year:04d}-{month:02d}","month_candidate"
  return None,None,"unknown"
 for fmt in ("%Y-%m-%d","%m/%d/%Y","%d-%b-%Y","%Y/%m/%d",
             "%Y-%m-%d %H:%M:%S","%Y-%m-%dT%H:%M:%S"):
  try:
   day=datetime.strptime(raw,fmt).date()
   return day.isoformat(),day.strftime("%Y-%m"),"day"
  except (ValueError,TypeError):pass
 try:
  day=datetime.fromisoformat(raw.replace("Z","+00:00")).date()
  return day.isoformat(),day.strftime("%Y-%m"),"day"
 except ValueError:
  return None,None,"unknown"

def exam_date(text):
 """Legacy compatibility; for month-only dates returns None, never day=1."""
 return parse_exam_time(text)[0]

def connect(db,engine):
 if engine=="duckdb":
  try:import duckdb
  except ImportError as err:raise RuntimeError("Install duckdb: pip install duckdb") from err
  return duckdb.connect(str(db))
 return sqlite3.connect(str(db))

def schema(db):
 db.execute("CREATE TABLE patients (cohort VARCHAR, patient_id VARCHAR, PRIMARY KEY(cohort,patient_id))")
 db.execute("""CREATE TABLE restricted_labels (
 cohort VARCHAR, patient_id VARCHAR, enrollment_code VARCHAR, baseline_diagnosis_code VARCHAR,
 candidate_label VARCHAR, clinically_validated BOOLEAN, PRIMARY KEY(cohort,patient_id))""")
 db.execute("""CREATE TABLE feature_catalog (
 cohort VARCHAR, modality VARCHAR, concept VARCHAR, raw_column VARCHAR, patient_must_request BOOLEAN,
 PRIMARY KEY(cohort,concept))""")
 db.execute("""CREATE TABLE observations (
 cohort VARCHAR, patient_id VARCHAR, event_id VARCHAR, modality VARCHAR, concept VARCHAR,
 raw_column VARCHAR, value_raw VARCHAR, exam_date VARCHAR, exam_month VARCHAR,
 date_status VARCHAR, date_precision VARCHAR, date_source VARCHAR,
 quality_status VARCHAR, medication_state VARCHAR, treatment_status VARCHAR,
 imputed_flag VARCHAR, upsit_source VARCHAR, source_file VARCHAR, source_row INTEGER)""")
 db.execute("""CREATE TABLE visits (
 cohort VARCHAR, patient_id VARCHAR, event_id VARCHAR, observation_rows BIGINT,
 PRIMARY KEY(cohort,patient_id,event_id))""")
 db.execute("CREATE TABLE build_metadata (name VARCHAR PRIMARY KEY,value VARCHAR)")

def put_batches(db,sql,iterator,size=2500):
 batch=[]
 for row in iterator:
  batch.append(row)
  if len(batch)>=size:db.executemany(sql,batch);batch=[]
 if batch:db.executemany(sql,batch)

def build(root,db_path,engine,cohorts,overwrite=False):
 root=root.expanduser().resolve();db_path=db_path.expanduser().resolve()
 if not root.is_dir():raise ValueError("Missing raw data root")
 if db_path==root or root in db_path.parents:raise ValueError("Database cannot be placed inside raw data root")
 if db_path.exists() and not overwrite:raise FileExistsError("Database exists; choose new path or --overwrite")
 sources={}
 for cohort in cohorts:
  spec=TABLES[cohort];folder=root/cohort
  sources[cohort]={
   "anchor":find_file(folder,spec["anchor"][0]),
   "diagnosis":find_file(folder,spec["diagnosis"][0]),
   "clinical":[(find_file(folder,p),mode,concepts) for p,mode,concepts in spec["clinical"]]
  }
 # Delete only after every required input is found.
 db_path.parent.mkdir(parents=True,exist_ok=True)
 if db_path.exists():db_path.unlink()
 db=connect(db_path,engine)
 counts=Counter();errors=Counter()
 try:
  schema(db)
  for cohort in cohorts:
   spec=TABLES[cohort]
   anchors=subject_codes(sources[cohort]["anchor"],spec["anchor"][1],False,errors)
   diagnoses=subject_codes(sources[cohort]["diagnosis"],spec["diagnosis"][1],True,errors)
   people=[];labels=[]
   for pid,code in anchors.items():
    label=candidate_label(cohort,code,diagnoses.get(pid))
    people.append((cohort,pid))
    labels.append((cohort,pid,code,diagnoses.get(pid),label,False))
    counts[f"{cohort}_patients"]+=1
    if label:counts[f"{cohort}_candidate_{label}"]+=1
   db.executemany("INSERT INTO patients VALUES (?,?)",people)
   db.executemany("INSERT INTO restricted_labels VALUES (?,?,?,?,?,?)",labels)
   for path,mode,concepts in sources[cohort]["clinical"]:
    db.executemany("INSERT INTO feature_catalog VALUES (?,?,?,?,?)",
                   [(cohort,mode,concept,raw,True) for raw,concept in concepts.items()])
    def stream():
     for row_number,get,columns in iter_rows(path):
      try:
       pid,event=get("PATNO"),get("EVENT_ID")
      except ValueError:
       errors["conflicting_patient_or_visit_columns"]+=1;continue
      if pid not in anchors:continue
      for raw,concept in concepts.items():
       if raw not in columns:
        errors[f"missing_{cohort}_{concept}"]+=1;continue
       try:
        value=get(raw)
        if not good(value):continue
        imputed=get("IMPUTED_DATA")
        med=get("PDSTATE")
        treatment=get("PDTRTMNT")
        origin=get("UPSIT_SOURCE")
        raw_exam=get("EXAMDT") if mode.startswith("motor_exam") and get("EXAMDT") else ""
        time_source="EXAMDT" if raw_exam else "INFODT"
        raw_exam=raw_exam or get("INFODT")
        day,month,precision=parse_exam_time(raw_exam)
       except ValueError:
        errors["conflicting_observation_column"]+=1;continue
       quality=("possible_missing_sentinel" if value in SUSPECT_MISSING else
                "imputed" if imputed.lower() in ("1","yes","true") else "observed_unverified")
       counts[f"{cohort}_{concept}_observations"]+=1
       yield (cohort,pid,event,mode,concept,raw,value,day,month,
              "parseable" if precision=="day" else precision,
              precision,time_source,
              quality,med or None,treatment or None,imputed or None,
              origin or None,path.name,row_number)
    put_batches(db,"INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",stream())
  db.execute("""INSERT INTO visits SELECT cohort,patient_id,COALESCE(event_id,''),COUNT(*)
             FROM observations GROUP BY cohort,patient_id,event_id""")
  db.execute("INSERT INTO build_metadata VALUES (?,?)",
             ["status","unvalidated_review_only"])
  db.execute("INSERT INTO build_metadata VALUES (?,?)",
             ["schema_version","2_month_precision"])
  db.execute("INSERT INTO build_metadata VALUES (?,?)",
             ["created_at_utc",datetime.now(timezone.utc).isoformat()])
  db.commit()
  return {"engine":engine,"database":str(db_path),"status":"unvalidated_review_only",
          "counts":dict(sorted(counts.items())),"exceptions":dict(sorted(errors.items()))}
 finally:db.close()

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--data-root",type=Path,required=True)
 parser.add_argument("--db-path",type=Path,default=Path("outputs/patient_db/pks.duckdb"))
 parser.add_argument("--engine",choices=("duckdb","sqlite"),default="duckdb")
 parser.add_argument("--cohorts",nargs="+",choices=tuple(TABLES),default=list(TABLES))
 parser.add_argument("--overwrite",action="store_true")
 a=parser.parse_args()
 try:result=build(a.data_root,a.db_path,a.engine,a.cohorts,a.overwrite)
 except (OSError,ValueError,RuntimeError) as e:parser.error(str(e))
 print(json.dumps(result,ensure_ascii=False,indent=2))
 print("Patient-level database must remain on the authorized cluster. DO NOT upload it.")
if __name__=="__main__":main()
