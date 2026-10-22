#!/usr/bin/env python3
"""Aggregate candidate PD/HC label counts and joint measurements; no patient records."""
import argparse,csv,json,sqlite3
from pathlib import Path

COMBOS={
 "PPMI":[("updrs3_total","moca_total"),
         ("updrs3_total","moca_total","rbd_vivid_dreams"),
         ("updrs3_total","moca_total","rbd_vivid_dreams","scopa_aut_item1"),
         ("updrs3_total","moca_total","rbd_vivid_dreams","upsit_total")],
 "BioFIND":[("updrs3_gait_post_dose","moca_total"),
            ("updrs3_gait_post_dose","moca_total","rbd_vivid_dreams")]}

def open_db(path,engine):
 if engine=="duckdb":
  import duckdb
  return duckdb.connect(str(path),read_only=True)
 return sqlite3.connect("file:"+str(path.resolve())+"?mode=ro",uri=True)

def group_coverage(db,cohort,concepts,view,strict_quality):
 event="AND o.event_id='BL'" if view=="BL" else "AND o.event_id IN ('SC','BL')"
 quality="AND o.quality_status='observed_unverified'" if strict_quality else ""
 clauses=[]
 params=[cohort]
 for concept in concepts:
  clauses.append("""EXISTS(SELECT 1 FROM observations o WHERE o.cohort=l.cohort
    AND o.patient_id=l.patient_id AND o.concept=? """+event+" "+quality+")")
  params.append(concept)
 query="""SELECT l.candidate_label,COUNT(*) FROM restricted_labels l
          WHERE l.cohort=? AND l.candidate_label IN ('PD','HC') AND """ + " AND ".join(clauses) + """
          GROUP BY l.candidate_label"""
 data=dict(db.execute(query,params).fetchall())
 return data.get("PD",0),data.get("HC",0)

def summarize(dbpath,out,engine):
 out.mkdir(parents=True,exist_ok=True)
 db=open_db(dbpath,engine)
 try:
  labels=db.execute("""SELECT cohort,candidate_label,COUNT(*) FROM restricted_labels
      WHERE candidate_label IN ('PD','HC') GROUP BY cohort,candidate_label
      ORDER BY cohort,candidate_label""").fetchall()
  rows=[]
  for cohort,combinations in COMBOS.items():
   for combo in combinations:
    for view in ("BL","SC_BL_undated_upper_bound"):
     for require_quality in (True,False):
      pd,hc=group_coverage(db,cohort,combo,view,require_quality)
      rows.append({"cohort":cohort,"view":view,"features":"+".join(combo),
                   "exclude_imputed_and_suspect_sentinels":require_quality,
                   "PD":pd,"HC":hc,"total":pd+hc})
  with (out/"candidate_joint_coverage.csv").open("w",newline="",encoding="utf-8") as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  summary={"labels":[{"cohort":c,"label":label,"patients":n} for c,label,n in labels],
           "aggregate_coverage_rows":len(rows),"research_status":"unvalidated_review_only",
           "warnings":[
             "SC_BL_undated_upper_bound does not verify chronology; not for model training.",
             "BL data also not yet proven to precede diagnostic decision.",
             "Repeated exams have not been resolved; presence is not a usable single value.",
             "Output is aggregate; database itself contains patient-level data and must not be uploaded."]}
  (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
  return summary
 finally:db.close()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--db-path",type=Path,required=True)
 p.add_argument("--engine",choices=("duckdb","sqlite"),default="duckdb")
 p.add_argument("--output-dir",type=Path,default=Path("outputs/patient_db_report"))
 a=p.parse_args()
 if not a.db_path.is_file():p.error("Database not found")
 result=summarize(a.db_path.resolve(),a.output_dir.resolve(),a.engine)
 print(json.dumps(result,indent=2,ensure_ascii=False))
 print("Only upload output reports, NOT the DB.")
if __name__=="__main__":main()
