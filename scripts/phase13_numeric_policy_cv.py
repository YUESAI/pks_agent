#!/usr/bin/env python3
"""Phase 13: numerical-evidence baseline on frozen PPMI TRAIN folds only.

Never access held-out val/test/BioFIND. Local DBs contain patient records.
Reports contain aggregate metrics only; no clinical validity is claimed.
"""
import argparse,csv,hashlib,json,math,random,sqlite3,sys
from pathlib import Path
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase10_leakage_splits import open_source,auc
from scripts.phase11_protocol_preflight import assign_folds
from scripts.budgeted_env_v0 import ACTION_SPECS

BINS={"age":(40,50,60,70,80),"sleep_item":(0,1,2,3),
 "cognitive_test":(17,22,25,28),"autonomic_item":(0,1,2,3),
 "motor_exam":(5,10,20,30,40,60),"smell_test":(10,20,30)}
LIMITS={"age":(0,120),"sleep_item":(0,20),"cognitive_test":(0,30),
 "autonomic_item":(0,20),"motor_exam":(0,132),"smell_test":(0,40)}
ORDERS={"clinical":("age","sleep_item","autonomic_item","cognitive_test","motor_exam","smell_test"),
 "smell_first":("age","smell_test","motor_exam","cognitive_test","sleep_item","autonomic_item")}
def bin_value(action,v):
 try:x=float(v)
 except (TypeError,ValueError):return None
 low,high=LIMITS[action]
 return sum(x>edge for edge in BINS[action]) if math.isfinite(x) and low<=x<=high else None
def train_model(records):
 model={};success={}
 for a in ACTION_SPECS:
  counts={"PD":Counter(),"HC":Counter()};n=0
  for label,anchor,values in records:
   b=bin_value(a,values.get(a))
   if b is not None:counts[label][b]+=1;n+=1
  width=len(BINS[a])+1
  model[a]={l:[(counts[l][i]+1)/(sum(counts[l].values())+width)
               for i in range(width)] for l in ("PD","HC")}
  success[a]=(n+1)/(len(records)+2)
 return model,success
def entropy(p):
 return -p*math.log(p)-(1-p)*math.log(1-p) if 0<p<1 else 0.
def gain(probs,score):
 p=1/(1+math.exp(-max(-500,min(500,score))))
 return entropy(p)-sum((p*x+(1-p)*y)*entropy(p*x/(p*x+(1-p)*y))
   for x,y in zip(probs["PD"],probs["HC"]) if p*x+(1-p)*y>0)
def policy_episode(private_values,model,success,policy,budget,seed,pid):
 score=0.;spent=0;asked=set();returned=0
 def request(a):
  nonlocal score,spent,returned
  asked.add(a);spent+=ACTION_SPECS[a]["cost"]
  b=bin_value(a,private_values.get(a))
  if b is not None:
   score+=math.log(model[a]["PD"][b]/model[a]["HC"][b]);returned+=1
 request("age")
 if policy=="adaptive":
  while True:
   options=[a for a in ACTION_SPECS if a not in asked
            and ACTION_SPECS[a]["cost"]<=budget-spent]
   if not options:break
   options.sort(key=lambda a:(-gain(model[a],score)*success[a]/
                             max(1,ACTION_SPECS[a]["cost"]),a))
   request(options[0])
 else:
  if policy=="random":
   order=[a for a in ACTION_SPECS if a!="age"]
   h=hashlib.sha256((str(seed)+"|"+pid).encode()).digest()
   random.Random(int.from_bytes(h[:8],"big")).shuffle(order)
  else:order=ORDERS[policy]
  for a in order:
   if a not in asked and ACTION_SPECS[a]["cost"]<=budget-spent:request(a)
 return score,spent,returned
def load_training(source,engine,manifest,splitdb):
 with sqlite3.connect("file:"+str(splitdb)+"?mode=ro",uri=True) as d:
  ids=sorted(pid for c,pid,s in d.execute("SELECT cohort,patient_id,split FROM patient_splits")
             if c=="PPMI" and s=="train")
 if not ids:raise ValueError("No PPMI TRAIN patients")
 man=sqlite3.connect("file:"+str(manifest)+"?mode=ro",uri=True)
 db=open_source(source,engine);out={}
 try:
  for offset in range(0,len(ids),200):
   chunk=ids[offset:offset+200];marks=",".join("?" for _ in chunk)
   labels=dict(db.execute("SELECT patient_id,candidate_label FROM restricted_labels WHERE cohort='PPMI' AND patient_id IN ("+marks+")",chunk).fetchall())
   anchors=dict(man.execute("SELECT patient_id,baseline_month FROM episode_candidates WHERE cohort='PPMI' AND patient_id IN ("+marks+")",chunk).fetchall())
   manifest_rows=defaultdict(dict)
   for pid,key,status,visit,file,row in man.execute("SELECT patient_id,action_key,status,selected_visit,source_file,source_row FROM episode_actions WHERE cohort='PPMI' AND patient_id IN ("+marks+")",chunk):
    manifest_rows[pid][key]=(status,visit,file,row)
   observations={}
   for pid,visit,concept,file,row,value,quality in db.execute("SELECT patient_id,event_id,concept,source_file,source_row,value_raw,quality_status FROM observations WHERE cohort='PPMI' AND patient_id IN ("+marks+")",chunk).fetchall():
    observations[(pid,visit,concept,file,row)]=(value,quality)
   for pid in chunk:
    if labels.get(pid) not in ("PD","HC") or pid not in anchors:
     raise ValueError("Missing training label/episode")
    values={}
    for action,spec in ACTION_SPECS.items():
     for internal in spec["internal"]:
      status,visit,file,row=manifest_rows[pid].get(internal,(None,None,None,None))
      if status!="ready":continue
      item=observations.get((pid,visit,internal.split("@")[0],file,row))
      if not item or item[1]!="observed_unverified":
       raise ValueError("Provenance mismatch")
      values[action]=item[0];break
    out[pid]=(labels[pid],bool(anchors[pid]),values)
 finally:man.close();db.close()
 return out
def evaluate(scored):
 n=len(scored);pd=sum(y for s,y,c,r in scored);hc=n-pd
 tp=sum(s>=0 and y for s,y,c,r in scored)
 tn=sum(s<0 and not y for s,y,c,r in scored)
 fp=hc-tn;fn=pd-tp
 f1pd=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
 f1hc=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0
 return dict(n=n,n_PD=pd,n_HC=hc,
  balanced_accuracy=round((tp/pd+tn/hc)/2,4),
  macro_f1=round((f1pd+f1hc)/2,4),
  auroc=round(auc([(s,y) for s,y,c,r in scored]),4),
  mean_cost=round(sum(c for s,y,c,r in scored)/n,3),
  mean_results=round(sum(r for s,y,c,r in scored)/n,3))
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ("source-db","manifest-db","split-db"):p.add_argument("--"+k,type=Path,required=True)
 p.add_argument("--source-engine",choices=("duckdb","sqlite"),default="duckdb")
 p.add_argument("--report-dir",type=Path,default=Path("outputs/phase13_policy_cv"))
 p.add_argument("--folds",type=int,default=5);p.add_argument("--budget",type=int,default=5)
 p.add_argument("--seed",type=int,default=20261018);args=p.parse_args()
 if args.folds<2 or args.budget<0:p.error("Invalid folds or budget")
 data=load_training(args.source_db.resolve(),args.source_engine,
                    args.manifest_db.resolve(),args.split_db.resolve())
 examples=[(pid,l,anchor,values) for pid,(l,anchor,values) in sorted(data.items())]
 assignments=assign_folds(examples,args.folds,args.seed)
 records=defaultdict(list)
 for fold in range(args.folds):
  training=[(label,anchor,values) for pid,label,anchor,values in examples if assignments[pid]!=fold]
  held=[(pid,label,values) for pid,label,anchor,values in examples if assignments[pid]==fold]
  if {label for label,anchor,values in training}!={"PD","HC"} or not held:
   raise ValueError("Invalid fold")
  model,success=train_model(training)
  for policy in ("clinical","smell_first","random","adaptive"):
   for pid,label,values in held:
    score,cost,returned=policy_episode(values,model,success,policy,args.budget,args.seed,pid)
    records[policy].append((score,label=="PD",cost,returned))
 out=args.report_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 fields=["policy","split","n","n_PD","n_HC","balanced_accuracy","macro_f1","auroc","mean_cost","mean_results"]
 with (out/"policy_numeric_cv.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for policy,items in records.items():
   w.writerow(dict(policy=policy,split="PPMI_train_out_of_fold",**evaluate(items)))
 summary={"status":"train_only_exploratory_unvalidated","n_train":len(examples),
          "folds":args.folds,"budget":args.budget,
          "notes":["No PPMI val/test or BioFIND observations accessed.",
           "Absent tests do not update predictions.",
           "Motor value mixes original treatment conditions; research confounding remains.",
           "Numeric bins are provisional, not medically validated.",
           "Only aggregate reports may be shared."]}
 (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
 print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
