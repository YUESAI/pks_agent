#!/usr/bin/env python3
"""Phase48: TRAIN-only masked Transformer observation WM, vs naive Bayes.

Learns p(y|observed scores) and p(unobserved score|y,observed scores).
Observed zero differs from missing. A neural probability is NOT a
measured clinical fact. This is SAME-V04 observation modelling, not a
longitudinal/action-conditioned Parkinson disease state-transition WM.
"""
from __future__ import annotations
import argparse,csv,json,math,sys
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.nn import functional as F
from scripts.phase45_world_model import ObservationWorldModel,entropy
from scripts.phase45_wm_rsi_agents import load_train,hashkey,measure
from scripts.phase47_question_budget_curve import patient_split

BUDGETS=(4,8,12,16)


class MaskedItemTransformer(nn.Module):
    """Masked item tokens -> posterior for y and conditional item scores."""

    def __init__(self,n_items,width=96,heads=4,depth=2,dropout=.15):
        super().__init__()
        if n_items<2 or width%heads:raise ValueError("Invalid architecture")
        self.n_items=n_items
        self.item_embeddings=nn.Embedding(n_items,width)
        self.value_embeddings=nn.Embedding(6,width) # scores 0-4; masked=5
        self.cls=nn.Parameter(torch.zeros(1,1,width))
        self.encoder=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,nhead=heads,dim_feedforward=width*2,
                activation="gelu",dropout=dropout,batch_first=True,
                norm_first=False),
            num_layers=depth,enable_nested_tensor=False)
        self.target_head=nn.Linear(width,1)
        self.item_head=nn.Linear(width,10) # label=0/1 and value=0..4
        nn.init.normal_(self.cls,std=.02)

    def forward(self,measured,available):
        if measured.ndim!=2 or measured.shape!=available.shape or measured.shape[1]!=self.n_items:
            raise ValueError("Expected [batch, item] tensors")
        batch,dim=measured.shape
        indices=torch.arange(dim,device=measured.device)
        tokens=self.item_embeddings(indices)[None]+self.value_embeddings(
            torch.where(available,measured,torch.full_like(measured,5)))
        tokens=torch.cat([self.cls.expand(batch,-1,-1),tokens],dim=1)
        encoded=self.encoder(tokens)
        return (self.target_head(encoded[:,0]).squeeze(-1),
                self.item_head(encoded[:,1:]).reshape(batch,dim,2,5))


def masked_batch(x,options=(0,2,4,8,12,16,20,27),generator=None):
    batch,dim=x.shape
    allowed=torch.tensor([q for q in options if 0<=q<=dim],dtype=torch.long)
    if not len(allowed):raise ValueError("Empty mask budgets")
    counts=allowed[torch.randint(0,len(allowed),(batch,),generator=generator)]
    perm=torch.rand(batch,dim,generator=generator).argsort(dim=1)
    return perm.argsort(dim=1)<counts[:,None]


def multitask_loss(output,inputs,labels,available,item_weight=.5):
    binary_logits,conditioned=output
    label=labels.long()
    binary=F.binary_cross_entropy_with_logits(binary_logits,labels.float())
    scores=conditioned[
        torch.arange(len(inputs),device=inputs.device),:,label,:]
    nll=F.cross_entropy(scores.reshape(-1,5),inputs.reshape(-1),
                        reduction="none").reshape(inputs.shape)
    hidden=(~available).float()
    reconstruction=(nll*hidden).sum()/hidden.sum().clamp_min(1)
    return binary+item_weight*reconstruction


def split_inner(records,seed,dev_fraction=.2):
    fitting=[];dev=[]
    for label in (0,1):
        group=sorted((r for r in records if r["label"]==label),
                     key=lambda r:hashkey(seed,r["pid"]))
        count=max(1,round(len(group)*dev_fraction))
        dev+=group[:count];fitting+=group[count:]
    if len(fitting)<24 or len(dev)<12:
        raise ValueError("Insufficient TRAIN inner-fit/development subjects")
    if set(r["pid"] for r in dev)&set(r["pid"] for r in fitting):
        raise AssertionError("Patient leakage")
    return fitting,dev


def dataset(records,items):
    x=torch.tensor([[int(r["features"][item]) for item in items]
                    for r in records],dtype=torch.long)
    y=torch.tensor([r["label"] for r in records],dtype=torch.float32)
    if (x<0).any() or (x>4).any():raise ValueError("Invalid Motor score")
    return x,y


def train_one(records,items,seed,device,epochs=60,width=96,depth=2,
              learning_rate=8e-4,patience=10,batch_size=64):
    torch.manual_seed(seed)
    if device.type=="cuda":torch.cuda.manual_seed_all(seed)
    rng=torch.Generator().manual_seed(seed+314)
    fitting,dev=split_inner(records,seed)
    x,y=dataset(fitting,items)
    xd,yd=dataset(dev,items)
    model=MaskedItemTransformer(len(items),width=width,depth=depth).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,
                                weight_decay=.02)
    dev_mask=masked_batch(xd,options=(0,4,8,12,16,len(items)),
        generator=torch.Generator().manual_seed(seed+731)).to(device)
    best=float("inf");state=None;best_epoch=0;stale=0
    for epoch in range(1,epochs+1):
        model.train()
        order=torch.randperm(len(x),generator=rng)
        for idx in order.split(batch_size):
            xi=x[idx].to(device);yi=y[idx].to(device)
            mask=masked_batch(x[idx],generator=rng).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=multitask_loss(model(xi,mask),xi,yi,mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            devloss=float(multitask_loss(model(xd.to(device),dev_mask),
                               xd.to(device),yd.to(device),dev_mask).cpu())
        if devloss<best-1e-5:
            best=devloss;best_epoch=epoch;stale=0
            state={k:v.detach().cpu().clone()
                   for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=patience:break
    if state is None:raise RuntimeError("Missing trained checkpoint")
    model.load_state_dict(state);model.eval()
    return model,{"seed":seed,"fit_patients":len(fitting),
                  "inner_dev_patients":len(dev),"best_epoch":best_epoch,
                  "best_inner_dev_loss":round(best,6)}


class NeuralObservationWM:
    """Ensemble; same query API as the original naive observation WM."""

    def __init__(self,items,models,device):
        if not models:raise ValueError("No trained ensemble models")
        self.items=tuple(items)
        self.index={item:i for i,item in enumerate(items)}
        self.models=models
        self.device=device
        self._cached_key=None
        self._cached=None

    def _infer(self,observations):
        if any(k not in self.index or isinstance(v,bool) or v not in range(5)
               for k,v in observations.items()):
            raise ValueError("Invalid or unavailable observation")
        key=tuple(sorted(observations.items()))
        if key==self._cached_key:return self._cached
        x=torch.zeros((1,len(self.items)),device=self.device,dtype=torch.long)
        mask=torch.zeros_like(x,dtype=torch.bool)
        for name,score in observations.items():
            i=self.index[name];x[0,i]=score;mask[0,i]=True
        joints=[]
        with torch.no_grad():
            for model in self.models:
                ylabel,items=model(x,mask)
                p=torch.sigmoid(ylabel)[0].clamp(1e-5,1-1e-5)
                cls=torch.stack((1-p,p))
                joints.append(items.softmax(-1)[0]*cls[None,:,None])
        joint=torch.stack(joints).mean(0).cpu() # [item,2,score]
        prior=float(joint[0].sum(-1)[1])
        self._cached_key=key
        self._cached=(prior,joint)
        return self._cached

    def posterior(self,observations):
        return self._infer(observations)[0]

    def predict_item(self,observations,name):
        if name not in self.index or name in observations:
            raise ValueError("Already acquired or invalid item")
        joint=self._infer(observations)[1][self.index[name]]
        return tuple(float(x) for x in joint.sum(0).tolist())

    def information_gain(self,observations,name):
        if name not in self.index or name in observations:
            raise ValueError("Already acquired or invalid item")
        p,joint=self._infer(observations)
        matrix=joint[self.index[name]]
        pvalue=matrix.sum(0)
        after=matrix[1]/pvalue.clamp_min(1e-8)
        return max(0.,float(entropy(p)-sum(
            float(pvalue[v])*entropy(float(after[v]))
            for v in range(5))))

    def static_order(self):
        return sorted(self.items,key=lambda k:(-self.information_gain({},k),k))


def negative_loglik(world,held,items,q,seed):
    """Equal random observed subsets for both WMs, heldout TRAIN only."""
    errors=[];risks={}
    for row in held:
        names=sorted(items,key=lambda k:hashkey(seed,row["pid"]+"|"+k))
        observed={k:row["features"][k] for k in names[:q]}
        risks[row["pid"]]=world.posterior(observed)
        for item in names[q:]:
            probabilities=world.predict_item(observed,item)
            value=int(row["features"][item])
            errors.append(-math.log(max(1e-8,probabilities[value])))
    return mean(errors),risks


def evaluate(records,items,out,ensemble=3,epochs=60,seed=20261018,
             limit=100,width=96,depth=2,device_choice="auto"):
    if device_choice=="auto":
        device_choice="cuda" if torch.cuda.is_available() else "cpu"
    device=torch.device(device_choice)
    if device.type=="cpu":torch.set_num_threads(min(4,torch.get_num_threads()))
    if not 1<=ensemble<=5 or not 2<=epochs<=200:
        raise ValueError("Invalid ensemble or epoch settings")
    fitting,held=patient_split(records,limit,seed)
    models=[];logs=[]
    for member in range(ensemble):
        model,meta=train_one(fitting,items,seed+1009*member,device,
                             epochs=epochs,width=width,depth=depth)
        models.append(model);meta["member"]=member;logs.append(meta)
    neural=NeuralObservationWM(items,models,device)
    naive=ObservationWorldModel(items).fit(fitting)
    metrics=[]
    for q in BUDGETS:
        for name,model in (("naive_bayes",naive),
                           ("masked_transformer",neural)):
            nll,risks=negative_loglik(model,held,items,q,seed)
            stat=measure(held,risks,seed,True);stat.pop("chosen")
            metrics.append({
                "model":name,"observed_items":q,
                "heldout_patients":len(held),
                "heldout_positives":sum(r["label"] for r in held),
                "masked_item_nll":round(nll,6),
                "outcome_brier":round(stat["brier"],6),
                "outcome_auroc":round(stat["auroc"],6),
                "topk_recall":round(stat["recall"],6),
                "hidden_burden_recall":round(stat["burden_recall"],6),
            })
    out=Path(out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    for path,rows in ((out/"wm_quality.csv",metrics),
                      (out/"ensemble_training.csv",logs)):
        with path.open("w",newline="",encoding="utf-8") as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    summary={
        "status":"TRAIN_only_same_visit_masked_Transformer_observation_WM",
        "train_world_model_patients":len(fitting),
        "heldout_train_patients":len(held),
        "heldout_positive":sum(r["label"] for r in held),
        "ensemble":ensemble,"epochs_max":epochs,
        "hidden_width":width,"depth":depth,
        "seed":seed,"device":str(device),
        "checkpoints":list(BUDGETS),
        "caveats":[
            "Same-visit observation model, not longitudinal disease dynamics.",
            "Masked score predictions are uncertainty distributions, not measured facts.",
            "Separate inner development selects checkpoint before TRAIN audit.",
            "Repeated exploratory TRAIN evaluation, no fresh independent claim.",
            "No Qwen, no full clinical-action WM and no RSI in this preflight.",
            "VAL and consumed Phase44 TEST remain unopened.",
            "No per-person scores, masks or labels exported."
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--ensemble",type=int,default=3)
    p.add_argument("--epochs",type=int,default=60)
    p.add_argument("--max-patients",type=int,default=100)
    p.add_argument("--seed",type=int,default=20261018)
    p.add_argument("--width",type=int,default=96)
    p.add_argument("--depth",type=int,default=2)
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    a=p.parse_args()
    root,manifest,out=[Path(x).expanduser().resolve()
                       for x in (a.data_root,a.manifest_db,a.report_dir)]
    if not (root/"PPMI").is_dir() or not manifest.is_file() or out==root or root in out.parents or out==manifest:
        raise ValueError("Missing PPMI inputs or unsafe output")
    records,items,qc,source=load_train(root,manifest)
    print(json.dumps(evaluate(records,items,out,a.ensemble,a.epochs,
                              a.seed,a.max_patients,a.width,a.depth,
                              a.device),indent=2))
    print("Upload only aggregate Phase48 report; do not send patient-level data.")


if __name__=="__main__":
    main()
