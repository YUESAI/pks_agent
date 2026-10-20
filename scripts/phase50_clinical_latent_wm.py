#!/usr/bin/env python3
"""Phase50: Dreamer/JEPA-inspired longitudinal latent WM, PPMI TRAIN only.

BL and observed 12m Motor -> nominal 24m Motor forecast. Patient-disjoint
nested CV, latent state transition, measured-observation belief update,
teacher-latent consistency, open-loop imagined rollout, strong baselines.
NOT a full Dreamer4 implementation nor a treatment causal world simulator.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,sys
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
LOGMAX=math.log1p(132.)

def fingerprint(seed,pid):
    return hashlib.sha256(f"phase50|{seed}|{pid}".encode()).hexdigest()

def folds_for(rows,n,seed):
    if not 2<=n<=5:raise ValueError("Invalid folds")
    ids=sorted((r["pid"] for r in rows),key=lambda p:fingerprint(seed,p))
    if len(ids)!=len(set(ids)) or len(ids)<n*12:raise ValueError("Duplicate / too few patients")
    return {p:i%n for i,p in enumerate(ids)}

def dev_split(rows,seed):
    ranked=sorted(rows,key=lambda r:fingerprint(seed,r["pid"]))
    k=max(12,round(.2*len(ranked)))
    if len(ranked)-k<24:raise ValueError("Insufficient fit patients")
    return ranked[k:],ranked[:k]

def batch(rows,device):
    for r in rows:
        if not (0<=r["bl"]<=132 and 0<=r["last"]<=132 and
                0<=r["target"]<=132 and 9<=r["gap12"]<=15):
            raise ValueError("Bad longitudinal input")
    return tuple(torch.tensor([r[key] for r in rows],
                              dtype=torch.float32,device=device)
                 for key in ("bl","last","target","gap12"))

def norm(x):return torch.log1p(x)/LOGMAX
def denorm(x):return torch.expm1(x.clamp(0,1)*LOGMAX).clamp(0,132)

class ClinicalLatentDynamics(nn.Module):
    def __init__(self,width=64):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(1,width),nn.SiLU(),nn.Linear(width,width))
        self.trans=nn.Sequential(nn.Linear(width+3,2*width),
                                 nn.SiLU(),nn.Linear(2*width,width))
        self.update=nn.GRUCell(width,width)
        self.head=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,2))
    def encode(self,s):return self.enc(norm(s)[:,None])
    def imagine(self,z,dt):
        u=dt/24
        time=torch.stack((u,torch.sin(math.pi*u),torch.cos(math.pi*u)),-1)
        return z+self.trans(torch.cat((z,time),-1))
    def observe(self,latent,measured):
        return self.update(self.encode(measured),latent)
    def decode(self,z):
        out=self.head(z)
        return out[:,0].sigmoid(),.035+.4*out[:,1].sigmoid()
    def forward(self,bl,score12,gap12):
        start=self.encode(bl)
        future12=self.imagine(start,gap12)
        belief=self.observe(future12,score12)
        future24=self.imagine(belief,24-gap12)
        # Hypothetical *open-loop* path with no measured intermediate visit.
        rollout=self.imagine(self.imagine(start,torch.full_like(gap12,12.)),
                            torch.full_like(gap12,12.))
        return future12,future24,rollout
    def forecast(self,bl,score12,gap12):
        future=self.forward(bl,score12,gap12)[1]
        mu,scale=self.decode(future)
        return denorm(mu),scale

def loss_fn(model,bl,score12,truth,gap12):
    z12,z24,zroll=model(bl,score12,gap12)
    def gaussian(mu,scale,target):
        return (.5*((mu-target)/scale).square()+scale.log()).mean()
    mu12,sd12=model.decode(z12)
    mu24,sd24=model.decode(z24)
    mur,_=model.decode(zroll)
    target12=norm(score12);target24=norm(truth)
    teacher12=model.encode(score12).detach()
    teacher24=model.encode(truth).detach()
    align=(F.smooth_l1_loss(z12,teacher12)+
           F.smooth_l1_loss(z24,teacher24)+
           .5*F.smooth_l1_loss(zroll,teacher24))
    embedding=model.encode(torch.cat((bl,score12)))
    spread=torch.sqrt(embedding.var(dim=0,unbiased=False)+1e-5)
    return (gaussian(mu12,sd12,target12)+2*gaussian(mu24,sd24,target24)+
            .5*F.smooth_l1_loss(mur,target24)+.08*align+
            .1*F.relu(.05-spread).mean())

def fit_one(train,dev,seed,width,epochs,device):
    torch.manual_seed(seed)
    if device.type=="cuda":torch.cuda.manual_seed_all(seed)
    rng=torch.Generator().manual_seed(seed+937)
    tensors=batch(train,device);validation=batch(dev,device)
    model=ClinicalLatentDynamics(width).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=.015)
    best=float("inf");weights=None;epoch_best=0;idle=0
    for epoch in range(1,epochs+1):
        model.train()
        for idx in torch.randperm(len(train),generator=rng).split(64):
            idx=idx.to(device)
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model,*(v[idx] for v in tensors))
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite WM loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step()
        model.eval()
        with torch.no_grad():
            pred,_=model.forecast(validation[0],validation[1],validation[3])
            err=float((pred-validation[2]).abs().mean())
        if err<best-1e-5:
            best,epoch_best,idle=err,epoch,0
            weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            idle+=1
            if idle>=10:break
    if weights is None:raise RuntimeError("WM training did not converge")
    model.load_state_dict(weights);model.eval()
    return model,epoch_best,best

def evaluate(rows,output,folds=3,members=3,epochs=60,
             seed=20261018,width=64,device_name="cpu",bootstraps=300):
    if not 2<=folds<=5 or not 1<=members<=5 or not 2<=epochs<=250 or bootstraps<100:
        raise ValueError("Invalid run settings")
    if len(rows)<100 or len({r["pid"] for r in rows})!=len(rows):
        raise ValueError("Too few distinct frozen TRAIN patients")
    device=torch.device(device_name)
    if device.type=="cpu":torch.set_num_threads(min(4,torch.get_num_threads()))
    assignments=folds_for(rows,folds,seed)
    predictions={};logs=[]
    for fold in range(folds):
        source=[r for r in rows if assignments[r["pid"]]!=fold]
        held=[r for r in rows if assignments[r["pid"]]==fold]
        fit,dev=dev_split(source,seed+fold*199)
        if {r["pid"] for r in fit}&{r["pid"] for r in dev}:raise AssertionError("Inner leakage")
        ensemble=[]
        for m in range(members):
            model,epoch,error=fit_one(fit,dev,seed+fold*1009+m*9176,width,epochs,device)
            ensemble.append(model)
            logs.append(dict(fold=fold,member=m,fit_n=len(fit),dev_n=len(dev),
                             held_n=len(held),best_epoch=epoch,
                             inner_dev_mae=round(error,5)))
        x=batch(held,device)
        with torch.no_grad():
            predictions_per_member=torch.stack([
                model.forecast(x[0],x[1],x[3])[0] for model in ensemble])
            estimate=predictions_per_member.mean(0).cpu().tolist()
        for r,value in zip(held,estimate):
            if r["pid"] in predictions:raise AssertionError("Repeated heldout")
            predictions[r["pid"]]=dict(truth=r["target"],latent=value,
                  last=r["last"],bl=r["bl"],gap=r["gap12"])
    if len(predictions)!=len(rows):raise RuntimeError("Incomplete CV")
    vals=list(predictions.values())
    methods={
      "last_value":lambda r:r["last"],
      "baseline_value":lambda r:r["bl"],
      "fixed_50_50_smoothing":lambda r:.5*(r["bl"]+r["last"]),
      "linear_extrapolation":lambda r:max(0.,min(132.,
        r["last"]+(r["last"]-r["bl"])*(24-r["gap"])/r["gap"])),
      "latent_imagination_wm":lambda r:r["latent"],
    }
    metrics=[]
    for name,fn in methods.items():
        error=[fn(r)-r["truth"] for r in vals]
        metrics.append(dict(model=name,n=len(vals),mae=round(mean(map(abs,error)),5),
                 rmse=round(math.sqrt(mean(e*e for e in error)),5),
                 bias=round(mean(error),5)))
    raw_delta=mean(abs(r["latent"]-r["truth"])-
                   abs(.5*(r["bl"]+r["last"])-r["truth"]) for r in vals)
    rng=random.Random(seed+401)
    draws=[]
    for _ in range(bootstraps):
        sample=[vals[rng.randrange(len(vals))] for _ in vals]
        draws.append(mean(abs(r["latent"]-r["truth"])-
                   abs(.5*(r["bl"]+r["last"])-r["truth"]) for r in sample))
    draws.sort()
    def q(x):
        v=x*(len(draws)-1);i=int(v);j=min(i+1,len(draws)-1)
        return draws[i]*(j-v)+draws[j]*(v-i)
    difference=[dict(model="latent_imagination_wm",reference="fixed_50_50_smoothing",
         delta_mae=round(raw_delta,5),bootstrap_lower_95=round(q(.025),5),
         bootstrap_upper_95=round(q(.975),5),bootstraps=bootstraps)]
    output=Path(output).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    for filename,items in (("oof_metrics.csv",metrics),
                           ("fold_training.csv",logs),
                           ("paired_smoothing_contrast.csv",difference)):
        with (output/filename).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(items[0]))
            writer.writeheader();writer.writerows(items)
    summary=dict(status="exploratory_TRAIN_only_latent_dynamics_JEPA_imagination",
       train_trajectories=len(rows),outer_folds=folds,ensemble_members=members,
       width=width,epochs_max=epochs,bootstrap_reps=bootstraps,seed=seed,
       timeline="BL+observed 12m -> nominal 24m",
       caveats=["Inspired by Dreamer/clinical JEPA, NOT a reproduction of Dreamer4",
        "No recorded therapeutic action sequence; no causal interventions",
        "Only sparse TRAIN trajectories; high overfitting risk",
        "Inner dev only selects epochs; no future actual visit time as input",
        "Phase44 TEST consumed and never accessed",
        "Only aggregate metrics exported; no patient identifiers",
        "Clinical forecasting, not yet cost-aware agent decision world modelling"],
       generated_utc=datetime.now(timezone.utc).isoformat())
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--folds",type=int,default=3)
    p.add_argument("--members",type=int,default=3)
    p.add_argument("--epochs",type=int,default=60)
    p.add_argument("--width",type=int,default=64)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--bootstraps",type=int,default=300)
    p.add_argument("--seed",type=int,default=20261018)
    a=p.parse_args()
    manifest=a.manifest_db.expanduser().resolve()
    out=a.report_dir.expanduser().resolve()
    if not manifest.is_file() or manifest==out:
        raise FileNotFoundError("Missing private manifest or unsafe output")
    from scripts.phase29_sequential import load
    rows=load(manifest) # frozen TRAIN-only query
    print(json.dumps(evaluate(rows,out,a.folds,a.members,a.epochs,
                              a.seed,a.width,a.device,a.bootstraps),indent=2))
    print("Share ONLY aggregate CSV/JSON; not private patient database")
if __name__=="__main__":main()
