"""Masked 33-item irregular-time recurrent latent WM (Phase54).

Time-conditioned retrospective forecasting: target visit gap is known.
Future event, future item mask, values and target-visit provenance
never enter the predictor. The decoder outputs five score logits/item.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class MaskedMotorWM(nn.Module):
    def __init__(self,nitems=33,width=64):
        super().__init__()
        self.nitems=nitems
        self.item=nn.Embedding(nitems,width)
        self.value=nn.Embedding(7,width) # 0..4, missing=5, UR=6
        self.visit=nn.Sequential(nn.Linear(width,width),nn.GELU())
        self.transition=nn.Sequential(nn.Linear(width+3,width),
                                      nn.Tanh(),nn.Linear(width,width))
        self.belief=nn.GRUCell(width,width)
        self.decoder=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),
                                   nn.GELU(),nn.Linear(width,nitems*5))
    def encode(self,value,status):
        # status 0=absent, 1=measured, 2=unable to rate.
        token=torch.where(status==1,value,torch.where(
            status==2,torch.full_like(value,6),torch.full_like(value,5)))
        pos=torch.arange(self.nitems,device=value.device)
        embedded=self.value(token)+self.item(pos)[None,:,:]
        return self.visit(embedded.mean(dim=1))
    def step(self,belief,gap):
        g=(gap/24).clamp(0,5)
        time=torch.stack((g,torch.sin(g),torch.cos(g)),-1)
        return belief+self.transition(torch.cat((belief,time),-1))
    def forward(self,values,status,months,visit_valid):
        # shape [B,T,33], [B,T,33], [B,T], [B,T]
        b,t,n=values.shape
        if n!=self.nitems or status.shape!=values.shape:raise ValueError("Bad shape")
        belief=torch.zeros(b,self.item.embedding_dim,device=values.device)
        predictions=[]
        for j in range(t):
            gap=torch.zeros(b,device=values.device) if j==0 else months[:,j]-months[:,j-1]
            if (gap<0).any():raise ValueError("Negative visit gap")
            prior=self.step(belief,gap)
            if j>0:predictions.append(self.decoder(prior).view(b,n,5))
            measured=self.encode(values[:,j],status[:,j])
            updated=self.belief(measured,prior)
            belief=torch.where(visit_valid[:,j,None],updated,belief)
        return torch.stack(predictions,dim=1)

def patient_batch(patients,device):
    if not patients:raise ValueError("Empty patient batch")
    b=len(patients);t=max(len(p["visits"]) for p in patients)
    x=torch.zeros(b,t,33,dtype=torch.long,device=device)
    s=torch.zeros_like(x)
    m=torch.zeros(b,t,device=device)
    valid=torch.zeros(b,t,dtype=torch.bool,device=device)
    for i,p in enumerate(patients):
        for j,v in enumerate(p["visits"]):
            x[i,j]=torch.tensor(v["score"],device=device)
            s[i,j]=torch.tensor(v["status"],device=device)
            m[i,j]=v["month"]
            valid[i,j]=True
        m[i,len(p["visits"]):]=p["visits"][-1]["month"]
    return x,s,m,valid

def observed_nll(logits,values,status,visit_valid):
    # logits [B,T-1,33,5] maps true visits indices 1...
    target=values[:,1:]
    keep=(status[:,1:]==1)&visit_valid[:,1:,None]
    ll=F.cross_entropy(logits.reshape(-1,5),target.reshape(-1),
                       reduction="none").reshape(target.shape)
    # Mean per patient, then mean across patients (not pooled item counts).
    denom=keep.sum((1,2)).clamp_min(1)
    return ((ll*keep).sum((1,2))/denom).mean()

def forecast(model,patients,device):
    model.eval()
    with torch.no_grad():
        x,s,m,valid=patient_batch(patients,device)
        return model(x,s,m,valid).softmax(-1).cpu(),x.cpu(),s.cpu(),valid.cpu()
