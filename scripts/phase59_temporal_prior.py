"""Phase59 temporal prior: use prior BL measurements and target horizon.

Target V04 scores/status are never put into the neural forecast input.
The simulation is retrospective and target visit month is known.
"""
import torch
from scripts.phase54_evaluate import train_model
from scripts.phase54_temporal_wm import forecast
from scripts.phase54_masked_sequences import load_train
from scripts.phase36_masked_site_feasibility import ALL,HIDDEN

SIX=tuple(ALL.index(k) for k in HIDDEN)

def fit_prior(root,manifest,excluded,width=64,epochs=8,seed=20261018):
    patients,meta=load_train(root,manifest)
    fitting=[p for p in patients if p["pid"] not in excluded]
    if len(fitting)<100:raise ValueError("Insufficient TRAIN history")
    device=torch.device("cpu")
    torch.set_num_threads(4)
    wm=train_model(fitting,width,epochs,seed,device).eval()
    index={p["pid"]:p for p in patients}
    report={"temporal_fit_patients":len(fitting),
            "temporal_model_parameters":sum(v.numel() for v in wm.parameters()),
            "temporal_epochs":epochs}
    return wm,index,device,report

def forecast_prior(wm,index,device,pid):
    person=index.get(pid)
    if person is None:return None
    baseline=next((v for v in person["visits"] if v["event"]=="BL"),None)
    future=next((v for v in person["visits"] if v["event"]=="V04"),None)
    if baseline is None or future is None:return None
    dummy={"month":future["month"],"event":"UNOBSERVED_TARGET",
           "score":(0,)*33,"status":(0,)*33}
    prob,*_=forecast(wm,[{"visits":[baseline,dummy]}],device)
    return max(1e-5,min(1-1e-5,
               1.-float(prob[0,-1,list(SIX),0].prod())))
