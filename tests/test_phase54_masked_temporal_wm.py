"""Synthetic Phase54 checks: no 101-as-zero, temporal scores, patient splits."""
import random,tempfile,unittest
from pathlib import Path
import torch
from scripts.phase36_masked_site_feasibility import ALL,HIDDEN
from scripts.phase54_masked_sequences import sequences_from_rows
from scripts.phase54_temporal_wm import (
    MaskedMotorWM,patient_batch,observed_nll)
from scripts.phase54_evaluate import split_patients,run

def exam(pid,event,month,hidden=False):
    m=month%12 or 12;y=(month-m)//12
    out={"PATNO":pid,"EVENT_ID":event,"EXAMDT":f"{m:02d}/{y:04d}",
         "PDSTATE":"","NP3TOT":""}
    for name in ALL:
        out[name]="101" if hidden and name in HIDDEN else "0"
    if not hidden:out["NP3TOT"]="0"
    return out

def synthetic(n=110):
    rng=random.Random(10)
    data=[]
    for i in range(n):
        visits=[]
        for t in (0,6,12,24):
            scores=tuple(rng.choice([0,0,0,1,2]) for _ in range(33))
            statuses=tuple(2 if t==6 and k<6 else 1 for k in range(33))
            visits.append({"month":24241+t,"event":"BL" if t==0 else "OTHER",
                           "score":scores,"status":statuses})
        data.append({"pid":f"PRIVATE_{i}","visits":visits})
    return data

class Phase54Tests(unittest.TestCase):
    def test_preserve_structural_unobservability(self):
        anchor=2020*12+1
        people={f"P{i}":anchor for i in range(10)}
        rows=[]
        for pid in people:
            rows.extend([exam(pid,"BL",anchor),
                         exam(pid,"R01",anchor+6,True),
                         exam(pid,"V04",anchor+12)])
        data,meta=sequences_from_rows(rows,people)
        self.assertEqual(len(data),10)
        remote=data[0]["visits"][1]
        self.assertEqual(sum(s==2 for s in remote["status"]),6)
        self.assertEqual(sum(s==1 for s in remote["status"]),27)
        self.assertEqual(meta["anchored_transition_pairs"],20)
    def test_masked_loss_and_temporal_forward(self):
        torch.set_num_threads(2)
        data=synthetic(4)
        values,status,months,valid=patient_batch(data,torch.device("cpu"))
        wm=MaskedMotorWM(width=16)
        logits=wm(values,status,months,valid)
        self.assertEqual(tuple(logits.shape),(4,3,33,5))
        loss=observed_nll(logits,values,status,valid)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in wm.parameters()))
    def test_future_value_does_not_enter_its_own_prediction(self):
        torch.set_num_threads(2)
        torch.manual_seed(54)
        rows=synthetic(2)
        x,s,m,v=patient_batch(rows,torch.device("cpu"))
        model=MaskedMotorWM(width=16).eval()
        # Synthetic R01 at visit index 1 has status=2 for items 0..5
        # (unrateable) and status=1 for items 6..32 (actually measured).
        self.assertTrue(torch.all(s[:,1,0]==2).item())
        self.assertTrue(torch.all(s[:,1,6]==1).item())
        with torch.no_grad():
            first=model(x,s,m,v)
            # Changing a 101/UR placeholder MUST NEVER affect belief
            # or predictions, including subsequent visits.
            unobserved=x.clone()
            unobserved[:,1,0]=(unobserved[:,1,0]+3)%5
            invisible_change=model(unobserved,s,m,v)
            # Changing a genuinely observed prior visit's score may
            # update belief and therefore alter subsequent forecasts.
            observed=x.clone()
            observed[:,1,6]=(observed[:,1,6]+3)%5
            measured_change=model(observed,s,m,v)
        self.assertTrue(torch.equal(first,invisible_change))
        # At visit 1, prediction occurs before incorporating its data.
        self.assertTrue(torch.equal(first[:,0],measured_change[:,0]))
        # At visit 2+, updated belief is allowed to reflect known visit 1.
        self.assertGreater(
            float((first[:,1:]-measured_change[:,1:]).abs().max()),1e-8)

    def test_unobserved_future_status_cannot_leak_into_current_forecast(self):
        torch.set_num_threads(2)
        torch.manual_seed(55)
        x,s,m,v=patient_batch(synthetic(2),torch.device("cpu"))
        model=MaskedMotorWM(width=16).eval()
        modified_status=s.clone()
        modified_status[:,1,6]=0
        with torch.no_grad():
            baseline=model(x,s,m,v)
            modified=model(x,modified_status,m,v)
        self.assertTrue(torch.equal(baseline[:,0],modified[:,0]))
    def test_nested_cv_and_reports(self):
        torch.set_num_threads(2)
        rows=synthetic()
        assigned=split_patients(rows,5,21)
        self.assertEqual(set(assigned.values()),set(range(5)))
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/"report"
            result=run(rows,output,folds=5,width=16,max_epochs=2)
            self.assertEqual(result["patient_trajectories"],110)
            self.assertTrue((output/"patient_oof_item_nll.csv").is_file())
            for path in output.iterdir():
                self.assertNotIn("PRIVATE_",path.read_text())
if __name__=="__main__":unittest.main()
