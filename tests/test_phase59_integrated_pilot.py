"""Phase59 split, Qwen tool grounding, temporal-prior & RSI rollback tests."""
import json,unittest
from scripts.phase36_masked_site_feasibility import VISIBLE
from scripts.phase59_grounded_collaboration import (
 JointBelief,episode,parse_evolver
)
from scripts.phase59_integrated_pilot import split,execute
from scripts.phase59_rsi_verifier import accepted_revision
from scripts.phase45_world_model import ObservationWorldModel

ITEMS=tuple(VISIBLE)

def people(n=220):
    return [{"pid":f"SYNTHETIC_{i:04d}",
             "features":{k:int((i+j)%9==0) for j,k in enumerate(ITEMS)},
             "label":int(i%3==0)}
            for i in range(n)]

class FakeDoctor:
    def __init__(self,invalid=False):self.invalid=invalid;self.calls=0
    def generate(self,messages):
        self.calls+=1
        content=messages[-1]["content"]
        if "configuration proposer" in content:
            return json.dumps({"weight":0.25,"temperature":1.0,
                               "style":"risk_reduction"})
        if self.invalid:return '{"ask":"UNKNOWN_PATIENT_OUTCOME"}'
        import re
        allowed=json.loads(re.search(
            r"Available tools \(only these names may be requested\): (\[[^\n]+\])",
            content).group(1))
        return json.dumps({"ask":allowed[0]})

class Phase59Tests(unittest.TestCase):
    def test_patient_split_disjoint_and_deterministic(self):
        rows=people()
        a,b,c=split(rows,12)
        a2,b2,c2=split(list(reversed(rows)),12)
        self.assertEqual([p["pid"] for p in c],[p["pid"] for p in c2])
        self.assertFalse({r["pid"] for r in a}&{r["pid"] for r in b+c})
    def test_invalid_action_does_not_hallucinate_ground_truth(self):
        rows=people()
        fit,dev,held=split(rows,12)
        world=ObservationWorldModel(ITEMS).fit(fit)
        backend=FakeDoctor(invalid=True)
        result=episode(held[0],JointBelief(world),backend,4,"plain")
        self.assertEqual(result["queries"],4)
        self.assertEqual(result["invalid"],4)
    def test_all_four_arms_and_proposal_verifier(self):
        rows=people()
        priors={r["pid"]:.35 for r in rows}
        for mode in ("plain","observation","dual","rsi"):
            doctor=FakeDoctor()
            m,changes,meta=execute(rows,ITEMS,mode,doctor,4,12,
                                   priors=priors,rsi_rounds=2)
            self.assertEqual(m["patients"],12)
            self.assertEqual(m["total_questions"],48)
            self.assertEqual(meta["train_site_fit_patients"],143)
            self.assertEqual(len(changes),2 if mode=="rsi" else 0)
    def test_allowlist_rejects_untrusted_code(self):
        base=(.5,1.,"information_gain")
        self.assertIsNone(parse_evolver(
            '{"weight":0.25,"temperature":1,"style":"run_shell"}',base))
        self.assertIsNone(parse_evolver(
            '{"weight":0.25,"temperature":1,"style":"uncertainty","code":"X"}',base))
        self.assertEqual(parse_evolver(
            '{"weight":0.25,"temperature":1,"style":"uncertainty"}',base),
            (.25,1.,"uncertainty"))
    def test_hidden_data_does_not_enter_llm_prompt(self):
        rows=people()
        fit,dev,held=split(rows,12)
        world=ObservationWorldModel(ITEMS).fit(fit)
        class PromptDoctor(FakeDoctor):
            def generate(self,messages):
                txt=json.dumps(messages)
                self_outer.assertNotIn("SYNTHETIC_",txt)
                self_outer.assertNotIn("hidden_truth",txt)
                return super().generate(messages)
        self_outer=self
        episode(held[0],JointBelief(world,.2,.5),PromptDoctor(),3,"dual")

if __name__=="__main__":unittest.main()
