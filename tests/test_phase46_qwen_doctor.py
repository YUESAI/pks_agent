"""Synthetic Phase46 strict LLM action verifier; no real model download."""
import unittest
from scripts.phase46_qwen_doctor import extract_action,make_messages,llm_doctor_rollout
from scripts.phase45_world_model import ObservationWorldModel

class FakeBackend:
 def __init__(self,replies):
  self.replies=list(replies)
 def generate(self,messages):
  return self.replies.pop(0)

class Phase46Tests(unittest.TestCase):
 def test_parser_rejects_hallucination_and_extra_keys(self):
  allowed={"NP3SPCH","NP3FACXP"}
  self.assertEqual(extract_action('{"ask":"NP3SPCH"}',allowed),"NP3SPCH")
  self.assertIsNone(extract_action('{"ask":"NP3RIGN"}',allowed))
  self.assertIsNone(extract_action('{"ask":"NP3SPCH","score":4}',allowed))
  self.assertEqual(extract_action('output: {"ask":"NP3FACXP"}',allowed),"NP3FACXP")
 def test_patient_tool_never_sees_hidden_truth(self):
  items=("NP3SPCH","NP3FACXP")
  train=[]
  for i in range(24):
   train.append({"label":i%2,
                 "features":{items[0]:i%5,items[1]:(i//2)%5}})
  wm=ObservationWorldModel(items).fit(train)
  record={"features":{items[0]:2,items[1]:1},
          "label":1,"hidden":20,"pid":"NEVER_TO_PROMPT"}
  class CapturingBackend(FakeBackend):
   def generate(self,messages):
    text=str(messages)
    assert "NEVER_TO_PROMPT" not in text
    assert "hidden" not in str(messages[-1]["content"]).lower() or "hidden-six" in text
    return super().generate(messages)
  agent=CapturingBackend(['{"ask":"NP3RIGN"}','{"ask":"NP3FACXP"}'])
  result=llm_doctor_rollout(wm,record,agent,items,2,"llm_wm")
  self.assertEqual(result["observed_count"],2)
  self.assertEqual(result["invalid_action_fallbacks"],1)
 def test_mode_hides_wm_when_plain(self):
  p=make_messages(["A"],{},.8,{"A":.1},"llm_plain")
  self.assertNotIn("0.8000",str(p))
  self.assertNotIn("0.1",str(p))
if __name__=="__main__":unittest.main()
