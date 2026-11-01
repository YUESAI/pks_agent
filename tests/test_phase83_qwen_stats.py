"""Phase83 Qwen statistics regression checks.

Run: python -m unittest discover -s tests -p test_phase83_qwen_stats.py -v
"""
import unittest
from collections import Counter
from scripts.phase83_conditional_evidence_agent import combine_qwen_stats

class QwenStatsRegression(unittest.TestCase):
    def test_no_eligible_qwen_actions(self):
        x=combine_qwen_stats(Counter(),Counter(),Counter())
        self.assertEqual(x["qwen_calls"],0)
        self.assertEqual(x["released"],0)
        self.assertEqual(x["hard_abstain"],0)

    def test_missing_call_key_on_eval_only(self):
        x=combine_qwen_stats(Counter({"hard_abstain":8}),
                Counter({"qwen_calls":3,"released":3}),
                Counter({"qwen_calls":1,"invalid":1}))
        self.assertEqual(x["qwen_calls"],4)
        self.assertEqual(x["hard_abstain"],8)
        self.assertEqual(x["released"],3)
        self.assertEqual(x["invalid"],1)

    def test_all_tasks_report_every_key(self):
        x=combine_qwen_stats(Counter({"qwen_calls":2,"released":1}),
                             Counter(),Counter({"llm_abstain":1}))
        self.assertEqual(set(x),{"qwen_calls","released","hard_abstain",
                                  "llm_abstain","invalid"})
        self.assertEqual(x["llm_abstain"],1)

if __name__=="__main__":
    unittest.main()
