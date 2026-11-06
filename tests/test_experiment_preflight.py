"""Tests for experiment preflight ledger and duplicate warning.

python -m unittest discover -s tests -p test_experiment_preflight.py -v
"""
import unittest
from scripts.experiment_preflight import load_registry,matching_records

class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=load_registry(
            "docs/experiment_registry.json")
    def test_all_phase60_to90_exist_once(self):
        self.assertEqual([r["phase"] for r in self.data["records"]],
                         list(range(60,92)))
    def test_phase91_planned_but_not_yet_run(self):
        found=matching_records(self.data,phase=91)
        self.assertEqual(len(found),1)
        self.assertEqual(found[0]["status"],"PLANNED_NOT_RUN")
    def test_phase90_does_not_collide_with_p89(self):
        found=matching_records(self.data,phase=90)
        self.assertEqual(len(found),1)
        self.assertEqual(found[0]["mechanism"],"partial33-likelihood")
    def test_exact_duplicate_has_match(self):
        found=matching_records(self.data,mechanism="partial33-likelihood",
                               split="global80-20-dev")
        self.assertEqual([r["phase"] for r in found],[90])
    def test_distinct_family_not_false_duplicate(self):
        found=matching_records(self.data,mechanism="new-state-mechanism",
                               split="global80-20-dev")
        self.assertEqual(found,[])
    def test_query_existing_agent_method(self):
        found=matching_records(self.data,query="Qwen")
        self.assertGreaterEqual(len(found),3)
    def test_unknown_query(self):
        self.assertEqual(matching_records(self.data,query="nothing_like_this_78787"),[])

if __name__=="__main__":unittest.main()
