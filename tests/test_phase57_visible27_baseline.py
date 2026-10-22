"""Synthetic Phase57 same-cohort real R01 visible sum decision baseline."""
import unittest
from scripts.phase57_visible27_same_cohort_baseline import build_scores,summary_at_budget
from scripts.phase36_masked_site_feasibility import ALL,HIDDEN

def people(n=60):
    data={}
    for i in range(n):
        bl={"month":10,"event":"BL","score":(0,)*33,"status":(1,)*33}
        remote_score=tuple(1 if j not in [ALL.index(x) for x in HIDDEN] and j < i%15 else 0 for j in range(33))
        remote_status=tuple(2 if item in HIDDEN else 1 for item in ALL)
        remote={"month":16,"event":"R01","score":remote_score,"status":remote_status}
        future=list(remote_score)
        if i%3==0:future[ALL.index(HIDDEN[0])]=1
        site={"month":22,"event":"V04","score":tuple(future),"status":(1,)*33}
        data["SIM"+str(i)]=(bl,remote,site)
    return data

class Phase57Tests(unittest.TestCase):
    def test_selects_exact_budget_and_positive_labels(self):
        records=build_scores(people())
        metrics=summary_at_budget(records,20261018)
        self.assertEqual(metrics["eligible"],60)
        self.assertEqual(metrics["positive"],20)
        self.assertEqual(metrics["selected"],12)
        self.assertAlmostEqual(metrics["random_expected_recall"],.2)
    def test_cannot_use_hidden_remote_codes(self):
        data=people()
        pair=list(data["SIM0"])
        changed=dict(pair[1])
        changed["score"]=tuple(4 if item in HIDDEN else value
                               for item,value in zip(ALL,changed["score"]))
        pair[1]=changed
        data["SIM0"]=tuple(pair)
        initial=build_scores(people())
        altered=build_scores(data)
        self.assertEqual(initial,altered)

if __name__=="__main__":unittest.main()
