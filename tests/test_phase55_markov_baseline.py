"""Phase55 deterministic conditional categorical baseline regression tests."""
import unittest
from scripts.phase55_markov_baseline import ItemMarkov,choose_markov

def mock(n=32):
    rows=[]
    for i in range(n):
        visits=[]
        for step in (0,6,12,24):
            values=[(i+j+step//6)%5 for j in range(33)]
            status=[1]*33
            if step==6:
                for k in range(6):status[k]=2
            visits.append({"month":24241+step,"score":values,"status":status})
        rows.append({"pid":"SIM"+str(i),"visits":visits})
    return rows

class MarkovTests(unittest.TestCase):
    def test_probabilities_and_missingness(self):
        rows=mock()
        model=ItemMarkov(10.,"gap").fit(rows)
        self.assertAlmostEqual(sum(model.predict(0,2,12)),1.,places=7)
        self.assertAlmostEqual(sum(model.predict(4)),1.,places=7)
        self.assertGreater(model.predict(0,2,12).min(),0.)
        scores=model.score_patients(rows[24:])
        self.assertEqual(len(scores),8)
        self.assertTrue(all(x>0 for x in scores.values()))
    def test_inner_dev_selection_only(self):
        rows=mock()
        result=choose_markov(rows[:24],rows[24:])
        self.assertIn(result["selected_gap_mode"],("gap","pooled"))
        self.assertIn(result["selected_alpha"],(2.,10.,50.,200.))
        self.assertEqual(result["candidate_count"],8)
    def test_unrateable_not_scored_as_zero(self):
        rows=mock(1)
        model=ItemMarkov().fit(mock(32))
        base=model.score_patients(rows)["SIM0"]
        rows[0]["visits"][1]["score"][0]=4 # 101/UR placeholder, not measured
        changed=model.score_patients(rows)["SIM0"]
        self.assertAlmostEqual(base,changed,places=10)

if __name__=="__main__":unittest.main()
