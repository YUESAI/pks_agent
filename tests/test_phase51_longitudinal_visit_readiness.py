"""Synthetic privacy and validity checks for Phase51 raw visit audit."""
import unittest
from scripts.phase51_longitudinal_visit_readiness import audit_records

def sample(pid,event,month,score,med=""):
    m=month%12 or 12
    year=(month-m)//12
    return {"PATNO":pid,"EVENT_ID":event,"EXAMDT":f"{m:02d}/{year:04d}",
            "NP3TOT":str(score),"PDSTATE":med}

class Phase51Tests(unittest.TestCase):
    def test_three_visits_and_conflicted_duplicates(self):
        # Month indexing is 12*year + month; 2020/01 = 24241
        anchor=2020*12+1
        anchors={f"P{i}":anchor for i in range(30)}
        rows=[]
        for pid in anchors:
            rows.extend([sample(pid,"BL",anchor,4),
                         sample(pid,"V04",anchor+12,5),
                         sample(pid,"V06",anchor+24,6)])
        # Reject one ambiguous followup, not entire patient.
        rows.append(sample("P0","V04",anchor+12,8))
        # Exclude leaked nonTRAIN data.
        rows.extend([sample("NONTRAIN","BL",anchor,22),
                     sample("NONTRAIN","V04",anchor+12,26)])
        stats,tables=audit_records(rows,anchors)
        self.assertEqual(stats["frozen_train_patients"],30)
        self.assertEqual(stats["primary_state_UNSPECIFIED_with_baseline"],30)
        self.assertEqual(stats["primary_state_consecutive_visit_pairs"],58)
        self.assertEqual(stats["usable_unique_patient_month_state_exams"],89)
        joined=str(tables)
        self.assertNotIn("NONTRAIN",joined)
        self.assertNotIn("P0",joined)
        self.assertIn("<10",joined)
    def test_medication_states_separated(self):
        anchor=2020*12+1
        anchors={f"P{i}":anchor for i in range(20)}
        rows=[]
        for pid in anchors:
            rows.extend([sample(pid,"BL",anchor,3),
                         sample(pid,"V04",anchor+12,4,"ON")])
        stats,tables=audit_records(rows,anchors)
        self.assertEqual(stats["primary_state_UNSPECIFIED_with_baseline"],20)
        self.assertEqual(stats["primary_state_consecutive_visit_pairs"],0)
        self.assertEqual(stats["usable_unique_patient_month_state_exams"],40)

if __name__=="__main__":unittest.main()
