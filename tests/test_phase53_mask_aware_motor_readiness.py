"""Phase53 synthetic safeguards: 101 is unrateable, never a score 0."""
import unittest
from scripts.phase53_mask_aware_motor_readiness import audit_items
from scripts.phase36_masked_site_feasibility import ALL,HIDDEN,VISIBLE

def month(m):
    mm=m%12 or 12
    year=(m-mm)//12
    return f"{mm:02d}/{year:04d}"

def exam(pid,event,m,remote=False,allzero=False):
    record={"PATNO":pid,"EVENT_ID":event,"EXAMDT":month(m),
            "PDSTATE":"","NP3TOT":""}
    for name in ALL:
        record[name]="101" if remote and name in HIDDEN else "0"
    if remote:
        record["NP3TOT"]=""
    else:
        record["NP3TOT"]="0"
    return record

class Phase53Tests(unittest.TestCase):
    def test_real_R01_missingness_kept_but_never_imputed(self):
        anchor=2020*12+1
        patients={f"P{i}":anchor for i in range(20)}
        rows=[]
        for pid in patients:
            rows.extend((exam(pid,"BL",anchor),
                         exam(pid,"R01",anchor+6,remote=True),
                         exam(pid,"V04",anchor+12)))
        # An ambiguous contradictory 12m onsite duplicate causes only
        # the affected patient's month to be dropped.
        duplicate=exam("P0","V04",anchor+12)
        duplicate["NP3SPCH"]="1"
        duplicate["NP3TOT"]="1"
        rows.append(duplicate)
        rows.append(exam("NOT_A_TRAIN_PATIENT","R01",anchor+6,remote=True))
        summary,tables=audit_items(rows,patients)
        self.assertEqual(summary["frozen_train_patients"],20)
        self.assertEqual(summary["remote_visible27_six_unrateable_episodes"],20)
        self.assertEqual(summary["unique_patients_with_real_R01_masked_episode"],20)
        self.assertEqual(summary["patients_with_masked_R01_and_later_verified_V04"],19)
        self.assertEqual(summary["baseline_anchored_at_least_3_episodes"],19)
        self.assertEqual(summary["baseline_anchored_episode_transition_pairs"],39)
        self.assertEqual(summary["all_usable_partial_or_complete_episodes"],59)
        self.assertIn("remote_27_valid_six_unrateable",str(tables))
        self.assertNotIn("NOT_A_TRAIN_PATIENT",str(tables))
        self.assertNotIn("P0",str(tables))
    def test_scores_101_not_imputed_to_zero(self):
        anchor=2020*12+1
        patients={f"P{i}":anchor for i in range(10)}
        rows=[exam(p,"BL",anchor) for p in patients]
        for pid in patients:
            r=exam(pid,"R01",anchor+6,remote=True)
            r["NP3SPCH"]="101"  # only 26 of 27 are observable
            rows.append(r)
        summary,tables=audit_items(rows,patients)
        self.assertEqual(summary["remote_visible27_six_unrateable_episodes"],0)
        self.assertEqual(summary["all_usable_partial_or_complete_episodes"],20)
        pattern={r["mask_type"]:r["episodes"] for r in tables["episode_mask_patterns.csv"]}
        self.assertEqual(pattern["partial_10to26"],"10")

if __name__=="__main__":unittest.main()
