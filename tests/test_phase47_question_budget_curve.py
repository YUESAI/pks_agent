"""Phase47 synthetic deterministic checkpoint and no-label-leakage checks."""
import random,tempfile,unittest
from pathlib import Path
from scripts.phase47_question_budget_curve import (
    patient_split,baseline_snapshots,aggregate
)
from scripts.phase45_world_model import ObservationWorldModel

def fixtures(n=120):
    rng=random.Random(11)
    items=tuple("N"+str(i) for i in range(12))
    rows=[]
    for i in range(n):
        features={k:rng.choice((0,0,1,2)) for k in items}
        label=int(sum(features[k] for k in items[:3])>1)
        rows.append({"pid":"PRIVATE_"+str(i),"features":features,
                     "label":label,"hidden":float(label)})
    return rows,items

class Phase47Tests(unittest.TestCase):
    def test_patient_partition_without_label_ranking(self):
        rows,items=fixtures()
        fit,held=patient_split(rows,40,123)
        self.assertFalse(set(r["pid"] for r in fit)&set(r["pid"] for r in held))
        self.assertEqual(len(held),30)
        flipped=[dict(r,label=1-r["label"]) for r in rows]
        f2,h2=patient_split(flipped,40,123)
        self.assertEqual([r["pid"] for r in held],[r["pid"] for r in h2])

    def test_static_and_dynamic_checkpoints_use_measured_prefix_only(self):
        rows,items=fixtures()
        wm=ObservationWorldModel(items).fit(rows[:80])
        for policy in ("static_questions","dynamic_questions"):
            values,paths=baseline_snapshots(wm,rows[90],items,(2,4,8),policy,123)
            self.assertEqual(set(values),{2,4,8})
            self.assertEqual(len(paths[8]),8)
            self.assertEqual(paths[2],paths[4][:2])
            self.assertEqual(paths[4],paths[8][:4])
            self.assertTrue(all(0<=v<=1 for v in values.values()))

if __name__=="__main__":
    unittest.main()
