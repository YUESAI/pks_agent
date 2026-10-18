"""Synthetic Phase28 tests; no real patient records."""
import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from scripts.phase28_longitudinal_signal import (
    audit, read_train_manifest, severity_bin, change_bin
)

class Phase28Tests(unittest.TestCase):
    def test_bin_boundaries(self):
        self.assertEqual(severity_bin(None),"no_baseline_motor")
        self.assertEqual(severity_bin(3),"0_to_3")
        self.assertEqual(severity_bin(4),"4_to_7")
        self.assertEqual(change_bin(-10),"decrease_10plus")
        self.assertEqual(change_bin(0),"unchanged")
        self.assertEqual(change_bin(5),"increase_5_to_9")

    def test_train_only_multistep(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"private.sqlite"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE patients(pid TEXT PRIMARY KEY,split TEXT,baseline_month INTEGER,baseline_motor REAL)")
                db.execute("CREATE TABLE readiness(pid TEXT,horizon INTEGER,status TEXT,PRIMARY KEY(pid,horizon))")
                db.execute("CREATE TABLE targets(pid TEXT,horizon INTEGER,baseline_month INTEGER,target_month INTEGER,months_apart INTEGER,baseline_motor REAL,target_motor REAL,delta_motor REAL,PRIMARY KEY(pid,horizon))")
                for i in range(30):
                    pid="PRIVATE_TRAIN_"+str(i)
                    bl=float(i%14);anchor=2025*12+1
                    db.execute("INSERT INTO patients VALUES (?,?,?,?)",(pid,"train",anchor,bl))
                    for h in (6,12,24):
                        yes=(h==12 or (h==24 and i<20) or (h==6 and i<5))
                        db.execute("INSERT INTO readiness VALUES (?,?,?)",
                                   (pid,h,"paired" if yes else "no_in_window_target"))
                        if yes:
                            delta=2 if h==12 else 1
                            db.execute("INSERT INTO targets VALUES (?,?,?,?,?,?,?,?)",
                                       (pid,h,anchor,anchor+h,h,bl,bl+delta,delta))
                for split in ("val","test"):
                    pid="SECRET_"+split
                    db.execute("INSERT INTO patients VALUES (?,?,?,?)",
                               (pid,split,2025*12+1,100))
                    for h in (6,12,24):
                        db.execute("INSERT INTO readiness VALUES (?,?,?)",(pid,h,"paired"))
                        db.execute("INSERT INTO targets VALUES (?,?,?,?,?,?,?,?)",
                                   (pid,h,2025*12+1,2025*12+1+h,h,100,132,32))
            self.assertEqual(len(read_train_manifest(path)),30)
            out=Path(tmp)/"report"
            result=audit(path,out,10)
            self.assertEqual(result["strict_12m_paired_train_patients"],30)
            self.assertEqual(result["strict_12m_and_24m_paired_train_patients"],"20")
            self.assertEqual(result["all_three_6_12_24_paired_train_patients"],"<10")
            with (out/"train_multistep_coverage.csv").open() as file:
                rows=list(csv.DictReader(file))
            self.assertEqual(
                next(r for r in rows if r["metric"]=="12_and_24_months")["train_patients"],"20")
            for file in out.iterdir():
                body=file.read_text()
                for private in ("PRIVATE_TRAIN_","SECRET_val","SECRET_test"):
                    self.assertNotIn(private,body)

    def test_missing_target_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"bad.sqlite"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE patients(pid TEXT PRIMARY KEY,split TEXT,baseline_month INTEGER,baseline_motor REAL)")
                db.execute("CREATE TABLE readiness(pid TEXT,horizon INTEGER,status TEXT,PRIMARY KEY(pid,horizon))")
                db.execute("CREATE TABLE targets(pid TEXT,horizon INTEGER,baseline_month INTEGER,target_month INTEGER,months_apart INTEGER,baseline_motor REAL,target_motor REAL,delta_motor REAL,PRIMARY KEY(pid,horizon))")
                db.execute("INSERT INTO patients VALUES(?,?,?,?)",
                           ("FAKE","train",2025*12,2))
                for h in (6,12,24):
                    db.execute("INSERT INTO readiness VALUES(?,?,?)",
                               ("FAKE",h,"paired" if h==12 else "no_target"))
            with self.assertRaises(ValueError):
                read_train_manifest(path)

if __name__=="__main__":
    unittest.main()
