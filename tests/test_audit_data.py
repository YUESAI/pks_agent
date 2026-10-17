"""Small local fixture tests. Run: python -m unittest discover -s tests -v"""
import csv
import tempfile
import unittest
from pathlib import Path

from scripts.audit_data import main


class AuditTests(unittest.TestCase):
    def test_scans_separate_directories_without_exporting_patient_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "raw"
            (root / "PPMI" / "clinical").mkdir(parents=True)
            output = base / "project" / "outputs" / "audit"
            file = root / "PPMI" / "clinical" / "exam.csv"
            with file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["PATNO", "EVENT_ID", "NP3TOT", "APPRDX"])
                writer.writerow(["SECRET_PATIENT_9001", "BL", 25, "PD"])
                writer.writerow(["SECRET_PATIENT_9002", "V01", "", "HC"])

            rc = main(["--data-root", str(root), "--output-dir", str(output),
                       "--cohorts", "PPMI", "--max-rows", "2"])
            self.assertEqual(rc, 0)
            inventory = (output / "files.csv").read_text()
            cols = (output / "columns.csv").read_text()
            join = (output / "join_candidates.csv").read_text()
            self.assertIn("exam.csv", inventory)
            self.assertIn("NP3TOT", cols)
            self.assertIn("PATNO", join)
            self.assertIn("APPRDX", join)
            for report in output.iterdir():
                body = report.read_text(encoding="utf-8")
                self.assertNotIn("SECRET_PATIENT_9001", body)
                self.assertNotIn("SECRET_PATIENT_9002", body)

    def test_rejects_writing_to_raw_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            root.mkdir()
            with self.assertRaises(SystemExit):
                main(["--data-root", str(root), "--output-dir", str(root / "report")])


if __name__ == "__main__":
    unittest.main()
