"""On-request Doctor–Patient retrospective evidence API.

Doctor can see cohort-wide action names, but cannot access labels or
patient-specific availability until an action is requested.
This is an API separation, NOT a database-level security boundary.
"""
import sqlite3
from pathlib import Path


class DoctorPatientEnv:
    def __init__(self, manifest_db, source_db, cohort, patient_id, source_engine="duckdb"):
        self.cohort = cohort
        self._patient_id = patient_id
        path = Path(manifest_db).resolve()
        self._manifest = sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True)
        if source_engine == "duckdb":
            import duckdb
            self._source = duckdb.connect(str(Path(source_db).resolve()), read_only=True)
        elif source_engine == "sqlite":
            self._source = sqlite3.connect(
                "file:" + str(Path(source_db).resolve()) + "?mode=ro", uri=True)
        else:
            self._manifest.close()
            raise ValueError("Invalid source_engine")
        if not self._manifest.execute(
            "SELECT 1 FROM episode_candidates WHERE cohort=? AND patient_id=?",
            (cohort, patient_id)
        ).fetchone():
            self.close()
            raise ValueError("Unknown patient episode")
        self.queries = []

    def list_actions(self):
        """Return only the general catalog (not a patient's recorded tests)."""
        return [r[0] for r in self._manifest.execute(
            "SELECT action_key FROM action_catalog WHERE cohort=? ORDER BY action_key",
            (self.cohort,)).fetchall()]

    def order_test(self, action_key):
        if not self._manifest.execute(
            "SELECT 1 FROM action_catalog WHERE cohort=? AND action_key=?",
            (self.cohort, action_key)).fetchone():
            return {"status": "invalid_action"}
        entry = self._manifest.execute("""
            SELECT a.status,a.selected_visit,a.source_file,a.source_row,c.concept
            FROM episode_actions a JOIN action_catalog c
              ON c.cohort=a.cohort AND c.action_key=a.action_key
            WHERE a.cohort=? AND a.patient_id=? AND a.action_key=?
            """, (self.cohort, self._patient_id, action_key)).fetchone()
        if not entry:
            return {"status": "not_recorded"}
        status, visit, filename, line, concept = entry
        self.queries.append(action_key)
        if status != "ready":
            return {"status": "not_available", "action": action_key}
        values = self._source.execute("""
            SELECT value_raw,exam_month,date_precision,medication_state,quality_status
            FROM observations WHERE cohort=? AND patient_id=? AND event_id=? AND concept=?
              AND source_file=? AND source_row=?
            """, (self.cohort, self._patient_id, visit, concept, filename, line)).fetchall()
        if len(values) != 1 or values[0][4] != "observed_unverified":
            return {"status": "source_consistency_error"}
        score, month, precision, state, _ = values[0]
        return {
            "status": "recorded", "action": action_key, "value": score,
            "visit": visit, "recorded_month": month, "date_precision": precision,
            "medication_state": state or "UNSPECIFIED",
            "disclaimer": "Retrospective recorded result, not real-time patient testimony",
        }

    def close(self):
        if getattr(self, "_manifest", None):
            self._manifest.close()
            self._manifest = None
        if getattr(self, "_source", None):
            self._source.close()
            self._source = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
