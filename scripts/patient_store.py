"""On-request Patient Agent database adapter; never reads restricted_labels."""
import sqlite3
from pathlib import Path

class PatientStore:
    def __init__(self, database, engine="duckdb"):
        if engine == "duckdb":
            import duckdb
            self.db = duckdb.connect(str(database), read_only=True)
        else:
            self.db = sqlite3.connect("file:"+str(Path(database).resolve())+"?mode=ro",uri=True)

    def public_catalog(self,cohort):
        """Lists possible exam types in a cohort, never individual patient's findings."""
        result=self.db.execute("SELECT concept FROM feature_catalog WHERE cohort=? ORDER BY concept",(cohort,))
        return [r[0] for r in result.fetchall()]

    def order_test(self,cohort,patient_id,concept,event_id="BL"):
        """Disclose only the explicitly ordered test. Patient never volunteers data."""
        allowed=self.db.execute("SELECT 1 FROM feature_catalog WHERE cohort=? AND concept=?",
                                (cohort,concept)).fetchone()
        if not allowed:return {"status":"unknown_test","concept":concept}
        values=self.db.execute("""SELECT value_raw,exam_date,quality_status,medication_state,
             source_file,source_row FROM observations
             WHERE cohort=? AND patient_id=? AND concept=? AND event_id=?
             ORDER BY source_file,source_row""",(cohort,patient_id,concept,event_id)).fetchall()
        if not values:return {"status":"not_recorded","concept":concept,"event_id":event_id}
        if len(values)>1:return {"status":"requires_duplicate_resolution","concept":concept,
                                "event_id":event_id,"n_records":len(values)}
        value,day,quality,medication,source,row=values[0]
        if quality!="observed_unverified":
            return {"status":"quality_review_required","concept":concept,"event_id":event_id,
                    "quality":quality}
        return {"status":"recorded","concept":concept,"event_id":event_id,"value_raw":value,
                "exam_date":day,"medication_state":medication,
                "source_file":source,"source_row":row}

    def close(self):
        self.db.close()
