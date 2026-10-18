"""Phase 12: minimal budgeted, state-blind PPMI Doctor–Patient replay API.

The Doctor sees a fixed action catalog, never cohort labels, patient IDs,
source dates, drug ON/OFF states or reasons why tests were not recorded.
The underlying PPMI observations remain retrospective and UNVALIDATED.

Application-level hiding is NOT a database security boundary. Protect source
and manifest DBs and do not hand direct DB connections to Doctor models.
"""
from __future__ import annotations

from scripts.doctor_patient_env_v0 import DoctorPatientEnv

# Stable, clinically modest action space. RBD/SCOPA are individual items,
# not full clinical questionnaire scores. Age is requested, not volunteered.
ACTION_SPECS = {
    "age": {"cost": 0, "internal": ("age_at_visit",)},
    "sleep_item": {"cost": 1, "internal": ("rbd_vivid_dreams",)},
    "cognitive_test": {"cost": 1, "internal": ("moca_total",)},
    "autonomic_item": {"cost": 1, "internal": ("scopa_aut_item1",)},
    "motor_exam": {
        "cost": 2,
        # Deterministic treatment-state preference; the state is withheld
        # from Doctor but selection could still confound value distribution.
        "internal": (
            "updrs3_total@UNSPECIFIED",
            "updrs3_total@OFF",
            "updrs3_total@ON",
        ),
    },
    "smell_test": {"cost": 2, "internal": ("upsit_total",)},
}


class BudgetedDoctorPatientEnv:
    def __init__(self, manifest_db, source_db, patient_id,
                 budget=5, source_engine="duckdb"):
        if budget < 0:
            raise ValueError("Budget cannot be negative")
        self._underlying = DoctorPatientEnv(
            manifest_db, source_db, "PPMI", patient_id, source_engine)
        self._ready_keys = set()
        try:
            for action, status in self._underlying._manifest.execute(
                "SELECT action_key,status FROM episode_actions "
                "WHERE cohort=? AND patient_id=?",
                ("PPMI", patient_id),
            ):
                if status == "ready":
                    self._ready_keys.add(action)
        except Exception:
            self._underlying.close()
            raise
        self.budget = int(budget)
        self.spent = 0
        self.requested = set()
        # Never export these internals in a transcript supplied to a Doctor.
        self._internal_selection = []

    def catalog(self):
        """Globally fixed catalog. Does NOT reveal patient-specific readiness."""
        return {action: spec["cost"] for action, spec in ACTION_SPECS.items()}

    def request(self, action):
        if action not in ACTION_SPECS:
            return {"status": "invalid_action"}
        if action in self.requested:
            return {"status": "already_requested", "action": action}
        cost = ACTION_SPECS[action]["cost"]
        if self.spent + cost > self.budget:
            return {"status": "insufficient_budget", "action": action,
                    "remaining_budget": self.budget - self.spent}

        # Any valid action incurs cost whether recorded or not.
        self.requested.add(action)
        self.spent += cost
        chosen = next((raw for raw in ACTION_SPECS[action]["internal"]
                       if raw in self._ready_keys), None)
        if chosen is None:
            return {"status": "not_available", "action": action,
                    "cost": cost, "remaining_budget": self.budget - self.spent}
        result = self._underlying.order_test(chosen)
        if result.get("status") != "recorded":
            return {"status": "not_available", "action": action,
                    "cost": cost, "remaining_budget": self.budget - self.spent}
        self._internal_selection.append(chosen)
        # No raw source files, dates, medication-state flags or alternate ON/OFF
        # action names appear in Doctor-visible responses.
        return {"status": "recorded", "action": action,
                "value": result["value"], "cost": cost,
                "remaining_budget": self.budget - self.spent}

    def close(self):
        self._underlying.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
