# Reproduction & provenance: Phase 23–44

This playbook accompanies [RESEARCH_SNAPSHOT_PHASE23_44.md](RESEARCH_SNAPSHOT_PHASE23_44.md).
The following is a record of the *completed* experiments, not an invitation
to repeat the held-out final TEST or to conduct more hyperparameter search.

## Environments and filesystem

- Project checkout: `/home/ylong66/workspace/pks_agent`
- Private cohort root: `/scratch/ylong66/data/pks/data`
- PPMI frozen longitudinal manifest: `outputs/longitudinal_manifests/prodromal_motor_v0.sqlite`
- Reports reside in `outputs/phaseNN_.../`, all excluded from version control.
- Research Python environment used on the cluster: `ms127`, Python 3.11.
- Original data `PPMI/`, `BioFIND/`, `FoxInsight/` remain private and are never added to this Git repository.

To track the precise source revision, execute the following **metadata-only** commands
(no patient data or test outcomes are re-read):

```bash
cd /home/ylong66/workspace/pks_agent
git status --short
git branch --show-current
git rev-parse HEAD
python --version
python -m pip freeze > outputs/environment_phase23_44_freeze.txt
sha256sum outputs/longitudinal_manifests/prodromal_motor_v0.sqlite > outputs/private_manifest.sha256
```

These metadata outputs may be stored locally. **Do not publish the manifest
file itself, or any raw patient table**. Even aggregate reports may allow
small-cell reconstruction. Retain original report archives privately.

## Integrity and archive index

See `phase23_44_report_index.csv`: SHA-256 checksums were computed from the
user-supplied original ZIP artifacts. All timestamps are from the internal
`summary.json` (`generated_utc`), not independently audited computer clocks.
Phase34 ZIP contains both the corrected Phase33 and Phase34 reports.
Phase33's original count of six item columns was a variable-shadowing
metadata bug; trust the corrected Phase33 count of 33 `NP3*` item fields.

Check a locally preserved report archive with:

```bash
sha256sum pks_phase44_report.zip
unzip -l pks_phase44_report.zip
```

Do not commit report ZIPs or their potentially identifying derived percentages.

## Existing scripts, ordered by scientific objective

| Objective | Original scripts | Outputs used |
|---|---|---|
| Frozen Prodromal cohort | `scripts/phase23_manifest.py` | `outputs/phase23_prodromal_manifest/summary.json`, `pairs_by_split.csv` |
| One-step Motor and added modalities | `scripts/phase24_motor_baselines.py`, `scripts/phase26_multimodal_oof.py`, `scripts/phase27_fair_incremental.py` | `outputs/phase27_fair_incremental/nested_multimodal_oof.csv` |
| Multi-step feasibility, simple dynamics and smoothing | `scripts/phase28_longitudinal_signal.py`, `scripts/phase29_sequential.py`, `scripts/phase30_smoothing.py` | `outputs/phase30_smoothing/model_metrics.csv` |
| Missingness / remote visit provenance | `scripts/phase31_motor_repetitions.py`, `scripts/phase32_exclusion_forensics.py`, `scripts/phase33_visit_audit.py`, `scripts/phase34_remote_motor.py`, `scripts/phase35_remote_101.py` | `outputs/phase32_exclusion_audit/summary.json`, `outputs/phase35_remote_101/remote_six_item_codes.csv` |
| Matched V04 simulated partial observations | `scripts/phase36_masked_site_feasibility.py`, `scripts/phase37_masked_site_benchmark.py`, `scripts/phase38_budgeted_acquisition.py` | `outputs/phase37_masked_site_benchmark/masked_site_oof_metrics.csv`, `outputs/phase38_budgeted_acquisition/budgeted_acquisition_curve.csv` |
| Real R01 vs site shift and prognostic ranking | `scripts/phase39_remote_domain_shift.py`, `scripts/phase40_budget_calibration.py`, `scripts/phase41_future_v04_risk.py`, `scripts/phase42_missing_followup_bounds.py` | `outputs/phase41_future_risk/prospective_rank_discrimination.csv`, `outputs/phase42_missing_followup_bounds/missing_outcome_budget_bounds.csv` |
| One-shot validation and final testing | `scripts/phase43_locked_validation.py`, `scripts/phase44_locked_final_test.py` | `outputs/phase43_locked_validation/locked_validation_policy.csv`, `outputs/phase44_locked_final_test/locked_validation_policy.csv` |

**Note on exact reproducibility:** current HEAD records the present codebase
after bug fixes. Exact reproduction of *historical* numbers also requires
pinning the code commit as it existed when each report was produced,
plus preserving the original source and private manifest checksums.
The report hash index alone does **not** prove that code, raw source data and
Python dependencies were bitwise identical across runs. The environment
freeze above is prospective metadata capture, not an original environment log.

## Final frozen policy (do NOT retune)

The binding policy is explicitly documented in
`phase43_frozen_validation_protocol.md` and
`phase44_final_test_protocol.md`. For all eligible R01 patients with
complete observable partial Motor inputs, compute

```text
S_i = sum of 27 valid remotely observable MDS-UPDRS Part III item scores.
Order patients by (-S_i, SHA256("Phase41|20261018|{patient_id}")).
k = floor(0.20 * total eligible R01 patients + 0.5).
Select first k patients. Missing future V04 outcomes remain UNKNOWN.
```

*Reference outcome*, six hands-on items' sum > 0 at the **later** onsite V04,
is not available at R01. For `a` selected known positives, `p` all known
positives, `v` **unselected** missing outcomes, `Recall_lower=a/(p+v)`.
Random exact-budget expected recall `k/N`. Primary descriptive comparison
is `100 * (Recall_lower-k/N)` percentage points.

`Phase43`: VAL, N=49 eligible, 41 later observed, 20% budget k=10,
Recall lower 25.00% vs random 20.41%, +4.59 pp.

`Phase44`: TEST, N=37 eligible, 34 later observed, 20% budget k=7,
Recall lower 30.77% vs random 18.92%, +11.85 pp.

Neither difference is a p-value nor a proven clinical referral utility.
The current TEST was irrevocably **consumed** on
`2026-10-19T02:55:59.614060+00:00`. **Never use it as a pristine final
evaluation again.** Do not re-run the actual Phase44 evaluator as routine
quality assurance.

## Reproduction checks that are safe now

Syntax-only checks, synthetic unit tests, and report-reading do not
re-open held-out clinical records:

```bash
cd /home/ylong66/workspace/pks_agent
python -m py_compile scripts/phase43_locked_validation.py scripts/phase44_locked_final_test.py
python -m unittest tests.test_phase43_locked_validation tests.test_phase44_locked_final_test -v
git rev-parse HEAD
```

To reproduce the reported figures **without accessing patient data**,
read the stored aggregate original ZIP reports and verify their checksums;
do not invoke `phase44_locked_final_test.py --confirm-final-test` again.
