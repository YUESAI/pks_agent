# Clinical mapping shortlist (2026-10-17)

This document is based on schema/column metadata from the user's structural audit of the actual local export. It is **not** a medically validated harmonization.

## Strategy

- **PPMI (development)**: subject ID `PATNO`; main cohort anchor `Participant_Status_06Oct2026.csv` with `COHORT` and `COHORT_DEFINITION`. This table has no `EVENT_ID`. Do not assume study cohort is the diagnosis at each visit.
- **BioFIND (external)**: `Screening_Demographics_08Oct2026.csv` contains `PATNO`, `EVENT_ID`, `APPRDX`; `Primary_Diagnosis_08Oct2026.csv` has `PRIMDIAG`.
- **Fox Insight (self-report shift)**: subject ID `fox_insight_id` (the first audit failed to recognize it), time coordinate `days_elapsed`, labels candidates `Users.csv: InitPDDiag` and `General.csv: CurrPDDiag`.

## Important candidate tables

| Concept | PPMI | BioFIND | Caveat |
|---|---|---|---|
| Clinically recorded diagnosis | `Primary_Research_Diagnosis:PRIMDIAG` | `Primary_Diagnosis:PRIMDIAG` | Must be hidden from Doctor |
| Cognitive testing | `Montreal_Cognitive_Assessment__MoCA__:MCATOT` | Same assay/field `MCATOT` | Check score definition |
| Motor exam | `MDS-UPDRS_Part_III:NP3TOT`, `NP3GAIT` | `MDS_UPDRS_Part_III__Post_Dose__:NP3GAIT` | Post-dose exam; not directly interchangeable |
| Smell test | `University_of_Pennsylvania_Smell_Identification_Test_UPSIT:TOTAL_CORRECT` | Not identified among first-pass clinical tables | PPMI remote tests separate |
| Sleep | `REM_Sleep_Behavior_Disorder_Screening_Questionnaire:DRMVIVID` | `REM_Sleep_Disorder_Questionnaire:DRMVIVID` | Code mapping required |
| Autonomic symptoms | `SCOPA-AUT:SCAU1` | Not confirmed | Item, not total |
| Age | `Age_at_visit:AGE_AT_VISIT` | Not confirmed | Do not infer from sensitive birthdates automatically |
| Patient history | `PD_Diagnosis_History:SXDT`, `PDDXDT` | `General_Medical_History:MHTERM` | Diagnosis history may leak target |

## Original research dictionaries are present

- PPMI: `Data_Dictionary_-_Harmonized_06Oct2026.csv`, `Code_List_-_Harmonized_06Oct2026.csv`.
- BioFIND: `Data_Dictionary_08Oct2026.csv`, `Code_List_08Oct2026.csv`.
- Fox Insight: `FoxInsightVariables.csv`, `FoxInsightValues.csv`.

Review definitions and missing-value codes before producing a cleaned modeling dataset.

## Structural audit findings

558 PPMI, 147 BioFIND, 78 Fox Insight files; 516 successfully sampled tables; 44,235 column entries. The initial structural issues comprise 10 duplicate header cases and 2 read failures. The `__MACOSX/._...` file is an Apple sidecar, not a workbook. The BioFIND LeWitt chemical ID key failed to load but is not a prerequisite for clinical profiling. The original audit sampled only 500 rows and the first Excel worksheet.

## Research safeguards

- Never expose `COHORT`, `APPRDX`, `PRIMDIAG`, `NEWDIAG`, or diagnosis-history outcomes as ordinary Doctor tools.
- Check `PATNO+EVENT_ID` **uniqueness and examination condition** before joining.
- Fox `days_elapsed` is not the same as PPMI `EVENT_ID`.
- Full selected-table coverage counts are produced by `scripts/profile_clinical.py`. Coverage at **any visit** is not same-visit joint availability.
- Do not fabricate tests absent from retrospective records or use WM-generated values as real held-out outcomes.
