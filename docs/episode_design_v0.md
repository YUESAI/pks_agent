# Episode audit design v0

These figures come from the user-uploaded full clinical_profile.zip, not an inferred full diagnosis cohort:

| Cohort | Anchor subjects | MoCA subjects at any visit | Motor subjects at any visit | RBD subjects at any visit |
|---|---:|---:|---:|---:|
| PPMI | 9,312 | 5,714 | 5,446 (UPDRS-III total) | 5,259 |
| BioFIND | 232 | 226 | 225 (post-dose gait item) | 226 |

PPMI diagnosis PRIMDIAG covers 5,494 patients at any visit, not necessarily BL. PPMI UPSIT mostly uses SC, while several other instruments mostly use BL.
PPMI UPDRS-III has 7,056 extra PATNO/EVENT_ID rows, and UPSIT has 1,174. Do not deduplicate before reviewing medication condition, exam date and imputation flags.

Tool scripts/episode_audit.py computes 3 views:

- strict_BL: only EVENT_ID=BL.
- SC_BL_candidate: BL or SC, with no temporal verification. This is an upper bound and must NOT be used for training.
- SC_BL_date_checked: BL or SC with INFODT inside 180 days prior to BL diagnosis-form date. Still does not establish BL test predates diagnostic decision.

Official cohort/diagnosis code decoding must be checked from study code lists before PD versus HC; all outputs are aggregate only and no patient IDs are exported.

BioFIND post-dose MDS-UPDRS item NP3GAIT is not a directly interchangeable PPMI UPDRS-III total. Fox Insight uses self-report and remains separate for now.

Before building DuckDB: validate code meanings, missing-value sentinels, medications, UPSIT imputation, time ordering and target leakage; split patients rather than visits.