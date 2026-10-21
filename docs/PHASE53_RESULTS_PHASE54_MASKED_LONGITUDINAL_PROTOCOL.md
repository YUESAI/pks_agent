# Phase53 measured outcomes and Phase54 masked longitudinal WM protocol

**Evidence:** user-uploaded aggregate archive `phase53_masked.zip`, Dusky CPU job `330498`, produced at `2026-10-21T17:39:28Z`. Read-only frozen PPMI Prodromal TRAIN, source `MDS-UPDRS_Part_III_06Oct2026.csv`. No patient-level data were uploaded.

## Results directly supported by report

| Metric | Aggregate count |
|---|---:|
| Frozen TRAIN participants | 1,312 |
| Valid BL-anchored masked/item-aware motor patients | **1,165** |
| All usable complete or partially observed motor episodes | **2,702** |
| Verified 33-item observations with exact NP3TOT | **2,317** |
| Real 27-visible + six hands-on 101/UR episodes, any event | **286** |
| Other partial episodes with 27+ valid items | **43** |
| Other partial episodes with 10–26 valid items | **56** |
| BL-anchored adjacent episode pairs | **1,490** |
| All adjacent episode pairs including unanchored | 1,515 |
| Distinct patients with real R01 masked episodes | **229** |
| R01 masked + later verified V04 pair patients (Phase53) | **197** |

**By event:** BL complete=1,149 and partial ≥27=16;
R01 complete=36, remote masked=229, partial ≥27=12,
partial 10–26=43; V04 complete=605; OTHER complete=527,
remote-masked=51, partial 10–26=12. Some event-category
cells under ten are suppressed in the report, and are not
reconstructed here.

**BL-anchored patient sequence capacity:**

| At least this many visits | Phase51 total-score-only | Phase53 item-mask-aware |
|---|---:|---:|
| ≥2 | 647 | **658** |
| ≥3 | 375 | **426** |
| ≥4 | 120 | **272** |
| ≥5 | 17 | **98** |

For Phase53 specifically, BL-anchored distribution:
one visit=507, two=232, three=154, four=174,
five-plus=98. These sum to 1,165; 658 have at
least one usable followup and thus contribute supervised
future-observation information.

Primary anchored adjacent gap bins:
1–3 months `<10` (suppressed), 4–8=696,
9–15=740, 16–24=37, 25–36=16,
37+ = 0. Exactly **1,490** anchored pairs in summary.
An adjacent *pair* is not an independent patient, and several
pairs share the same person's history and outcomes.

Quality: 38 BL month mismatches, 607 source rows with missing
BL anchor, 48 rows from other medication states excluded,
28 R01 and 45 V04 events outside strict study windows,
fewer than ten rows with <10 valid items (suppressed).
Small-cell entries are not reconstructed or pooled with
patient-level data. The audit requires UNSPECIFIED medication
state, not confirmed no therapy.

**Do not directly interpret the 2,702 episodes as 2,702
independent patients or subtract them from Phase51's
2,405 valid `NP3TOT` keys.** Phase51's 2,405 includes
ON/OFF states separately, while Phase53 focuses on
UNSPECIFIED and also retains partial item records. The
denominators and QC protocols differ.

## One cohort discrepancy must be reconciled

Previously frozen Phase36/Phase41 reports described
**199** genuine R01 -> subsequent verified V04
Prodromal TRAIN paired people. Phase53 reports **197**
for the analogous-sounding pairing. The two audit
implementations differ in episode deduplication and
eligibility, including per-event versus per-month
conflict handling. The *reason for the two-person
difference has NOT yet been established.*

For every future training and results table, include
`cohort_protocol_version` and distinguish:
- Phase23–44 **locked R01 prognostic cohort**,
- Phase53 **item-mask longitudinal episode cohort**.

Do NOT retroactively revise the locked 199 paired
people or test metrics based on Phase53's count.
Run local patient-set reconciliation with **aggregated
reason counts only** before claiming a direct comparison.

## Phase54 objective: genuine longitudinal partially observed clinical WM

Shift from Phase50's two scalar inputs and nominal 24m total
to **patient-disjoint next-visit item prediction**
conditioned on actual past measured visits:

`q_phi(z_t | z_{t-1}, delta_t, observed_item_values_t,
        validity_mask_t, event_type_t)`

`p_theta(z_{t+1} | z_t, future_delta_t)`

`p_theta(X_{t+1,j}=0..4 | z_{t+1}, item_j)`

A 101/UR is its own structural unobservability category;
ordinary missing is also distinct from an observed 0.
During training, score future items **only where
measured 0..4**; do not label unknown future
items as zero. At inference, NEVER give the model
future visit values or observed-item masks. A future
time gap can be conditioned on only as a predeclared
scheduled horizon, or should be explicitly labelled
retrospective `time-conditioned prediction`.

**Model design:** item-token Transformer encoder,
irregular-time latent transition, recurrent
observation/belief update (RSSM/clinical-JEPA
inspired), categorical observation heads, uncertainty
calibration, and open-loop multiple-step rollouts.
These ideas are motivated by Dreamer-style
world models but do not constitute reproduction of
the Dreamer4 pretrained weights or a clinical
counterfactual action simulator.

**Baseline and evaluation plan:**
- patient-disjoint 5-fold nested TRAIN CV with inner
  epoch selection and FULL outer TRAIN refit;
- unit of training: patient and their observed transitions,
  not independently randomly split visit pairs;
- fixed item-wise marginal predictive distributions
  learned only from fit-patient records;
- per-item last-observed-score persistence where known;
- existing simple masked-item observation WM as
  a same-visit reference, not an equivalent temporal WM;
- primary outcome: **next-visit observed-item categorical NLL**
  aggregated by patient; also item-score MAE, multiclass
  Brier and calibration;
- secondary: next-visit R01->V04 verified onsite item
  prediction with strict cohort provenance, longer-horizon
  open-loop forecast and uncertainty;
- compare on matched available target item masks and
  identical heldout patients; use paired patient bootstrap,
  not treating 1,490 transitions as independent samples;
- report performance stratified by scheduled time horizon,
  number of prior visits and observation mask, with
  small-cell suppression.

**Qwen Doctor and verifier-guided RSI integration follows
only after the temporal WM passes next-visit heldout
observation checks.** Measurement actions update the
agent's belief; they do not themselves change a disease
state. Treatment-counterfactual modelling is NOT supported
by the current UNSPECIFIED medication data.

## Compute/data policy

Private raw PPMI and patient-level derived sequences stay
within the user's cluster Scratch; metadata reports are
aggregate only. CPU-first fitting is acceptable for
small latent models; existing offline
`pks_qwen46_cu118` handles Torch and CUDA 11.8.
No new model weights are needed to *start* Phase54.
Any later foundation checkpoint download happens ONLY
on an internet-enabled login node, with compute-node
offline loading. Original Phase44 TEST is consumed and
must never be reused for confirmation.

**Decision:** Phase53 proves a richer PARTIALLY OBSERVED
longitudinal modelling problem than Phase50/52, but not
by itself superior trained World Model performance.
