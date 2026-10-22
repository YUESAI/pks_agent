# Phase57 real R01 visible-27 sum result (2026-10-22)

**Source:** user-provided `phase57_visible27.zip` from Dusky
job **330756** (`generated_utc=2026-10-22T00:27:21Z`).
Only frozen Prodromal PPMI TRAIN. The original Phase44 TEST
was consumed and never reopened.

## Observed standalone metrics

The exact strict Phase56 R01 -> later verified V04 cohort
has **197** eligible people with **57** later hidden-six-positive.
The 20% top-k budget is **39 patients**.

For a **direct sum of the 27 actually scored R01 visible items**:

- captured **22/57** future positives;
- positive recall **0.385965 (38.60%)**;
- AUROC **0.675188**;
- expected random positives at exactly k=39: **11.28426**;
- random expected recall **0.19797 (19.80%)**;
- standalone 500-bootstrap recall 95% **[0.256217,0.474545]**;
- standalone AUROC bootstrap 95% **[0.579126,0.755417]**.

The ranking score is a **sum, not a calibrated future-event
probability**, so no probabilistic NLL/Brier is assigned to it.

## Point comparison with independently uploaded Phase56 metrics

| Decision rule | AUROC | future positives captured / 57 (k=39) |
|---|---:|---:|
| Inner-tuned item Markov | 0.650940 | **23** |
| R01 27-visible score sum | 0.675188 | **22** |
| Temporal latent WM | 0.679261 | **19** |
| V04 item marginal prior | 0.436842 | **10** |

**Do not claim these differences statistically reliable.**
Phase57's bootstrap intervals concern **only one rule**,
not patient-*paired* differences between rules. In fact, a
one-person gap between Markov and visible sum on the 39-person
selected set is very small.

The WM-vs-Markov future-six categorical NLL difference in
Phase56 was +0.001341, patient-paired 95%
[-0.024964,+0.025792]. That result does not measure whether
their top-k selections differ significantly. Nor does AUROC
tell whether top-k utility must improve.

Important: the Phase57 summary says it is the same
strict cohort as Phase56 **conditional on matching the
Git commit and eligibility implementation**; the aggregate
197/57/39 counts agree with Phase56. Patient-level set
identity must be checked locally for hard pairwise claims.

## Recommended next experiment

A **single-pass Phase58 matched decision audit** should
regenerate all methods' OUTER-OOF predictions on the same
strict frozen TRAIN R01->V04 people, then compute:

1. Equal-budget top-k capture/recall on exactly the same
   patients using the same deterministic tie break.
2. **Patient-paired bootstrap** CIs for WM-minus-visible
   and WM-minus-Markov recall and AUROC, recomputing
   top-k on each patient bootstrap.
3. Pairwise top-k selected-list overlap and agreement.
4. Decision diagnostics stratified by available R01 Motor
   burden, with suppression of small cells.

No patient IDs, scores, full predictions or actual
individual selections should be exported. Models and
baselines must only fit non-heldout TRAIN people;
all evaluation candidates are frozen at this stage.

This is further **exploratory TRAIN** work, not independent
confirmation. A prospective, preregistered test on a truly
new cohort is required before clinical generalization claims.

**Research priority after this check:** action-conditioned
measurement planning and verifier-gated WM/Doctor/RSI
with actual belief updates, plus longitudinal multi-step
open-loop uncertainty evaluation. A better global NLL
alone is insufficient justification for complex WM.
