# Phase 44 — Final held-out TEST protocol (frozen before opening TEST)

Version: `phase44-test-v1` | Frozen: 2026-10-18 | Repository: `YUESAI/pks_agent`.

## Status and rationale

Phase43 one-shot Validation was positive on its **locked** 20% primary
metric, but only 49 validation patients had eligible R01, only 41 had a
known V04 future label, and just 13 were future-positive. The conservative
recall lower bound was 25.00%, vs 20.41% random expected recall (+4.59
percentage points). AUROC was 0.61401, AP was 0.42286. This evidence is
directionally encouraging but low precision and **not a significance test**.

The held-out TEST records have never been examined for this task. This
protocol fixes a **single final evaluation** and does not tune or select
anything based on the observed Validation outcome.

## Identical frozen policy and estimand

The sole primary analysis must be IDENTICAL to the previously registered
`docs/phase43_frozen_validation_protocol.md`, except querying
`patients.split='test'` rather than `patients.split='val'`:

- Prodromal PPMI subjects in the **original frozen Phase23 TEST split**.
- R01: visit month 4–8 after BL anchor; conservative unique-visit gating;
  exactly 27 MDS-UPDRS III remotely observable numeric item scores,
  six unrateable hands-on items, no valid R01 total.
- Future V04 outcome: complete same-visit-sum-verified 33-item onsite exam
  at month 9–15 after BL, strictly after R01; future **six hidden items'
  sum > 0**. No eligible V04 = unknown, not negative.
- Descending R01 observable **27-item sum**; no model fit, no age or
  future information, no post-hoc choice of features.
- Exact top-k quota `floor(0.20 * n + 0.5)` among **all eligible R01 TEST
  people**, before checking V04 outcome availability.
- Tie-break: ascending SHA256 UTF-8 of
  `Phase41|20261018|{patient_id}`, never exported.
- Primary: difference in percentage points between sharp missing-label
  future-V04-positive recall lower bound `a/(p+v)` and the expected recall
  `k/n` for random exact-quota allocation. Positive direction
  `lower_bound - k/n > 0` is a **descriptive directional criterion**,
  NOT a test of statistical significance.
- Secondary: 10/30/40/50% budget bounds, AUROC and tie-aware AP on the
  known-future-label subset only when both classes have >=10 members.
  None of the secondary metrics can replace the locked primary.
- No patient identifiers, dates, individual scores or decisions exported.
  Aggregates involving cells <10 are suppressed; check any published
  aggregate for possible secondary disclosure.

**No interim peeking, retuning, switching budgets, reruns with changed
codes, or adding new models based on final TEST outcomes.**

## Interpretive boundaries

- This is a **future** clinic-risk proxy, NOT same-day remote missing
  score recovery, clinical referral utility, value of information, or
  a causal treatment effect.
- The worst-case recall identification lower bound is not a statistical
  confidence bound. Real-world external validation and prospective design
  remain necessary.
- Validation and TEST cohorts may be too small for a stable inference;
  a negative or inconclusive final result must be reported as such.
- After executing the final TEST evaluation, mark this held-out split as
  consumed. Further experiments cannot characterize it as untouched.

The runner `scripts/phase44_locked_final_test.py` requires the explicit
`--confirm-final-test` flag. Do not execute that command until ready to
treat TEST as used permanently. The generated report directory must not
already exist.
