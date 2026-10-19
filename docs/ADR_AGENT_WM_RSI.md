# Architecture decision: Agent, World Model, RSI

Date: 2026-10-19. Status: recommendation, not validated model comparison.

**Decision:** core = grounded partial observation + cost-aware sequential
decision making. Keep the fixed visible 27-item Motor score ranking as a
mandatory baseline. Implement Doctor/Patient dialogue only as a tested
interface. Do not make World Model (WM) or Recursive Self-Improvement (RSI)
required modules without independent evidence that they improve utility.

## Existing evidence

- Phase 30: the fixed mean of BL and 12m Motor had MAE 2.9081 on 332
  TRAIN trajectories, whereas two-timepoint Ridge had MAE 3.0694;
  learning disease transitions has not been established.
- Phase 37: predicting six masked items at the same V04: always-zero
  MAE 0.7785 versus 27-item Ridge 0.8477, though Ridge RMSE improved.
- Phase 38: at 20% simulated acquisition budget, simple visible-sum
  ranking captured 81/206 positives, and 27-item Ridge did not
  clearly improve on that ranking.
- Phases 41–44: real remote R01 visible total predicts later V04
  six-item abnormalities, with frozen 20% primary lift positive in
  TRAIN, VAL and final TEST. This does not establish referral efficacy.
- There is no experiment so far showing that a Multi-Agent system,
  RSI or WM outperforms a simple clinical score on this task.

## Minimal proposed architecture

Clinical observations with patient/time/provenance and missing-code masks
-> Patient evidence service (known measurements only; never invent a test)
-> uncertainty/risk state estimator
-> constrained Doctor decision policy (ask, measure, defer, abstain)
-> action-cost and availability verifier
-> observation update and audit log.

Patient Agent should begin as a deterministic fact service. An LLM can
optionally render natural language, with citations and leakage checks.
Doctor Agent may be rule based, single-LLM or multi-agent, but all must
receive identical available evidence and action budgets in experiments.

## Evidence gates

1. Verify visit modality, data-codebook semantics, treatment conditions,
   action costs and observation timestamps across cohorts.
2. Establish genuine multi-action sequential decisions and uncertainty;
   a one-shot budgeted ranking does not itself require multi-agent design.
3. Compare a simple single-controller policy, fixed ranking, and learned
   decision rules before running a multi-agent design.
4. Only add a WM if it beats persistence, historical averaging and
   simple probabilistic filters on independently evaluated predictive
   AND decision objectives; sparse three-visit records may limit this.
5. RSI remains optional OFFLINE candidate prompt/policy search with
   frozen rules, protected heldouts and no self-grading on test labels.
6. Evaluate tools, information, cost, tokens, latency and safety on equal
   footing; report ablations for each claimed agent component.
7. Phase44 TEST is CONSUMED. Any new final claim requires independent
   data or an appropriately separate prospective validation design.

## Distinct research tasks

- Same-visit V04 six-item masking: idealized acquisition with real
  contemporaneous V04 labels; not a true remote clinic trial.
- Real R01 -> later V04: observational future-risk prediction with
  unknown outcomes and selection-bias bounds; not causal referral effect.
- BL + 12m -> later 24m score: longitudinal forecasting with strong
  smoothing baseline, not automatically evidence of a biological WM.

Goal: a reproducible, privacy-preserving partial-clinical-observation
benchmark with resource-aware decisions and grounded uncertainty. Agents
or WM are contributions only when they measurably outperform the simple
policies on new independent evaluations.
