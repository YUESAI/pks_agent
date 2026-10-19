# Prospective research proposal: World Model × RSI × Doctor–Patient Agents

Created 2026-10-19. **Unvalidated hypothesis**, not a retrospective result.
Complements (does not change) the frozen Phase23–44 research snapshot.
The previously untouched TEST split has already been consumed; all new
model/architecture comparisons require fresh independent data.

## Working title

**VERA-PD: Verifiable Self-Evolution of World-Model-Guided Clinical Agents**

## Central hypothesis

A shared model of clinical *state, observation availability and future
measurement uncertainty* can improve the selection of follow-up actions.
A verifier-guided RSI loop can recursively improve **both** the world
model and the decision policy without allowing simulated observations
to masquerade as real patient measurements. The novelty to test is
**coupled improvement**, not mere presence of three familiar modules.

## Architecture

Patient Evidence Service: strictly grounded observation masks, visit
timestamps and measured values, coded missingness and provenance.
Doctor Agent: chooses clinically feasible requests, stop or defer;
consumes belief-state uncertainty and a fixed action-cost schedule.
World Model: predicts distributions of future recorded observations
conditional on history, elapsed time, modality and candidate action.
Verifier: checks factual observability, no-leakage invariants,
calibration and realized held-out measured outcomes.
RSI Evolver: proposes versioned changes to policy prompts, memory,
skill graph, controller or model update rules. Candidate must pass
a separate verification split and an independent audit before promotion.

Two nested loops:

1. Inner patient loop: doctor requests information -> evidence service
   either releases a *recorded* admissible observation or UNKNOWN ->
   WM updates beliefs -> choose next action subject to budget.
2. Outer evolution loop: collect verified failure modes -> propose
   a versioned policy/WM change -> offline experiments on TRAIN
   only -> independent verification and promotion or rollback.

Core interaction target: does a better WM change what the Doctor chooses
to observe, and does the resulting verified observation reduce WM error?
Use ablations for model-only, policy-only, alternating independent
updates and fully coupled updates. Require per-round held-out gains
and diminishing/growing cost accounting; guard against self-confirmation.

## Formal objective (candidate, not yet fitted)

Let h_t be the available clinical history, m_t the observation mask,
a_t an available measurement action and y_t a *recorded* outcome.
The world model predicts p_theta(y_next,m_next | h_t,a_t,delta_t,modality).
The constrained policy pi_phi(a | h_t, belief_theta, budget) selects
actions. A possible risk-aware objective is expected *verified*
decision loss plus lambda times information cost, with uncertainty
penalties for unsupported action-outcome counterfactuals.
RSI optimizes versioned (theta, phi, skills, prompts) with a
hard constraint: acceptance must be based on independent measured
outcomes and cannot be decided by the evolving agent's own rating.

## Competing prior work

- MedAgentSim (MICCAI 2025) has doctor/patient/measurement agents and
  self-improvement already; our novelty cannot be just this combination.
- Multimodal AMIE (Nature Medicine 2026) has state-aware, multimodal
  simulated dialogue with ablations; no novelty in dynamic dialogue alone.
- MedEnv / DiagWorld (MICCAI 2026) supplies large longitudinal clinical
  interaction simulation; a generic EHR world alone is insufficient.
- AIDE^2 (2026 preprint) demonstrates self-modifying research agents;
  simply iterative code rewriting is not new either.

Research must demonstrate *new* verifier-grounded, coupled
world-model/policy improvement for modality-dependent clinical
observability and action budgets, and benefits beyond fixed baselines.

## Available data, but no invented counterfactual truth

- PPMI Train: 605 complete V04 MDS-UPDRS III exams support a
  same-visit six-item masking **simulation** with actual labels.
- Real remote R01 has many six-item 101/UR statuses, but their
  contemporaneous hidden clinical scores are never observed.
- BL/12/24 complete Train sequences: only 332. All 6/12/24: only 25.
  A high-capacity neural disease dynamics model may overfit.
- BioFIND, Fox Insight and authorized private data require cohort-specific
  codebooks, licensing checks and feature/visit harmonization before
  making generalization claims.

Rules: a simulator may sample **hypothetical** trajectories for
training stress tests, but those trajectories must be visibly tagged
synthetic, never counted as measured ground truth or used to validate
real clinical benefit. Off-policy counterfactual evaluation requires
recorded actions, overlap and credible assumptions; otherwise report
only supported masked-observation simulations.

## Hypotheses that could support a strong ML paper

H1: uncertainty-aware WM planning improves information efficiency
over the best simple visible-item sum rank and standard active learning.
H2: coupled WM+policy RSI improves more consistently than independent
WM updates, fixed-WM policy tuning, fixed-prompt self-reflection,
or random proposal search at equal experimental compute.
H3: improvements transfer to held-out patients and a genuinely
independent cohort; avoid declaring repeated TRAIN or consumed
Phase44 TEST observations independent evidence.
H4: verifier grounding reduces fabricated clinical responses and
prevents self-generated data from becoming a self-confirming shortcut.

## Minimal experiment matrix

Controls: no agent/rules; visible-score rank; supervised predictor;
single-agent with the same tools; multi-agent without WM;
multi-agent with frozen WM; multi-agent with RSI but frozen WM;
multi-agent + WM + one-pass tuning; full joint WM+RSI system.
Also isolate components of RSI (failure mining, proposal, verifier,
accept/rollback) and WM (temporal uncertainty, mask/modality model,
counterfactual simulator). Fix tokens, model backbone and action costs.

Metrics: risk/diagnostic outcome quality, cost-normalized information
gain, selected-test regret, calibration, probabilistic predictive error,
change across verified recursion rounds, gain per compute/token,
hallucinated-observation frequency and held-out robustness.

## Practical gating plan

Gate 1: proof-of-mechanism using V04 same-visit masked outcomes and
verified tool calls; explicit missingness and non-LLM baselines.
Gate 2: evaluate a small probabilistic WM with strict patient splits,
compared against persistence and the 50/50 BL/12 smoothing baseline.
Gate 3: engineer offline RSI proposal/verification loop and compare
uncoupled and coupled updates under fixed budget; demonstrate actual
verified iterative gains, not training-set fitting.
Gate 4: obtain an independent dataset/split for final confirmation;
the Phase44 TEST is already consumed, so no fresh claim from it.

Target contribution: a generalizable algorithm and benchmark for
**verifiable recursive co-improvement of predictive and decision
components under partially observed clinical evidence**. Its merit
is empirical and algorithmic, not based on the popularity of WM/RSI.
