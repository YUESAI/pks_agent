# Phase59–62: VERA-PD integrated WM × Qwen Doctor × verifier-guided RSI

**Decision date: 2026-10-22. Status: engineering-ready, scientifically unverified.**
See [frozen evidence ledger](EXPERIMENT_LEDGER_PHASE23_58.md) and
[original prospectus](PROSPECTUS_WM_RSI_MULTIAGENT.md).
The prior Phase44 final TEST is consumed. Use TRAIN patient-disjoint
development and seek a genuinely new independent cohort for any final
claim.

## Why begin integration NOW

We have: real Qwen3-4B offline Doctor on Dusky L40S; tool schema and
injection-resistant action parser; 33-item masked, irregular-time
temporal WM; strong item Markov and 27-item visible-score controls;
real R01→later V04 observed progression; Phase45 constrained
offline RSI-lite. Phase58 demonstrated **possible complementary
risk rankings** from a 55k-param WM64 and Markov 50/50 ensemble,
not reliable top-k gain. This is enough for a modular,
falsifiable integration experiment, not enough to
claim full joint RSI or clinically actionable causal WM.

## The crucial mathematical separation

**(A) Within-visit observation acquisition.** The Qwen Doctor
may ask for a recorded item. The PatientEvidenceService releases
an ACTUAL observed 0–4 score, or explicitly UNKNOWN/101/UR.
Its requested measurement action updates the agent's epistemic
belief, NOT actual disease physiology. The conditional
observation kernel should estimate:
`p_theta(x_{t,j} | history_before_visit, currently_observed_items, mask)`.
At Phase59, the original Phase45 categorical observation WM
can serve as an explicit weak fallback but not as a modern,
trained clinical joint distribution. The Phase48 neural masked
Transformer observation model is a plausible higher-capacity
candidate and should be *adapted/evaluated* using a common
interface; its outcome-conditioned six-score head is NOT
identical to the Phase54 longitudinal WM head.

**(B) Across-visit disease-state prediction.** After actual
observations update the belief, the temporal WM predicts
`p_theta(x_{t+Δ} | previous_real_observations, Δ)`.
The Phase54/58 masked recurrent model is a small-data prototype,
not Dreamer4 pretrained weights. For real R01 forecasting,
the actual future V04 month is only known in retrospective
time-conditioned evaluation. A prospective test would have
to provide a scheduled, prespecified horizon.

**NO LEAKAGE:** Do not query the actual V04 future six hidden
items from a real R01 Patient tool. In the *separate*
same-V04 masking simulation, asking a visible 27-item V04
question may reveal a real score; hidden six items are NEVER
a tool action. A true R01 action may only replay R01-recorded
fields, not magically reveal contemporaneous onsite measurements.

## Phase59 — testable integrated end-to-end read-only loop

Build a common in-process immutable `ObservationEvent` carrying
`patient_hash_local`, visit marker, item status (measured/UR/absent),
score, provenance, action cost and allowed-to-see timestamp.
Patient-facing tool **never gets future ground-truth labels**,
and handles invalid/duplicate/unavailable actions by rejecting
rather than inventing a replacement measurement.

Standardized API boundaries:

```python
patient.available_actions() -> tuple[str, ...]
patient.request(item: str) -> GroundedObservation | Unavailable
observation_wm.predict_unasked(belief, item) -> p(score 0..4)
temporal_wm.predict_future(belief, horizon_months) -> FutureDistribution
planner.propose_next(belief, allowed, estimates, remaining_budget) -> Action
verifier.validate_action(...) -> PassOrFailure
verifier.evaluate_heldout_episode(...) -> VerifiedMetrics
```

Doctor inputs: previously released observed evidence,
unverified WM probability predictions marked as predictions,
hard allowed-actions schema and remaining budget. It does NOT
see site truth, patient ID, split membership or loss labels.

**Qwen3-4B-Instruct-2507** on offline Dusky L40S via existing
`pks_qwen46_cu118` and
`/scratch/ylong66/models/Qwen--Qwen3-4B-Instruct-2507`.
WM/patient/verifier can run CPU; no VLM because these
Motor rows are structured scores and contain no visual
observations. Add VLM only when eligible actual images
or video and an explicit imaging decision task exist.

Initial proof-of-mechanism: **same-V04 27 visible item masking
simulation with q=4/8/12/16** and real Qwen decisions.
Use all 605 complete Train V04 patients via nested patient
CV, with a fixed budget and no peeking at 6 hidden labels.
Compare plain Qwen, Qwen + simple observation WM,
Qwen + stronger observation WM, Qwen + temporal future
value advice, greedy information gain, simple visible-sum,
and compute-matched rules. Note visible 27 sum sees 27
scores: it is a high-information reference, not q-matched.

For true real-R01→future-V04 task, **all 27 visible R01
scores are already recorded at R01**; asking 27 separately
does not produce new clinical information. Use it as a
forecast/referral gate and do not counterfeit a sequential
measurement gain. Real clinical action value requires
additional actual recorded selectable modalities/actions.

## Phase60 — real model-based imagined information planning

Replace mere point risk display with explicit likelihoods
for candidate recorded **within-visit observable actions**.
For each candidate action, integrate over 0..4 hypothetical
scores (not fabricated actual answers) to estimate expected
uncertainty or decision loss after receiving that action.
Use **multi-step rollouts** with budget=4/8/12 and
belief re-updating on real observed responses.

A critical property test: zero/UR missingness cannot be
silently imputed; imaginary predicted samples never become
the Patient tool's observed scores or verifier's truth.
Report actual measured-item likelihood, calibration,
question-set overlap, counterfactual prediction sensitivity,
cost per action/token, and top-k selection on real
masked V04 outcomes.

## Phase61 — versioned joint WM/policy RSI

RSI iteration = a **proposal and verified acceptance**,
not unconstrained test-set-facing self-rewriting:

1. Mine de-identified failures from TRAIN-internal
   FIT/DEV only (WM calibration, action informativeness,
   invalid tool call, cost, prediction loss).
2. Qwen Evolver proposes a versioned change to *one*
   of: structured Doctor prompt/strategy; WM observation
   loss/mask policy; calibrated risk combiner; planning
   hyperparameters; or alternate lightweight WM candidate.
3. Train/evaluate candidate on inner TRAIN development
   under strict fixed resource budget. Compute
   prespecified `S = gain_at_equal_action_budget -
   λ_action*cost - λ_model*forecast_loss -
   λ_hallucination*invalid_rate` with constants frozen
   before evaluations.
4. Independent deterministic Verifier checks provenance,
   patient disjointness, no test leakage, model calibration,
   cost limits and paired score; **reject or accept** and
   snapshot old/new configs, parent hashes and reasons.
5. Repeat only for a fixed small number of rounds (e.g.
   3–5). Use an additional untouched **TRAIN-only audit
   fold** once *after* development; do not select rounds
   using its labels.
6. Clear ablations: fixed WM/fixed Doctor; policy-only RSI;
   WM-only RSI; independently alternating upgrades;
   **joint coupled RSI**; random mutation; unverified
   self-reflection. Equal tokens, training steps,
   query actions, selection quota and GPU hours.

Any configuration-search advantage must be interpreted as
an exploratory development result. Prevent the evolving
LLM from using private patient-level records as prompt
content or uploading intermediate samples.

## Phase62 — external confirmation & paper

Lock model weights and hashes, cohort and outcome definition,
action budget, primary comparison, confidence interval,
small-cell suppression and analysis plan BEFORE accessing
new independent BioFIND/private cohort subjects.
Licensing, study selection and cross-study harmonization
must be checked first. Do not reuse Phase44 TEST for
architecture claims. Main headline must be about a
**verifiable co-improvement algorithm**, not simply a
larger generative model or Agent roleplay.

## Suggested development sequence (no fictitious Slurm job)

Phase59 first deliver:
- `phase59_patient_tools.py` with no-fabrication tool contracts;
- `phase59_belief_interfaces.py` for observation and future heads;
- `phase59_qwen_doctor.py` using existing Qwen adapter;
- `phase59_verified_loop.py` with strict split-safe tests;
- Dusky CPU synthetic test and Dusky L40S offline pilot job;
- aggregate-only traces and metrics; no patient-level data export.

Then Phase60 imagination-planning and Phase61 actual RSI.
Do **not** run an undeveloped `phase59` Slurm command yet;
the code and tests must exist and be verified first.

### Reuse established cluster assets

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
"$PY" -m unittest tests.test_phase46_qwen_doctor \
  tests.test_phase54_masked_temporal_wm -v
# Only existing jobs, not future Phase59:
# sbatch slurm/phase46_dusky_qwen_doctor.sbatch
```

No new downloads needed to begin Phase59. **Compute nodes
have no network**; only networked login nodes may prefetch
later foundation-model weights to Scratch for offline use.
