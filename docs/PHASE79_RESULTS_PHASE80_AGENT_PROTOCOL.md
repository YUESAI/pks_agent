# Phase79 full results -> Phase80 clinically grounded active evidence Agent

phase79_reports.zip, 20/20 complete, each 3 clinical tasks. 255
model rows, 1470 paired comparisons, 15 model calibration audits.
Repeated in-job reference predictions deduped before computing
fivefold patient-weighted metrics.

| model | Prodromal12 median MAE, mean RMSE, RPS | Prodromal24 median MAE, mean RMSE, RPS |
|---|---|---|
| Categorical | .7586,1.4794,.0245 | .8730,1.6937,.0276 |
| Hurdle | .7688,1.4738,.0248 | .8492,1.7042,.0280 |
| Ordinal proper-scored | .7620,1.4615,.0243 | .8519,1.6885,.0276 |
| DEV/AUDIT temperature-verified | .7517,1.4356,.0241 | .8571,1.6890,.0276 |

Calibration passed only one of fifteen independent audit events:
Prodromal12 fold1, temperature 0.75. Remaining fourteen audit
events selected identity or vetoed calibration. Ordinal RPS gain
over baseline categorical is small; do not claim significance.
All fivefold evaluations are reused original Prodromal TRAIN, not
new generalization confirmation.

## Phase80 experimental protocol

Pause further pure forecast-loss tweaking. Use same frozen per-fold
FIT-trained categorical Motor WM as starting belief; 5 patient
folds x 4 independently submitted CPU strategies:

0 q=0: no optional test evidence; identical model and patients.
1 q=1 fixed FIT-only-priority true baseline optional clinical test.
2 q=1 WM expected-KL-information-gain selected true baseline test.
3 q=1 Qwen3-4B-Instruct-2507 chooses from *available names* using
  FIT-derived hypothetical expected gains, followed by true
  evidence release and independently reviewed scalar-revision of
  allowed likelihood strength. Same Qwen decision also evaluated
  before revision for isolating Verifier contribution.

Crucial evidence semantics: The test values are already recorded in
PPMI at original BL; this is a retrospective masking/acquisition
benchmark only. The LLM receives the list of available measured test
*names* but not their values, labels, IDs or later clinical evidence.
LLM input shows no patient scores. This selection is availability
aware and is NOT a prospective clinical ordering workflow.

ClinicalEvidenceChannel learns smoothed stratified likelihoods
P(measured BL optional test bucket | future six-item Motor severity)
from FIT only, 4 outcome burden strata. Same categorical WM prior
is updated with actual measured test value when selected. This is
an approximation to conditional Bayes update, not a learned true
causal or dynamical clinical mechanism. OOV selections are rejected
and fall back deterministically to WM utility maximizing choice.
A single query is allowed per available patient.

Verifier arm3 selects an alpha candidate on DEV from
{0,.25,.5,.75,1,1.25}, independently audits exactly one
candidate against incumbent alpha=1 and only then produces heldout
forecast using that chosen alpha. Report MAE for median, RMSE for
mean, RPS, Brier, NLL, and patient-level contrasts on same fold.

Result files are task-level incremental aggregates. If 12h walltime
kills a Qwen job, partial task metrics survive. Qwen inference on CPU
may require considerably more time than other arms. Never interpret
an empty or partial report as model failure. 8 reserved CPU, 48GB,
12h hard cap. No Slurm arrays; no GPU; no new branch. This is
verifier-gated scalar evidence update NOT recursive self-improvement.
Use previously consumed final TEST neither for training nor scoring.

## Submit twenty independent CPU jobs

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
  sbatch --export=ALL,PHASE80_ARM_ID="$arm" slurm/phase80_dusky_cpu.sbatch
done
```
