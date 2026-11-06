# Phase91 — pre-registered FIT-only conditional proper scoring (NOT YET RUN)

**Design date:** 2026-11-06. **Status: PLANNED_NOT_RUN.**
**Repository branch:** `feat/phase62-ppmi-cohort-expansion`.
**Unique mechanism:** `conditional-proper-score`. Preflight-check P90
`partial33-likelihood` first; this study alters optimization of the
exact same mixture-based hidden-six conditional posterior rather than
introducing a newly claimed independent model architecture.

## Motivation and prespecified hypothesis

Phase90 (complete jobs 342535..342538) trained a full33 latent joint
emission with real FIT complete visits and FIT partial observed
adjacent visits, but optimized the **future unconditional** hidden
burden plus full33 likelihood. Conditional future-visible27
inference was only performed at evaluation time.

In Prodromal12, P90 static full33 RPS = .022504544 before
releasing 27 same-visit visible ratings and .023857876 AFTER:
a deterioration of .001353333. Prodromal24 static went from
.023078527 to .025078095 (deterioration .001999567).
A theoretically coherent Bayesian conditional of the learned
joint distribution need NOT improve finite-sample conditional RPS
when that joint is miscalibrated.

**Falsifiable P91 hypothesis:** direct FIT-proper-scored training of
`p(hidden6 total | history, real observed subset of future visible27)`
will improve heldout same-visit conditional RPS versus P90,
without materially worsening the original strictly pre-cutoff
future RPS. We explicitly distinguish the retrospective recovery
task from prospective future forecasting.

## Exactly three new standalone CPU jobs — no duplicated P90 baselines

| Arm | New training policy | Purpose |
|---|---|---|
| 0 | STATIC full33 joint WM; true FIT future-visible27 observed in full during conditional loss | Is explicit conditional proper scoring enough? |
| 1 | STATIC full33; choose 7/14/21/27 actually measured visible items at random per FIT minibatch, with rest masked `-100` | Does training under variable real measurement availability help? |
| 2 | CONTINUOUS-TIME full33; same randomized 7/14/21/27 subset conditional loss | Does latent flow benefit when the conditional task is supervised? |

All arms retain Phase90 true FIT future prior NLL + RPS and full33
joint likelihood + true FIT auxiliary partial-visit marginal
likelihood. Adds a conditional posterior loss evaluated using
**actual FIT six-hidden-item labels** and only observed visible
values (unobserved ones are `-100`), fixed weight 0.5. No real
patient rating is invented. No policy/LLM queries; no GPU; no
fivefold outer evaluation; no separate Slurm smoke run.

**Training constants:** 18 epochs; loss weights:
unconditional total NLL=1; unconditional RPS=1.5;
joint full33 per-item NLL=1; auxiliary per-observed-item
NLL=0.5; conditional NLL=0.5; conditional RPS=0.75.
These are prespecified exploratory engineering choices,
not an exhaustive optimized hyperparameter search.

## Same data, same patient-global split, same privacy controls

Existing frozen original TRAIN manifest; original consumed final
TEST excluded. Same selected patient-global FIT80/DEV20 hash as
Phase84–90. Prodromal12 FIT470/DEV114; Prodromal24
FIT307/DEV71; PD12 FIT119/DEV37.
Prodromal12/24 share FIT model and true FIT-only adjacent
visit evidence; PD12 trains separately. Any real complete33
future target is attached ONLY as training labels (for FIT)
or at **same-visit evaluation AFTER the visit** (for DEV);
`sequence_features()` never accesses this field.

For each DEV patient evaluate:
1. **Future prior:** no future visit values, true future hidden6
  total 0..24, full probability distribution.
2. **Same-visit posterior:** reveal 7, 14, or all 27 TRUE
  target-visit visible ratings, chosen by deterministic hashed
  subset independent of hidden outcomes; still NEVER reveal hidden6.

Do not compare a conditional posterior score to Phase90
future-prior as if it were a forecasting improvement.
Use matching information budget and metrics.
Primary same-visit comparison at 27 visible: P91 Arm0/1
vs Phase90 static Arm2, and P91 Arm2 vs Phase90 dynamic Arm3.
The additional 7/14 observed subsets assess behavior within
P91 and are not compared against unreported P90 7/14 baselines.

Record median MAE, mean RMSE, RPS, NLL, Brier, 80% interval
coverage/width; also the difference of each posterior RPS
from its *own* model's future RPS as a descriptive metric,
not a significance test.

**Go/no-go for conditional mechanism:** if full27
conditional RPS fails to improve against the matching P90
same-visit baseline on Prodromal 12/24, OR if the genuine
future RPS worsens materially relative to the matching P90
prior, do not replace strong forecasting models. Report mixed
or negative results honestly. Given small repeatedly selected
DEV sets, do not assert statistical significance or clinical
benefit even if point metrics improve.

## Run tests and three jobs after pulling branch

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
"$PY" -m unittest discover -s tests -p test_phase91_conditional.py -v
for arm in 0 1 2; do
  sbatch --export=ALL,PHASE91_ARM_ID="$arm" slurm/phase91_dusky_cpu.sbatch
done
```

Each Slurm job account `def-hup-ab_cpu`, partition `base`,
8 CPUs reserved / 1 Torch compute thread, 48GB RAM, 12h max.
Outputs `/scratch/ylong66/pks_agent_outputs/phase91_conditional_*/`
with `dev_metrics.csv`, `coverage.csv`, `progress.json`,
`summary.json` only after successful completion.

**Logging policy:** Phase91 is already recorded as
`PLANNED_NOT_RUN` in [canonical machine registry](experiment_registry.json)
and [living ledger](EXPERIMENT_LEDGER_CURRENT.md). After uploaded
results, add job IDs, completion status, true numerical metrics
and failure modes; do not append fabricated metrics.
Source code [Phase91](../scripts/phase91_proper_conditional_wm.py),
[Slurm](../slurm/phase91_dusky_cpu.sbatch),
[tests](../tests/test_phase91_conditional.py).
