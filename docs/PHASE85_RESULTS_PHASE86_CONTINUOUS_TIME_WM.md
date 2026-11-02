# Phase85 results and Phase86 continuous-time progression development

ZIP phase85_reports.zip contained 8 full completed runs, two
replications of each arm 0..3, all identical down to entire
dev_metrics.csv contents across repetitions. Patient train/dev
fingerprints matched for all arms and repetitions; previously
observed hash-order nondeterminism is now controlled for these jobs.

Same global 80/20 development split:
Prodromal12 470 FIT/114 DEV; Prodromal24 307/71;
PD12 119/37. Results RPS (lower better):
| Arm | Model | Prod12 | Prod24 | PD12 |
|---|---|---:|---:|---:|
| 0 | direct 25-cat | .023481 | .024387 | .069167 |
| 1 | independent six-site | .023545 | .023108 | .072093 |
| 2 | 4-component correlated site mixture | .023256 | .023635 | .067275 |
| 3 | shared-horizon correlated mixture | **.023103** | **.022844** | **.067275** |

Shared Prodromal24 median-MAE .676056 vs direct categorical
.746479; mean-RMSE 1.391304 vs 1.479135; NLL 1.105538
vs 1.154199. Prodromal12 shared RMSE 1.378694, RPS
.023103. PD12 remains separately fitted. Four mixture
components have effective entropy-based usage of ~1.6 for
Prodromal, not evidence of four distinguishable clinical subtypes.
The observed 80% nominal interval coverage is above 90% on
Prodromal, suggesting broad conservative intervals that should
be assessed separately from high point-forecast accuracy.
DEV is repeatedly examined, NOT independent heldout.

## Phase86 prespecified mechanism

Test whether explicit latent temporal progression helps beyond
the Phase85 strong shared-horizon correlated-mixture baseline.
Model reads measured irregular Motor visits through an
exponential-decay GRU state. Between last observed visit and
true forecast horizon, integrates neural drift in that state:
  dh/dt = softplus(a) * tanh(MLP(LayerNorm(h))).
Four deterministic Euler integration steps cover each sample's
actual remaining gap. The resulting future state decodes
4-component correlated six-site probability mixture and an exact
0..24 future burden distribution.

Three ordinary CPU jobs on exactly same 80/20 patient-global
split, per-task FIT/DEV hashes retained:
0. Strong Phase85 shared correlated-mixture incumbent.
1. Same kind of decoder but with continuous-time latent flow.
2. Same flow with numerical semigroup consistency penalty,
   comparing direct integration and two half-intervals.
   Loss remains total NLL+.2 joint six-site NLL+1.5 RPS.

This is an explicit neural state-evolution hypothesis, NOT a
verified biological disease mechanism. Flow semigroup consistency
can become a weak regularizer under small Euler step size, and
does not guarantee longitudinal trajectory truth. If it fails to
improve RPS or clinically relevant secondary metrics, revert
to Phase85 incumbent. Do NOT run an expansive fivefold or
hyperparameter sweep yet.

Run tests from login node before launching three real jobs:
```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python -m unittest discover -s tests -p test_phase86_flow.py -v
for arm in 0 1 2; do
 sbatch --export=ALL,PHASE86_ARM_ID="$arm" slurm/phase86_dusky_cpu.sbatch
done
```
3 standalone CPU jobs, not arrays, 8 CPUs allocated, 48GB,
single-thread Torch and disabled MKLDNN, max 12h each. Results
`/scratch/ylong66/pks_agent_outputs/phase86_flow_*/`.
