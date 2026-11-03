# Phase86 complete: numerical time-flow constraint vs true visit-anchored supervision

User-uploaded `phase86_reports.zip` contains exactly three complete
jobs, 336773, 336774, 336775, each all 3 tasks. Patient-level
FIT/DEV fingerprints match across arms; original already-consumed
final TEST was never accessed. Same global deterministic 80/20
patient split. Patients: Prod12 FIT470 DEV114, Prod24 FIT307
DEV71, PD12 FIT119 DEV37.

| Task | Phase85 shared mixture RPS | Phase86 flow RPS | Flow+semigroup RPS |
|---|---:|---:|---:|
| Prodromal12 | **.023103038** | .023393650 | .023392855 |
| Prodromal24 | **.022843899** | .023143352 | .023142850 |
| PD12 | .067274773 | .064142725 | **.064139636** |

Relative flow-only vs Phase85: Prod12 RPS 1.26% worse,
Prod24 1.31% worse, PD12 4.66% better. Prod12 MAE:
.736842 vs .745614; mean RMSE 1.378695 vs 1.397309.
Prod24 MAE: .676056 vs .704225; mean RMSE 1.391295
vs 1.420524. PD12 flow mean RMSE 2.821917 vs 2.963295;
median MAE 2.216216 vs 2.189189, a tradeoff.

Phase86 arm2's numerical semigroup regularizer made almost
no difference: mean semigroup discrepancy 5e-5..1e-4,
all score differences tiny. This is not observation-based
transition learning and cannot establish biological time-flow.

## Phase87 candidate hypothesis: transition-supervised latent flow

Rather than yet another equation-only time consistency penalty,
use the *actual measured next-visit Motor scores* as supervised
state transitions. Existing phase72 `build_auxiliary`
provides real adjacent same-medication-state visits, input
through previous visit and 27 actual visible Motor item targets
from the next visit, with unmeasured labels ignored (-100).
Patient source is strictly FIT-only. Deduplicate auxiliary
pairs when sharing Prodromal 12m and 24m tasks.

Arm0: Refit Phase85 best shared correlated mixture with
the same seed and the same FIT/DEV split.
Arm1: Refit Phase86 flow without intermediate visit supervision.
Arm2: Phase86 flow architecture with a 27-item next Motor
prediction head and loss weight 0.05 on true adjacent visits.
Arm3: Same architecture, weight 0.15.

This is a targeted, four-candidate mechanism-development study,
not a wide hyperparameter search, fivefold or an ablation suite.
Use median-MAE, mean-RMSE, full RPS, NLL, Brier, 80% interval
coverage. Training supervision is observed clinical ratings,
not generated synthetic labels, and model forecasts target
future episodes only. Such a system remains a *predictive*
world model, NOT a validated causal disease simulator or
autonomous RSI. Development set is repeatedly used and
results require an independent confirmation cohort.

The near-zero semigroup penalty did not solve the missing
observed-trajectory constraint; this direct transition loss
tests a more falsifiable mechanism. If both weights fail to
exceed Phase85 across Prod12 and Prod24, retain the simpler
shared mixture and revisit clinical state observability
instead of inventing more flow losses.

## Launch only 4 CPU jobs

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python \
 -m unittest discover -s tests -p test_phase87_visit.py -v
for arm in 0 1 2 3; do
 sbatch --export=ALL,PHASE87_ARM_ID="$arm" \
   slurm/phase87_dusky_cpu.sbatch
done
```

One ordinary Slurm job per arm; account def-hup-ab_cpu,
partition base, 8 CPUs reserved, 48GB, 12h maximum,
torch CPU threads=1, MKLDNN disabled, no GPU and no arrays.
Output `/scratch/ylong66/pks_agent_outputs/phase87_visit_*/`.
