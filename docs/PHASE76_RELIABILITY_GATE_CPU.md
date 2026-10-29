# Phase76 reliability-gated nonmotor clinical observation WM

Result basis: [PHASE75_RESULTS_PHASE76_PLAN.md](PHASE75_RESULTS_PHASE76_PLAN.md).
Phase75 available only fold1 and fold3 (8/20 full report folders).
Those two folds show a stronger motor-only CT than motor-only HGB
for Prodromal 12m, while blindly adding nonmotor clinical baseline
values worsened 12m CT prediction in both folds. These results
remain repeated development-data explorations, not a paper claim.

Implemented Phase76 tests five identical outer person folds x four
methods (20 **ordinary** CPU jobs, no array):

0. Motor-only CT direct WM with FIT-only adjacent Motor auxiliary.
1. Early-fusion multimodal CT using actual BL clinical scale values.
2. Gated multimodal CT with learnable sigmoid feature gate initialized
   near 0.047 (to prefer the Motor-only representation at start), and
   65% stochastic masking of actually measured clinical optional inputs.
3. Gated multimodal revised objective with an independent FIT-
   exclusion DEV candidate selection and one AUDIT accept/reject;
   incumbent = Motor-only CT, candidate = gated CT with L1 loss and
   different multi-task objective.

Every job computes matched Motor-only and multimodal Ridge/HGB and
its own matching Motor-only CT. Contrasts include CT vs same-patient
Motor-only CT and multimodal HGB; arm3 logs audit gate and rollback.
Note: this gate is a **checkpoint/loss revision proxy**, not a
self-modifying RSI agent or external validation.

The trained clinical gates do not infer value/cost of an entirely
unmeasured modality; they only suppress unreliably predictive
features already present at BL. The gating network is neither causal
nor guaranteed to improve outcomes. Counts of genuinely available
clinical features may be much smaller than feature cardinality.

Each task writes aggregate results immediately; when Slurm TIME_LIMIT
kills a job, earlier task-level metrics and progress.json remain.
Full completion indicated ONLY by summary.json and status=complete;
directory existence alone is insufficient. Max 12h/job. No GPU/Qwen.

## Submit 20 non-array CPU jobs
```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
  sbatch --export=ALL,PHASE76_ARM_ID="$arm" slurm/phase76_dusky_cpu.sbatch
done
```

8 CPU, 48GB, 12h/job. Do not submit if still investigating
Phase75 empty job directories; check squeue and sacct first.
After completion zip all `phase76_gated_*/` output folders
with method_metrics.csv, paired_bootstrap.csv, coverage.csv,
audit.json, progress.json and summary.json.
