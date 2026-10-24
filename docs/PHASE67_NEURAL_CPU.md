# Phase67 formal CPU run: neural masked clinical WM, 20 non-array jobs

Results basis: [PHASE66_RESULTS_PHASE67_PLAN.md](PHASE66_RESULTS_PHASE67_PLAN.md).
Same existing Git branch, no new branch. Original frozen Prodromal
TRAIN and strict PD cohort only; previously consumed Phase44 TEST
not accessed.

**Method**: 96-dim masked MLP trained with outcome-class cross-entropy
from correctly labelled FIT patients plus masked Motor reconstruction
from actually observed FIT visits. Unlabelled FIT-only patients
contribute auxiliary Motor feature reconstruction, without invented
future burden labels. 0..Q randomly visible items per training sample.
Actual PatientTool always returns true recorded BL/pre-target R01
measurements and never network-generated values.

The model's auxiliary reconstruction head currently covers
MDS-UPDRS Part III Motor (0..4) and uses genuine scoring codes,
while clinical non-motor features (MDS-UPDRS I/II, MoCA, UPSIT, RBD,
SCOPA-AUT) enter the shared neural outcome encoder.
**Non-motor query-action values are currently zero**, because no
calibrated distribution head has yet been implemented for them.
Thus this is not a full multimodal generative World Model and may
favor Motor-only acquisition. Avoid claiming a state-transition
causal model. Extra training real Motor visits are used only from
FIT subjects, not eval, tune or audit subjects.

Twenty tasks = five outer patient hash folds (0..4) times four
policies (greedy, plain Qwen, Qwen+WM, Qwen+WM+verified proxy).
All share same patient folds for a task. Existing verifier still
chooses allowlisted WM planning style and temperature using
DEVELOPMENT/AUDIT greedy proxy, not actual Qwen rollouts. This is
not full recursive neural self-improvement.

The model and script have been pushed to GitHub; cluster deployment
and real-data execution must verify local compatibility.

## Submit all 20 ordinary CPU jobs; no array, no separate smoke jobs

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
  sbatch --export=ALL,PHASE67_ARM_ID="$arm" slurm/phase67_dusky_cpu.sbatch
done
```

8 cores / 48 GB RAM / 12 hours each, no GPU. Qwen4B
CPU inference may take several hours per task. All 20 jobs
are submitted independently and Slurm decides how many run
at once; 20 simultaneously may request up to 160 cores and
960 GB memory.

Aggregate outputs per arm:
`fold_metrics.csv`, `cohort_coverage.csv`,
`verifier_trace.json`, `summary.json`.
Never upload any identifiable PPMI files or per-patient predictions.
```bash
squeue -u "$USER"
cd /scratch/ylong66/pks_agent_outputs
zip -r phase67_reports.zip phase67_neural_*/
```

Evaluation remains TRAIN-internal exploratory, not new patient or
external-center validation. Paired gains against Phase66 require
matched fold assignments; Phase67 reuses the Phase66 fold hash
deliberately to maintain direct aggregate N per fold and task.
