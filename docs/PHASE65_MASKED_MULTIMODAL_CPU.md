# Phase65: twenty independent CPU jobs, no Slurm array

Branch: feat/phase62-ppmi-cohort-expansion; no extra branches.

Documented source result and motivation:
[PHASE64_ACTUAL_RESULTS_PHASE65_PLAN.md](PHASE64_ACTUAL_RESULTS_PHASE65_PLAN.md).

Phase65 is an experimental improvement over Phase64's unmasked-Ridge
full-input vs few-question inference mismatch. World Model now trains
its future-six burden regressor under 0..Q randomly sampled masks, each
patient repeated 12 times in training. A separate FIT-only empirical
observation-likelihood table preserves original score units for MoCA,
UPSIT and other questionnaire fields. No imagined value can enter the
PatientTool. These are statistical prototypes, NOT multi-step neural
disease simulators or a learned causal patient-state transition.

Five data panels retain the Phase64 P0..P4 definitions, four policies:
0 masked-WM greedy, 1 plain real Qwen, 2 real Qwen+WM guidance,
3 real Qwen+WM+one-shot verifier-gated policy/scalar revision.
The last mode is explicitly restricted RSI-lite (not LLM evolution
nor a jointly trained neural WM). The four policy arms per panel
share identical masked WM seeds and patient partitions.

Baseline actual observed clinical modalities: MDS-UPDRS I/II/III,
MoCA, UPSIT, RBD and SCOPA-AUT, if correctly identified in the
downloaded release. R01 Motor only where previously observed.
Never includes image, raw MRI, DaTSCAN, CSF or genetic data.
Positive/negative labels are future real measured six-site Motor
outcomes; qualifying strict source records are never provided to
the Doctor. Prodromal patients from original frozen TRAIN only.
PD task is separate; no reused original Phase44 TEST.

Individual CPU job requests 8 cores, 48 GB, 12h walltime. Real Qwen
CPU inference can exceed the queue time cap, especially when many
subjects and modalities are eligible. Do not claim it is fast merely
because the clinical patient count is small.

## Submit all 20 separate sbatch jobs in one shell loop

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
  sbatch --export=ALL,PHASE65_ARM_ID="$arm" \
    slurm/phase65_dusky_cpu.sbatch
done
```

Do not submit an array. No standalone smoke job. Jobs independently
produce arm_metrics.csv, paired_greedy_bootstrap.csv,
cohort_sizes.csv, verifier_trace.json, summary.json under
/scratch/ylong66/pks_agent_outputs/phase65_masked_<ARM>_<JOBID>.

When results have finished, ZIP the COMPLETE directories including
summary.json and verifier_trace.json. Aggregate only, never upload
private raw PPMI rows/SQLite manifest or per-patient predictions.

No fresh confirmatory cohort is included. Paired bootstrap conditions
on fitted models and repeated development data. Stronger WM model
families, longitudinal self-supervision, matched-compute optimization
and independent validation still needed before a top-conference
algorithmic or clinical effectiveness claim.
