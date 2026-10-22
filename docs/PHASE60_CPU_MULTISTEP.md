# Phase60: CPU-only full experiments

This branch adds a coherent class-conditional categorical latent-mixture
World Model, greedy and adaptive two-step imagined question planning.
It is a same-V04 *recorded 27-item masking simulation*, NOT a real
future clinical intervention or a Qwen/RSI integrated experiment.

Four arms: naive_greedy, naive_rollout, mixture_greedy,
mixture_rollout. Mixture component count K in {1,2,3} is selected only
on DEV masked-item negative log-likelihood. Same 65/15/20 hash
FIT/DEV/EVAL split as Phase59, all frozen TRAIN internal evaluation
people (no Phase59 cap). Queries at budgets 4,8,12. Paired bootstrap
Brier comparisons within the single all-arms job. Aggregate-only
reports with no patient IDs, prompt transcripts or per-patient output.

## Full CPU job without smoke tests

```bash
cd /home/ylong66/workspace/pks_agent
git fetch origin feat/phase60-cpu-rollouts
git switch --track origin/feat/phase60-cpu-rollouts
sbatch --export=ALL,PHASE60_ARM=all slurm/phase60_dusky_cpu.sbatch
```

Optional 4 independent jobs (do not claim paired CI from split jobs):

```bash
for arm in naive_greedy naive_rollout mixture_greedy mixture_rollout; do
  sbatch --export=ALL,PHASE60_ARM="$arm" slurm/phase60_dusky_cpu.sbatch
done
```

Check job with `squeue -u "$USER"` and `sacct -j JOBID
--format=JobID,State,Elapsed,AllocCPUS,MaxRSS`.

Output at `/scratch/ylong66/pks_agent_outputs/phase60_all_JOBID/`:
arm_metrics.csv, paired_brier_bootstrap.csv, summary.json.

The original Phase44 TEST is consumed, and these TRAIN evaluation
people have been repeatedly explored: this does not independently
confirm generalization. Model-imagined scores do not become measured
Patient tool responses, ever. Phase60 invokes no Qwen, no GPU.
