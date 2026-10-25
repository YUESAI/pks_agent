# Phase68: Phase67 results and 20 CPU multimodal neural WM experiments

Source `phase67_reports.zip`: all 20 arm reports, 60 fold metrics.
Fold-patient-weighted MAE, Brier, exploratory:
- Prodromal12 n584: greedy 1.0034/.21492; plain Qwen .99408/.22087;
  Qwen neural WM .98928/.21185; verified .94427/.21026.
- Prodromal24 n378: greedy 1.17663/.21845; plain 1.19426/.23108;
  Qwen neural WM 1.19748/.22042; verified 1.12697/.22287.
- PD12 n156: greedy 2.72661/.05042; plain 2.60983/.04937;
  Qwen neural WM 2.72230/.05035; verified 2.71113/.05059.
The verifier gate accepted 4 of 15 proposals (11 rejected).
Plain Qwen invalid actions: Prodromal12 430/2336; Prodromal24
253/1512; WM-guided 14/2336 and 33/1512.
Many PD folds contain one class; AUROC undefined. No pooled AUROC
is computed from fold-aggregated output.

Phase68 improvement: all available baseline clinical panels and real
pre-target R01 are encoded in a neural masked-outcome WM and an
auxiliary masked categorical observation head. Non-motor
MoCA/UPSIT/PartI/PartII/RBD/SCOPA observations now have FIT-only
quantile bins and actual original-unit representative values.
The model's hypothetical clinical score branches are NEVER passed
as actual patient-tool values; only measured observations are.
Unlabelled FIT subjects can contribute measured-score reconstruction
without fabricated future outcome labels.

Five outer folds x four decision policies: WM greedy; plain Qwen;
Qwen with WM; Qwen with WM and verifier-gated style/temperature.
The verifier still optimizes a GREEDY proxy, NOT a Qwen policy on
DEV and is not neural self-modification. This remains exploratory
on repeatedly used original Prodromal TRAIN, not independent test.
This is not causal disease-state modeling; raw imaging and multiomics
are not part of these experiments.

## Submit 20 separate jobs, no array

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
  sbatch --export=ALL,PHASE68_ARM_ID="$arm" slurm/phase68_dusky_cpu.sbatch
done
```

Each standalone job: 8 cores/48GB/12h, no GPU. Real Qwen CPU
generation can still time out. Logs `slurm-phase68-<JOBID>.log`.
Upload aggregate output folders `phase68_multimodal_*` with metrics,
cohort_coverage, verifier_trace and summary. Do not upload PHI.
