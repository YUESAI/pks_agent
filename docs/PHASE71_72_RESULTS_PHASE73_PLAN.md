# Phase73: paired proof-of-component experiment after Phase71 and Phase72

Evidence:
- Phase71 upload: all 20/20 reports; five-fold n=584 Prodromal12,
  n=378 Prodromal24, n=156 PD12. MAE for variants Static / temporal
  auxiliary-only / fixed .35 temporal blend / gated blend:
  Prodromal12 1.032535 / 1.027037 / 1.070316 / 1.027850;
  Prodromal24 1.159253 / 1.149556 / 1.166563 / 1.147463;
  PD12 2.647425 / 2.681049 / 2.696497 / 2.683166.
  All 15 DEV/AUDIT temporal blends selected zero. This decisively
  fails to demonstrate direct-fusion benefit; the auxiliary training
  improvements were tiny and variable.
- Phase72 uploaded ZIP contains ONLY five arm3 reports (3,7,11,15,19),
  NOT the 15 baseline arms. Five-fold arm3 MAE / Brier:
  Prodromal12 1.016393 / .225588; Prodromal24 1.093535 / .232651;
  PD12 2.677010 / .049212. In the 15 real-Qwen Qwen-checkpoint
  audits 9 accepted ensemble, six rejected. Cannot claim 72
  beats Ridge, single or ensemble until matched arm data exists.

Phase73 fixes the comparison by training ALL baseline models within
EVERY job, on the exact same people and same true observation features.
It also exposes multiple pre-target visits: at 12 months, only
observed Motor visits through +8 months; at 24 months, through
+16 months. The target is independently selected at +12±3 or +24±4
months, never used as an input. All models see fixed BL optional-test
policy for the three non-Qwen arms; only arm3 uses real Qwen to select
one recorded BL optional clinical test from FIT-admissible names.
Qwen evidence is matched among ensemble-versus-single candidate
checkpoint comparisons. The explicit baseline sample-size/visit-count
coverage helps identify how much true longitudinal history the data
actually supports.

Twenty standalone CPU jobs = five original patient folds x 4 arms:
0 Ridge primary, 1 single CT latent WM primary,
2 CT ensemble primary, 3 real Qwen with DEV/AUDIT-gated checkpoint
selection primary. Within each job, reference ridge, single and
ensemble on fixed evidence are all scored, and subject-level
paired MAE bootstrap differences exported as aggregates.
In arm3, additionally evaluate single, ensemble and verified
checkpoint on SAME Qwen-chosen evidence to isolate RSI revision.
This remains restricted verifier-gated checkpoint selection,
not autonomous recursive code modification or clinical utility.

Known limitations: main outcome is six-site future Motor burden;
new PPMI modalities beyond locally available baseline structured
clinical scales are not imported. The selected BL optional test
is retrospective acquisition from recorded data, not a newly
performed clinical action. Qwen's evidence-selection policy has no
same-fixed-evidence Qwen control; thus its effect cannot be attributed
solely to reasoning. Still original frozen TRAIN, not independent
external or untouched final test. All results exploratory.

## Submit 20 ordinary jobs (never Slurm array)

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
  sbatch --export=ALL,PHASE73_ARM_ID="$arm" slurm/phase73_dusky_cpu.sbatch
done
```

Each requests 8 CPU, 48GB and 12 hours. Only 5 arm3 jobs load Qwen.
Outputs: method_metrics.csv, paired_bootstrap.csv, coverage.csv,
audit.json and summary.json. Submit after git pull and upload aggregate
ZIP with those files. No patient identifiers/records should leave HPC.
Code not yet validated on Dusky source data.
