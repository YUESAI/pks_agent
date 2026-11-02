# Phase84 results and Phase85 correlated clinical WM experiment

User ZIP `phase84_reports.zip` confirms all three 80/20 jobs
336600,336601,336602 finished each 3 tasks. Development split is
patient-global across 12m and 24m. Counts: Prod12 FIT470/DEV114,
Prod24 FIT307/DEV71, PD12 FIT119/DEV37.

| Task | Model | median MAE | mean RMSE | RPS | NLL |
|---|---|---:|---:|---:|---:|
| Prod12 | categorical | .719298 | 1.383840 | .023481 | 1.130712 |
| Prod12 | six-site independent | .736842 | 1.395340 | .023545 | 1.137121 |
| Prod12 | six-site + selfsup | .719298 | 1.380470 | .023304 | 1.127502 |
| Prod24 | categorical | .746479 | 1.479135 | .024387 | 1.154199 |
| Prod24 | six-site independent | **.704225** | **1.382216** | **.023108** | **1.139160** |
| Prod24 | six-site + selfsup | .985915 | 1.656782 | .029834 | 1.285325 |
| PD12 | categorical | 2.378378 | **2.910202** | .069167 | 2.635343 |
| PD12 | six-site independent | 2.378378 | 3.133729 | .072093 | 2.524700 |
| PD12 | six-site + selfsup | **2.351351** | 2.992071 | **.068696** | **2.456928** |

24m six-site structure gives a development RPS improvement of
5.24% compared with direct categorical prediction, but 12m
and PD results are mixed. At 24m self-supervision makes RPS
22.3% worse; do not add it blindly.

## Phase85 structural research questions

The six-site independent model assumes
p(S1,...,S6|H)=product_i p(Si|H), losing cross-site correlation.
Phase85 uses a low-rank discrete latent **mixture** with K=4:
p(S1,...,S6|H)=sum_k pi_k(H) product_i p(Si|H,k).
Compute exactly normalized distribution of total score 0..24
by differentiable convolution **inside each mixture component**,
then mix. This is a statistical disease-state mode mixture,
NOT identified clinical subtypes or causal physiology.
Loss = total negative log-likelihood +
.2 actual six-site joint negative log-likelihood +
1.5 normalized ranked probability score. All observed labels
come from real FIT-only future exams.

4 models, same fixed global patient 80/20 split:
0 Phase84 original categorical baseline, rerun to confirm exact
  deterministic baseline after stable ordering fix.
1 Phase84 independent site-convolution, rerun.
2 Correlated four-component mixture per horizon.
3 Shared Prodromal 12m/24m four-component mixture FIT, evaluate
  on each horizon's own DEV. PD12 gets separate mixture.
  Patient repeated across horizons can contribute two FIT rows;
  all DEV patient IDs are excluded across both horizons.

Metrics: median MAE, mean RMSE, RPS, NLL, Brier and 80% interval
coverage, plus effective mixture component count and anonymous
hashes of each FIT and DEV patient-id set. No patient identifiers
are exported. No fivefold, no Qwen, no GPU.

## Commands

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python -m unittest discover -s tests -p test_phase85_wm.py -v
for arm in 0 1 2 3; do
  sbatch --export=ALL,PHASE85_ARM_ID="$arm" slurm/phase85_dusky_cpu.sbatch
done
```

Each ordinary job 8 CPU allocated / PyTorch threads=1,
48 GB, 12h walltime, no array, no smoke job. These remain
previously-explored development patients, not an independent
scientific test. Any eventual high-impact paper needs external
or genuinely unexamined patient confirmation.
