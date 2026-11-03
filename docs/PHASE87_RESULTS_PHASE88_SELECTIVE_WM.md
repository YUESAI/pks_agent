# Phase87 final outcomes and Phase88 selective latent progression WM

Source: user-uploaded `phase87_reports.zip` (2026-11-03).
Exactly 4 completed jobs 337345–337348; each 3 tasks and a full
summary.json. Patient-global 80/20 split hashes match across arms.

| Arm | Candidate | Prodromal12 RPS N114 | Prodromal24 RPS N71 | PD12 RPS N37 |
|---|---|---:|---:|---:|
| 0 | Phase85 shared static mixture | .023103038 | **.022843899** | .067274773 |
| 1 | Phase86 continuous-time latent flow | .023393650 | .023143352 | .064142725 |
| 2 | Phase87 real next-visit supervision weight .05 | **.022641547** | .023190976 | .064713132 |
| 3 | Phase87 real next-visit supervision weight .15 | .022642466 | .023180877 | **.063139055** |

Mean-RMSE: Prod12 baseline1.378695, arm2 1.345129.
Prod24 baseline1.391295, arm2 1.409470.
PD baseline2.963295, arm3 2.794671.
Median MAE: Prod12 baseline.736842 -> arm2/3 .666667;
Prod24 baseline.676056 -> arm2 .704225;
PD baseline2.189189 -> arm3 2.135135.
Additional Brier, NLL, interval coverage and width are in archived
source results.

Prod12 improvement on RPS ~2.0%, PD12 arm3 ~6.1%, while
Prod24 becomes ~1.5% worse. Real FIT adjacent-visit supervision
helps some horizons, but does not uniformly improve the WM.
At only N37 PD DEV and N71 Prod24 DEV, no inferential or final
confirmation claim is justified.

Important: the original site mixture decoder uses a conditional
independence assumption within each component. Components are
latent statistical factors, not validated clinical disease
subtypes or causal states.

## Phase88 falsifiable hypothesis

Train one *selective* WM that learns to preserve static observed
state or apply learned time flow per sample. This is explicitly
different from manually choosing different models for 12m and 24m
based on DEV results.

The model is a coherent convex mixture of the two distributions:
P(Y|H,t)=(1-g(H,t))P_static(Y|H,t)
         +g(H,t)P_flow(Y|H,t)
with exact 0..24 normalization and six-site joint likelihood in
each branch. Dynamic branch also predicts actual FIT-only
27-item next visits using the Phase87 real-supervision weight .15.
The initial gate prior is 0.2, learned end-to-end. A small
companion NLL is placed on both branches to prevent an
immediately collapsed gate masking one branch's learning.
An exploratory gate regularizer discourages needless dynamic
updates.

Exactly 4 ordinary CPU jobs, same deterministic patient-global
80% FIT / repeatedly examined 20% DEV:
0. Phase85 best correlated static shared-horizon WM;
1. Phase87 observed-visit flow weight .15;
2. New time-only gate g(t);
3. New patient-state and time-dependent gate g(h,t).

All candidates use same 18 epochs default. PD uses separate
FIT patient model; shared Prodromal12/24 FIT remains unchanged
and auxiliary real adjacent visits are deduplicated.

Records report median-MAE, mean-RMSE, RPS, NLL, Brier,
80% interval coverage/width, and for gated arms both
standalone branch RPS and gate mean/std. Under a robust
clinical benchmark, no manual DEV-selected horizon policy
should be marketed as generalization. Pre-register any later
confirmatory cohort once the algorithm is chosen.

Commands:
```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python \
 -m unittest discover -s tests -p test_phase88_wm.py -v
for arm in 0 1 2 3; do
  sbatch --export=ALL,PHASE88_ARM_ID="$arm" slurm/phase88_dusky_cpu.sbatch
done
```
8 CPU reserved, 48GB RAM, 12h max walltime, no array,
no GPU, PyTorch CPU threads=1, MKLDNN disabled.
Result dirs: /scratch/ylong66/pks_agent_outputs/phase88_selective_*/

Code has not yet been run on Dusky as of its commit.
