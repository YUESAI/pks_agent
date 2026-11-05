# Phase89 outcome and Phase90 partially observable 33-item WM

Source: uploaded `phase89_reports.zip`, four successful jobs
342492..342495, all three tasks complete. Same FIT/DEV
patient-global fingerprints across all four arms.

| Method | Prodromal12 DEV=114 RPS | Prodromal24 DEV=71 RPS | PD12 DEV=37 RPS |
|---|---:|---:|---:|
| Independent static Phase85 | .023103038 | .022843899 | .067274773 |
| Independent dynamic Phase87 | **.022642465** | .023180876 | **.063139069** |
| Inner 2fold OOF horizon gate | .022934358 | .022784346 | .063306590 |
| Inner 2fold OOF patient-aware gate | .022932176 | **.022783286** | .063255981 |

The *independent* gate increases 24m RPS accuracy by ~0.27%
relative to static but is worse than pure dynamic on 12m and PD.
OOF gate training proper RPS decreases only .02550479 to
.02544696 on 777 mixed 12m+24m FIT rows; application gate
~0.19 on both 12m and 24m despite different best experts.
PD gates ~0.91-0.95 dynamic, similarly near-constant per subject.
Twofold OOF experts train on half FIT, whereas full refitted
experts train on all FIT, likely contributing a shift. This
is a limitation, not evidence that dynamics is globally best.

The repeated same-split DEV is no longer a genuine untouched
holdout. Do not use these comparisons to claim statistical
significance, prospective performance, or learned disease
subtype discovery.

## Phase90 research question: partially observed joint clinical state

Previous models forecast six future *hands-on-only* ratings
(HIDDEN), using histories of 27 VISIBLE Motor items. Their
mixture outputs modeled the six, while 27-item transitions
were trained in a separate auxiliary prediction head.
This design has no coherent same-visit observation likelihood
for both hidden6 and visible27.

Phase90 uses one 4-component latent joint emission over all
33 real MDS-UPDRS III item categories:
  P(S_1,...,S_33 | H,t)
   = sum_k pi_k(H,t) prod_j p(S_j | H,t,k).

It provides (a) an unconditional posterior of the hidden six
total score 0..24 from pre-cutoff history and (b) an exact
conditional hidden-six distribution after the same future
visit's visible27 ratings have been genuinely observed.
The second is explicitly a retrospective **same-visit masked
recovery task**, NEVER a prospective future forecasting
performance metric.

Supervision:
- Eligible verified FIT future visits have *all 33 measured*
  and train full33 joint log likelihood plus hidden-six RPS.
- Auxiliary FIT adjacent visits contain measured future visible27;
  they train a marginal log likelihood over **only measured items**,
  integrating missing other categories and six hidden scores.
- No fabricated zeros; missing labels retain mask -100.
- Missing auxiliary trajectories from DEV patients cannot enter
  training. Site masks never become target information in
  pre-cutoff model input.
- All site probabilities share the same 4-component mixture,
  unlike Phase87's independently supervised 27-item head.

Four same-protocol standalone CPU jobs, single global
patient-disjoint 80% FIT / 20% previously explored DEV:
0 Phase85 strong shared static mixture.
1 Phase87 real visit-supervised dynamic mixture.
2 NEW full33 partial-observation static mixture.
3 NEW full33 partial-observation continuous-time mixture.

Main outcome comparison MUST use unconditional hidden-six
future forecast RPS, median MAE, mean RMSE, NLL, Brier.
Masked recovery posterior metrics appear in separately
prefixed fields and cannot be equated with forecasting.
A large gain in masked recovery with no forecast gain is
still a potentially interesting *different* research task,
not evidence of longer-horizon forecasting improvement.

## Run

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python \
  -m unittest discover -s tests -p test_phase90_joint33.py -v
for arm in 0 1 2 3; do
 sbatch --export=ALL,PHASE90_ARM_ID="$arm" slurm/phase90_dusky_cpu.sbatch
done
```

Each job requests 8 CPUs and 48GB but uses conservative
single-thread PyTorch with MKLDNN disabled; hard wall limit
12h, Slurm partition base/account def-hup-ab_cpu.
No fivefold, GPU, Qwen, or Slurm arrays. Outputs under
`/scratch/ylong66/pks_agent_outputs/phase90_joint33_*/`.
Code not yet run on Dusky.
