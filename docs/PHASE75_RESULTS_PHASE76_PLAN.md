# Phase75 observed results and Phase76 reliability-gated multimodal plan

Source: `phase75_matched.zip` uploaded October 29, 2026. Archive
contains 20 directories, but **only 8 complete reports**: arms
4-7 (outer fold 1) and 12-15 (outer fold 3). Other 12 are empty.
Jobs 4-7 and 12-15 reported completed, 3 tasks per job, ~5.7–37.6
seconds each. Do not assume empty jobs timed out: no Slurm logs
or exit codes are included. **This is 2/5-fold exploratory evidence**.

Within each fold method metrics are replicated across arms; deduplicate
by (cohort,horizon,fold,model). Two-fold patient weighted MAE:

| Model | Prodromal12 n219 | Prodromal24 n155 | PD12 n67 |
|---|---:|---:|---:|
| Ridge Motor | 1.02283 | 1.30032 | 3.01579 |
| HGB Motor | .94772 | 1.17577 | 2.53540 |
| CT Motor | approx .765 | .987 | approx 2.59 |
| Ridge Motor + BL nonmotor | 1.08869 | 1.37957 | 2.98704 |
| HGB Motor + BL nonmotor | .97020 | 1.14063 | 2.45730 |
| CT Motor + BL nonmotor | approx .804 | .995 | approx 2.59 |
| Verified CT multimodal | approx .807 | .899 | approx 2.59 |

Minor per-arm CT numerical differences (~.001-.007 MAE) remain
despite equal fold/hyperparameters; model reproducibility is imperfect
at the current CPU runtime. Reported CT values above aggregate
the arm3 within-job matched comparisons where available, with
approximations to avoid implying false precision.

Prespecified matched paired MAE CIs (candidate minus reference):
- Prodromal12 HGB Motor -> CT Motor: fold1 -0.1703
  [ -0.2817,-0.0672 ]; fold3 -0.1913 [ -0.2869,-0.1063 ].
- Prodromal12 CT Motor -> CT Multimodal: fold1 +0.04674
  [ +0.01282,+0.07997 ]; fold3 +0.03667
  [ +0.00998,+0.06214 ]. Adding nonmotor baseline features
  is **adverse** in both currently available outer folds.
- Prodromal24 CT Motor -> CT Multimodal: fold1 -0.01550,
  CI crosses zero; fold3 +0.03290, CI crosses zero.
- Prodromal24 CT Multimodal -> verified: fold1 0,
  fold3 -0.17823 [ -0.29379,-0.05995 ].
- PD12 HGB Motor -> CT Motor changes sign by fold, CIs cross zero.

Actual independent gate: 6 decisions (2 folds * 3 tasks),
1 accepted (Prodromal24 fold3). Two Prodromal12 proposed
MAE revisions passed DEV improvement but were **rejected on
AUDIT because Brier deteriorated**. The evidence is NOT proof
of neural-model self-improvement.

**Interpretation**: A good Motor-only CT predictor is emerging for
Prodromal, but baseline nonmotor modalities currently seem noisy,
sparse or overfit, especially for 12m. A new architecture should
learn to suppress unreliable modality input rather than blindly
concatenate it. Five folds and independent new patients are still
needed for any top-conference generalization claim.

## Phase76 implementation target

Five outer held-person folds x four distinct experimental strategies
= twenty independent, non-array CPU jobs:

0. motor-only CT WM, with within-job same-panel HGB references;
1. early-fusion CT with raw real BL clinical nonmotor features;
2. clinical feature reliability gating (learnable per-feature gate
   initialized near zero) and heavy clinical feature dropout;
3. allowlisted gated-MAE revised checkpoint against a frozen
   same-fold Motor-only incumbent, selected by DEV and accepted/rejected
   exactly once by independent AUDIT with a binary Brier guardrail.

In each job, train the same Motor reference and fit-only baseline
HGB; no Qwen overhead. Explicitly report model panel, optional-item
coverage and per-feature gate summary, with patient-paired bootstrap
against same-input HGB and Motor-only CT. Same original strict
Prodromal TRAIN and PD, actual BL clinical modalities and prior
visible27 Motor histories; no target leakage, no imaging.
Only after proving a stable WM benefit should Qwen Doctor and
genuine code/checkpoint RSI take large CPU budgets again.

**Development-only**: the 12 missing Phase75 arms may simply
be pending/failed; check Slurm logs/sacct, not model accuracy.
