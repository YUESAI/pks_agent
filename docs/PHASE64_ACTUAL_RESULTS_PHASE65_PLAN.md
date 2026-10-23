# Phase64 observed results and Phase65 decision ledger

Source: user-uploaded `phase64_multimodal.zip` (October 23, 2026).
Contains 20 arm folders, exactly 60 rows of `metrics.csv` and 60
cohort-size entries. **Archive did not include `summary.json` or
`verifier_trace.json`**. It cannot establish RSI acceptance history,
generated-token counts or final Slurm exit statuses.

## Shared TRAIN-internal engineering evaluations

| cohort/task | eval n | outcome positives | limitations |
|--|--:|--:|--|
| Prodromal, 12 months | 126 | 45 | repeated frozen TRAIN only |
| Prodromal, 24 months | 80 | 24 | small positive count |
| PD, 12 months | 35 | 35 | AUROC undefined: all target-positive |
| PD, 24 months | absent | absent | not evaluated in this archive |

Exactly four actual requests were budgeted per patient. Sample sizes
refer to held-out within-development folds, **not the total PPMI cohort**.

### Representative Phase64 findings

| arm/task | MAE lower better | AUROC |
|--|--:|--:|
| P0 Greedy, Prodromal12 | 0.678953 | 0.613580 |
| P0 Plain Qwen, Prodromal12 | 0.622431 | 0.661179 |
| P0 Qwen+WM, Prodromal12 | 0.651582 | 0.615363 |
| P0 Qwen+WM+verified calibration, Prodromal12 | 0.665469 | 0.615363 |
| P2 Greedy, Prodromal24 | 0.614143 | 0.728794 |
| P2 Qwen+WM, Prodromal24 | 0.698929 | 0.563988 |
| P3 Qwen+WM, Prodromal12 | 2.251743 | 0.509876 |
| P3 verified, Prodromal12 | 1.706687 | 0.488065 |
| P4 Plain Qwen, Prodromal12 | 3.486644 | 0.412208 |

Plain Qwen P0 invalid actions: 135 requests on Prodromal12 (504 total);
the WM-guided Qwen P0 reported 0 on that task. Invalid outputs were
replaced by the deterministic fallback in the prior adapter. This is
an important policy-dependent confound, not independent proof of safety.

## Concrete implementation defects and implications

1. Previous Ridge WM trained on patients' already-known full measured
   panel but got sparse queried inputs at inference. It did not learn
   conditional predictions under the same 4-query masking process.
2. Previous hypothetical score enumeration used 0,1,2,3,4 as raw
   candidate values even for 0..30 MoCA and 0..40 UPSIT.
   Bucketed values were not mapped back to valid clinical units.
3. Invalid Qwen actions silently fell back to model-selected measures;
   under such a fallback, the plain arm is not a fully model-free arm.
4. Prior 'RSI' was deterministic scalar calibration selected using a
   greedy development proxy. It was not neural-weight/Doctor coevolution.
5. Only the 20 folders' CSVs were supplied; don't infer verifier
   accept/reject counts or complete individual-job walltime from it.

## Phase65 prespecified follow-up

- Load real BL clinical modalities and pre-target recorded R01 only.
- Use all qualifying frozen TRAIN Prodromal and strict PD/HC
  **fit-fold baseline visits** for masked-item representation learning,
  including patients with no scored future labels.
- Do supervised future-burden training only where actual later verified
  six-item scores exist. Hold out whole people globally across tasks.
- Train a lightweight **masked MLP with categorical measurement heads**
  and future-outcome heads using stochastic masks at training time.
  Counterfactual values are empirical TRAIN-observed original-score
  bucket representatives, never released by the patient tool.
- Four arms per input panel: deterministic WM planning, real Qwen plain,
  Qwen+WM and Qwen+WM+proposed/verified allowlisted WM-policy revision.
- Separately track invalid Qwen calls and fallbacks; plain fallback
  will use a deterministic label-free hash ordering rather than WM.
- Tune on TRAIN-internal DEV, one-shot independent AUDIT gate, then
  final same-split evaluation. Paired bootstrap inside each job
  against its deterministic counterpart, no patient data exported.

This is exploratory and may fail to show improvement. No clean
independent test remains from the original Phase44 exploratory setup.
A converged system still needs independent verification before clinical
or top-conference generalization claims.
