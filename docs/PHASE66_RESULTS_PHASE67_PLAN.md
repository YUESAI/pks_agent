# Phase66 actual results (uploaded report, October 24, 2026)

Source: `phase66_reports.zip` uploaded by project owner. Report includes
**all 20 arms**, 60 task-level fold metrics, 60 coverage records, and
20 `summary.json` plus 20 `verifier_trace.json`.
Slurm exit codes were not supplied; completed report files alone do
not establish job exit state.

## Five-fold weighted means (each patient contributes once per policy/task)

| Task | N | Policy | MAE | Brier | Invalid / 4*N |
|---|---:|---|---:|---:|---:|
| Prodromal 12m | 584 | greedy | .96772 | .21688 | 0 / 2336 |
| Prodromal 12m | 584 | Qwen unhinted policy | .98889 | .21936 | 407 / 2336 |
| Prodromal 12m | 584 | Qwen + WM hints | .95244 | .21259 | 13 / 2336 |
| Prodromal 12m | 584 | Qwen + verified proxy config | .97749 | .21966 | 6 / 2336 |
| Prodromal 24m | 378 | greedy | 1.07983 | .21709 | 0 / 1512 |
| Prodromal 24m | 378 | Qwen unhinted policy | 1.14951 | .22534 | 251 / 1512 |
| Prodromal 24m | 378 | Qwen + WM hints | 1.08271 | .21764 | 9 / 1512 |
| Prodromal 24m | 378 | Qwen + verified proxy config | 1.08982 | .22176 | 31 / 1512 |
| PD 12m | 156 | greedy | 2.44024 | .04927 | 0 / 624 |
| PD 12m | 156 | Qwen unhinted policy | 2.56217 | .05405 | 145 / 624 |
| PD 12m | 156 | Qwen + WM hints | 2.43988 | .05463 | 9 / 624 |
| PD 12m | 156 | Qwen + verified proxy config | 2.57589 | .05364 | 24 / 624 |

Weighted MAE/Brier are reconstructible exactly from aggregate fold
metrics. DO NOT calculate pooled AUROC by averaging fold AUROCs;
per-person predictions never left the cluster. PD high positive
prevalence renders fold-wise AUC undefined in some folds.

RSI trace: 15 real proposals, **9 accepted, 6 rejected** by the
greedy-proxy tuning plus once-only audit gate; these are not updates
validated on real Qwen decisions. Improvement did not transfer
reliably to Qwen heldout performance.

## Decision

The 12m Qwen+WM MAE gain relative to greedy is small (~1.6%).
24m Qwen+WM does not improve over greedy. Invalid Qwen actions are
greatly reduced by WM action hints versus unhinted prompting.
Existing probabilistic WM is class-conditional naive Bayes on as
many as 107 correlated features; it does not model patient-specific
multi-visit disease-state dynamics, and its implicit independence
assumptions may cause poor calibration.

Phase67 must target:
1. Multi-visit actual Motor score self-supervision using FIT persons
   only, with a real masked neural observation model.
2. Predict future measured burden in a supervised outcome head,
   trained only on FIT labelled patients with stochastic q-sized masks.
3. Neural item predictive distributions used only to **simulate**
   hypothetical candidate answers. Actual PatientTool always returns
   recorded BL/pre-target R01 measurements.
4. 5-fold x 4 strategies, with consistent feature pools/patient folds.
5. Real Qwen proposer and real Qwen verifier (allowlisted revision,
   one-shot independent audit) in RSI arm, unlike Phase66 greedy proxy.
6. Export aggregates and matched in-job baseline diagnostics; original
   consumed Phase44 TEST remains untouched.

Even this is not a causal multi-step world model or unconstrained
self-modifying RSI, and repeated internal TRAIN analysis is
exploratory rather than an independent confirmatory test.
