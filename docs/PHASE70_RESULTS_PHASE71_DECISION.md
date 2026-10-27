# Phase70 actual uploaded results / Phase71 prespecified WM mechanism gate

User report: phase70_horizon.zip uploaded October 27, 2026.
Contains exactly 20 complete arm folders, 60 fold_metrics.csv task records,
60 coverage rows, 20 verifier_trace.json, and 20 summary.json. 20
unique arm IDs with no duplicated reports. The ZIP does not include
Slurm exit statuses or per-patient predictions.

## Phase70 measured results

These are **patient-count-weighted means of fold metrics**, NOT pooled
individual-level AUROC. Each strategy evaluates Prodromal12 n=584 (198
positive), Prodromal24 n=378 (133 positive), and PD12 n=156 (148 positive).

| Task | Plain Qwen MAE/Brier | static Qwen+WM | temporal Qwen+WM | temporal+Qwen verified |
|---|---|---|---|---|
| Prodromal12 | 1.041609 / .222778 | 1.051050 / .221246 | 1.084144 / .224701 | 1.045716 / .217560 |
| Prodromal24 | 1.133464 / .228254 | 1.199853 / .223850 | 1.304294 / .241130 | 1.299471 / .240432 |
| PD12 | 2.620503 / .050785 | 2.594345 / .050728 | 2.706235 / .050574 | 2.698432 / .050638 |

Prodromal12 plain Qwen invalid actions 258/1168; static and
temporal hinted policies reported 0. Prodromal24 plain 192/756;
hinted policies reported 0. Across 15 real-Qwen Verifier audits,
**only 1 accepted**, on a Prodromal12 inner fold, 14 rejected.
Actual verified arm still did not beat plain Qwen on main tasks.

### Methodology issues that prevent a positive temporal claim

1. The temporal head produces future Motor likelihoods but its
   six-site binary risk assumes per-item conditional independence.
2. Temporal prediction was blended using a globally fixed .35
   coefficient that was not shown to generalize to every horizon.
3. The time elapsed since true R01 was hardcoded to six months,
   rather than using patient-specific measured visit offset.
4. Different Qwen acquisition actions confound a fair comparison of
   the STATIC versus TEMPORAL risk prediction model.
5. These are repeated internal studies on historically explored
   Prodromal TRAIN, not independent external validation.

## Phase71 prespecified primary comparisons

20 independent normal Slurm CPU jobs = FIVE identical outer patient
folds x FOUR WM variants, with NO Qwen inference. This directly
tests whether the temporal head should be enabled *before*
investing more CPU resources on LLM/RSI.

Variant 0: static masked neural model (no temporal auxiliary).
Variant 1: temporal auxiliary training ON, but temporal fusion OFF.
Variant 2: temporal-trained and fixed fusion 0.35.
Variant 3: temporal-trained fusion selected from {0, .1, .2, .35, .5}
 on DEV using fixed real-evidence two-question inputs; independently
 accepted on AUDIT once only if it improves MAE without materially
 worsening binary Brier versus blend 0.

Every variant reports: identical **fixed-evidence** two-item policy,
and an optional separately scored deterministic WM-greedy two-item
information acquisition protocol. Patient-level paired bootstrap
against a simultaneously fit fold-matched static reference is performed
in memory; no individual predictions or identifiers are exported.
The top two FIXED evidence items are selected by frequency among
FIT patients (with predeclared deterministic lexical tie-break);
outcome labels are not used in the policy ranking. Model reference
and candidate share the same true measurements, labels and folds.

This is a real WM method/ablation gate with shared clinically recorded
BL/R01 and optional baseline non-motor clinical modalities, not an
LLM+RSI paper main experiment. It does not train a causal
disease-intervention model or validate outcomes on untouched PPMI TEST.
All temporal transitions are from FIT-only people with compatible
medication state and real chronological visits. No partial patient
data exported.

Commands and resource allocation are documented in
PHASE71_TEMPORAL_GATE_CPU.md.