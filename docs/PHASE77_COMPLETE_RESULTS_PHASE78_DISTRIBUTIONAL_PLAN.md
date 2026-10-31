# Phase77 verified results and Phase78 distributional WM hypothesis (Oct 31, 2026)

Source `phase77_fixed_reports.zip` uploaded by the project owner.
Archive has many historic empty directories, but **20 newest complete
reports** (arm 0..19, job IDs 335758..335777). Each has
method_metrics.csv, paired_bootstrap.csv, coverage.csv, audit.json,
progress.json(status complete) and summary.json (3 tasks).
There are 300 method rows and 315 paired comparison rows, not 300
independent patients; same patient-fold reference outcomes are
repeated across policy-arm reports and MUST be deduplicated by
(cohort, horizon, outer_fold, method).

## Five-fold weighted mean absolute error (MAE lower better)

| Method | Prodromal12 N584 | Prodromal24 N378 | PD12 N156 |
|---|---:|---:|---:|
| Motor HGB | .99633 | 1.14285 | **2.48944** |
| Multimodal HGB | .99315 | 1.15516 | 2.52688 |
| Motor CT-WM | .87227 | .98009 | 2.58726 |
| Early multimodal CT-WM | .88432 | 1.02249 | 2.62473 |
| Motor CT-WM + FIT OOF median bias | .76658 | **.86081** | 2.71881 |
| Motor CT-WM + FIT OOF nonmotor residual | .77427 | .89325 | 2.77882 |
| Verified bias+residual vs motor rollback | **.76430** | .86689 | 2.54460 |

Other metrics: Prodromal12 CT Motor RMSE 1.45963, bias-only
RMSE **1.55157**; Prodromal24 CT Motor RMSE 1.72675,
bias-only **1.86300**. Brier of bias/residual is identical to
Motor CT-WM since Phase77 changes ONLY severity, NOT risk.

Bias-only vs learned nonmotor residual:
- Prodromal12: correction only .76658 vs residual .77427. Inner
  matched per-fold residual-minus-bias MAE contrasts vary in sign;
  only fold1 clearly favors learned residual.
- Prodromal24: correction only .86081 vs residual .89325:
  all five folds favor bias-only.
- PD12: bias-only 2.71881 vs residual 2.77882; generally worse
  than Motor HGB and Motor CT-WM.
- Verifier gate: 13 of 15 proposed revisions accepted: all 10
  Prodromal tasks, 3/5 PD tasks. Crucially, the candidate includes
  the OOF-learned negative intercept. Thus acceptance is NOT
  proof that nonmotor modality-specific residual learning helped.

Motor CT-WM minus Motor HGB MAE improvement: .12406 (~12.5%)
at Prodromal12 and .16276 (~14.2%) at Prodromal24; PD is worse
by .09782. Bootstrap intervals on individual fold comparisons do
not always exclude zero. Phase62 frozen Prodromal TRAIN has been
repeatedly analyzed throughout many phases, so these are
development-only exploratory metrics, not independent confirmation.

Runtime: each completed Phase77 run reported approximately 18-35 s
with PKS_TORCH_THREADS=1 and MKLDNN disabled. The earlier 20
Phase77 jobs timed out after 12h, caused by a pathological slow
CPU GRU backward path; a later diagnostic exposed an additional
missing-item slope indexing error. Both were fixed.

## Phase78 hypotheses

The original CT model emits a single nonnegative burden via softplus
plus an independent binary risk head. Its mean-oriented Huber training
can favor high-burden cases, while the primary MAE is minimized by
the conditional median. A FIT-OOF negative intercept improves MAE
but harms RMSE in Prodromal, suggesting prediction-functional
misalignment rather than value added by nonmotor modalities.

Develop two MOTOR-ONLY distributional variants using identical
chronological true visible27 historical features, same patient
folds and FIT-only true adjacent-visit auxiliary loss:
- Hurdle: separate probability of ANY positive future site burden,
  plus positive-only expected severity, with mean and approximate
  median decision functions.
- 25-category future six-item total burden (0..24) with calibrated-in-
  principle categorical posterior; report exact expected mean,
  categorical median and risk=1-p0.
- Compare both against original CT Motor and same-fold HGB. In each
  fold report MAE, RMSE, Brier, AUROC and paired bootstrap. The
  conditional mean and median are distinct predictions even though
  generated from the same model.
- A strictly allowlisted candidate predictor decision is selected
  on a separate DEV subset and audited ONCE on AUDIT with guardrails
  (MAE improvement, RMSE and Brier noninferiority), then evaluated
  on outer held-out patients.
- No Qwen, no medically actionable treatment advice, no autonomous
  neural RSI: this is a **WM forecast-distribution and verifier
  correctness** experiment. Conditional distributions require
  calibration assessment before any clinical claims.

Twenty normal CPU jobs = 5 outer patient folds x 4 arms.
Results still on exploratory frozen TRAIN; do not treat 20 jobs
as 20 independent datasets, and do not average fold AUROCs as
pooled patient-level AUROC.
