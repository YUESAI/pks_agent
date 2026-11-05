# Phase90 actual results — 33-item partial-observation WM (2026-11-05)

**Evidence source:** user-supplied `phase90_reports.zip` (received 2026-11-05), containing 16 files in exactly four complete job folders: 342535, 342536, 342537, 342538. All jobs have `summary.json`, `progress.json` with `status=complete`, `coverage.csv`, and `dev_metrics.csv`; 3/3 tasks completed each. Only aggregate statistics are recorded here; no original patient-level data, IDs, or predictions are committed.

## Protocol and repeatability

One globally hashed patient-disjoint **80% FIT / 20% DEVELOPMENT** split, already inspected repeatedly in Phases84–89. Same FIT/DEV SHA-256 membership fingerprints across all four arms. Prodromal 12m: FIT470, DEV114; Prodromal 24m: FIT307, DEV71; PD12m: FIT119, DEV37. Original previously consumed final TEST never accessed. Prodromal training has 1,176 actual FIT-only adjacent-visit pairs after 12/24 deduplication, PD 338 pairs. Four ordinary standalone CPU jobs, 18 epochs default. This is **not** a fresh holdout or an independently validated study.

Arms: 0 Phase85 static six-site correlated mixture; 1 Phase87 true-visit-supervised dynamic mixture; 2 NEW joint 33-item static mixture with partial-observation likelihood; 3 NEW joint 33-item dynamic mixture with partial-observation likelihood. Reported future outcomes are hidden six target total 0..24. Each target is a real complete 33-item exam.

## Task A: genuine future forecast — pre-cutoff history only

All metrics lower is better; MAE from distribution median, RMSE from distribution mean, RPS evaluates all 25 total-score categories.

| Task | Arm | Median MAE | Mean RMSE | RPS | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Prodromal 12m (114 DEV) | 0 | 0.736842 | 1.378695 | 0.023103037 | 1.114543 | 0.196362 |
| Prodromal 12m | 1 | 0.666667 | 1.345196 | 0.022642466 | 1.108317 | 0.196894 |
| Prodromal 12m | **2** | **0.657895** | 1.345737 | **0.022504544** | **1.105624** | **0.192409** |
| Prodromal 12m | 3 | 0.684211 | **1.342531** | 0.022512851 | 1.106605 | 0.195329 |
| Prodromal 24m (71 DEV) | **0** | **0.676056** | **1.391295** | **0.022843898** | 1.105538 | 0.194905 |
| Prodromal 24m | 1 | 0.718310 | 1.410088 | 0.023180877 | 1.104924 | 0.197105 |
| Prodromal 24m | 2 | 0.718310 | 1.396470 | 0.023078527 | 1.108002 | 0.198874 |
| Prodromal 24m | 3 | 0.718310 | 1.401546 | 0.023020852 | **1.100245** | **0.193590** |
| PD 12m (37 DEV) | 0 | 2.189189 | 2.963295 | 0.067274782 | 2.415542 | **0.005530** |
| PD 12m | **1** | **2.135135** | **2.794671** | **0.063139055** | **2.361246** | 0.005984 |
| PD 12m | 2 | 2.405405 | 3.095654 | 0.070967031 | 2.465033 | 0.006641 |
| PD 12m | 3 | 2.243243 | 2.839929 | 0.065243713 | 2.404739 | 0.005632 |

Phase90 **Arm2 is the new best Prodromal12 future RPS** 0.02250454 (0.61% better than Phase87 Arm1 RPS 0.02264247). Phase85 **Arm0 remains best Prodromal24** RPS 0.02284390; Phase87 **Arm1 remains best PD12** RPS 0.06313906. Do not declare a single all-task winner or assume that conditional reconstruction gains imply future forecast gains.

Effective number of mixture components on DEV: Arm2 approximately 2.34/2.47/2.99 and Arm3 approximately 2.26/2.35/2.27 across Prodromal12/24/PD12 respectively. These latent factors are NOT independently discovered clinical disease subtypes.

## Task B: retrospective same-visit masked recovery — different information budget

The model receives the 27 *actually observed* visible item scores from the same target visit **after that visit occurred**; it then predicts the six hidden scores. These are not forecast-time inputs and this is NOT a 12m/24m future prediction under the pre-cutoff information budget. Arm0/1 do not implement this conditional recovery and their fields are blank (not failures).

| Task | Arm | Future-mode RPS (no target-visible27) | Same-visit conditioned RPS | Change on revealing visible27 |
|---|---:|---:|---:|---:|
| Prodromal12 | 2 | 0.022504544 | 0.023857876 | **worse** (+0.001353333) |
| Prodromal12 | 3 | 0.022512851 | **0.021724460** | better (-0.000788391) |
| Prodromal24 | 2 | 0.023078527 | 0.025078095 | **worse** (+0.001999567) |
| Prodromal24 | 3 | 0.023020852 | **0.022585271** | better (-0.000435580) |
| PD12 | 2 | 0.070967031 | **0.057189258** | better (-0.013777772) |
| PD12 | 3 | 0.065243713 | 0.059626874 | better (-0.005616839) |

The same-visit conditioning procedure is an exactly specified mixture-component posterior update under the fitted model; it is **not guaranteed to improve an empirical finite-sample proper score** when the joint distribution is miscalibrated. Arm2's Prodromal degradation is therefore a meaningful failure mode. In future analyses compare against proper same-visit baselines and expose a conditional NLL/RPS calibration check; do not fuse the recovery metric with future forecasting.

## Decision and no-repeat rule

- **Keep** Phase85 static shared mixture for Prodromal24; Phase87 observed-visit-supervised flow for PD12; Phase90 static full33 for Prodromal12 as DEVELOPMENT candidates.
- **Do not rerun** Phase84–90 identical 80/20 splits, 18-epoch models and original endpoint labels merely to reproduce the same aggregate results; split has already been used for repeated model selection.
- **Do not restart** Phase80–83 Qwen, evidence-value, hard-gate, or active acquisition strategy loops without a new real-world query utility definition or new independent evidence; fixed-policy controls repeatedly performed as well or better.
- The next scientific innovation should address observed conditional-state calibration and multi-timepoint consistency rather than yet another nearly constant expert gate. Prior to publication require an independent untouched confirmation population and formal fair baseline, robustness, ablation and uncertainty audits.
- Log every new unique research design in `docs/experiment_registry_phase60_90.json` / `docs/EXPERIMENT_LEDGER_PHASE60_90.md` *before* running jobs. Do not label DEV results confirmatory or claim an autonomous/causal RSI.

**Source scripts:** [Phase90 main code](../scripts/phase90_masked_joint33_wm.py), [Slurm job](../slurm/phase90_dusky_cpu.sbatch), [tests](../tests/test_phase90_joint33.py); predecessor [Phase89 ledger](PHASE89_RESULTS_PHASE90_PARTIAL_OBSERVABILITY.md).
