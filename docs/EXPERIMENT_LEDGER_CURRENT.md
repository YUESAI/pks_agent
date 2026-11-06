# Unified experiment ledger — Phase60 onward (living)

**Updated:** 2026-11-06. **Branch:** `feat/phase62-ppmi-cohort-expansion`. This is the **single starting index** for all post-Phase59 research on this working branch.

**Phase23–58 original ledger:** [EXPERIMENT_LEDGER_PHASE23_58.md](EXPERIMENT_LEDGER_PHASE23_58.md). **Phase59** real Qwen four-arm pilot and Phase60 are summarized in [PHASE61_CPU_PAIRED_VERIFIER.md](PHASE61_CPU_PAIRED_VERIFIER.md). The old ledger does **not** contain Phase60+, which caused the apparent missing-history problem. It was never automatically updated by the individual phase writeups.

### Read this before starting any new experiment

- Search [structured registry](experiment_registry.json) by model mechanism, split and phase; optionally run `python scripts/experiment_preflight.py --query 'your idea'`.
- Repeated phase80–83 active-test selection, OOF-abstention, Qwen tools and verifier loops did **not** demonstrate stable uplift versus fixed clinical test; **do not rerun** those methods with only prompt edits.
- Repeated Phase84–90 results all use the **same selected, repeatedly viewed patient-global 80/20 development split**. Do not report these as independent estimates, do not compare across different tasks or distinct information budgets.
- Do not touch original consumed Prodromal TEST. Require actual independent confirmation only **after** freezing a candidate. Historical scores are exploratory, even when nominal 5fold was run.
- Register all new designs and outcomes in this ledger and the JSON registry (include job IDs and exact source report), update the README link, and review the overlap matrix **before any Slurm submission**.

## Current research state / best *development* candidates

| Prediction objective | Development candidate | RPS ↓ | Other metric | Evidence limitation |
|---|---|---:|---|---|
| Prodromal 12-month future six hidden-site total | **P90 full33 joint static**, P90 Arm2 | **0.022504544** | Median MAE 0.657895 | Only 114 repeatedly selected DEV subjects |
| Prodromal 24-month future six hidden-site total | **P85 shared static K4 six-site mixture**, P90 Arm0 rerun | **0.022843898** | Median MAE 0.676056 | Only 71 repeatedly selected DEV subjects |
| PD 12-month future six hidden-site total | **P87 observed-visit dynamic flow**, P90 Arm1 rerun | **0.063139055** | Median MAE 2.135135 | Only 37 repeatedly selected DEV subjects |
| Same-target-visit hidden-six reconstruction after observing true visible27 | P90 conditional full33 mixture, notably dynamic in Prodromal and static in PD | See [P90 actual results](PHASE90_ACTUAL_RESULTS_AND_DECISION.md) | Outcome has *more information* than future forecast | Not comparable with future forecast; conditioning can worsen empirical scores |

## Phase91: pre-registered, no results yet

| Phase | Mechanism | Status | Pre-registered design | Primary document |
|---|---|---|---|---|
| **91** | True FIT-only posterior supervision with full-27 / randomized observed subset masks | **PLANNED_NOT_RUN** | 3 new arms: full-visible static; subset-aware static; subset-aware dynamic. Compare two distinct information budgets (future forecast vs observed-target-visit masked recovery); match original 80/20 patient split and archived Phase90. No duplicated older Phase90 baseline job. | [P91 preregistration](PHASE91_PRE_REGISTERED_PROTOCOL.md) |

Research motivation: in P90 an exact Bayes conditional posterior sometimes had **worse observed RPS** than the future prior despite access to real same-visit visible27, which is compatible with incorrect conditional likelihood calibration. P91 explicitly scores the conditional posterior during FIT using genuine FIT hidden6 outcome and genuine complete33 measured visit, while retaining pre-cutoff future forecast loss. This is a different *training objective*, not an independent test of the resulting model. No results are known yet.

## Full historical experiment index

Status definitions: `COMPLETE` means documented outputs or prior completed result report; `COMPLETE_REPLICATED` independently repeated configuration; `COMPLETE_AFTER_FIX` succeeded after the original failed runs; `PARTIAL_TWO_FOLDS` means **not** a fivefold result; `DOCUMENTED_PROTOCOL` / `HISTORICAL_SUMMARY` / `COHORT_AUDIT` must **not** be upgraded to verified completion without raw evidence.

### Phase60–63: compute and cohort protocols

| Phase | Mechanism / question | State | Key recorded result / warning | Full result source |
|---|---|---|---|---|
| 60 | same-visit multi-step mixture rollout | `COMPLETE` | 4/8/12-question budget; 4 arms produced identical outcome metrics; K1 item NLL .4510 better K2 .4537, K3 .4559 | [P60](PHASE61_CPU_PAIRED_VERIFIER.md) |
| 61 | paired deterministic planner and verifier | `DOCUMENTED_PROTOCOL` | Retrospective paired verifier CPU protocol; completion metrics not independently verified from checked source | [P61](PHASE61_CPU_PAIRED_VERIFIER.md) |
| 62 | PPMI strict cohort eligibility expansion | `COHORT_AUDIT` | Strict Motor longitudinal labels; Prodromal12 n584, Prodromal24 n378, PD12 n156; cohort feasibility not predictive superiority | [P62](PHASE62_MULTICOHORT_CPU.md) |
| 63 | matched classical baselines | `HISTORICAL_SUMMARY` | Ridge / elastic net / forest / HGB evaluated; results vary by cohort and horizon; consult primary source | [P63](PHASE63_TWENTY_CPU.md) |

### Phase64–73: early multimodal WM / Qwen / verifier comparisons

| Phase | Mechanism / question | State | Key recorded result / warning | Full result source |
|---|---|---|---|---|
| 64 | Qwen WM and audited clinical action policies | `COMPLETE` | 20 folders with per-arm exploratory Qwen and WM predictions; no stable WM/LLM advantage | [P64](PHASE64_ACTUAL_RESULTS_PHASE65_PLAN.md) |
| 65 | masked multimodal clinical agent | `COMPLETE` | Plain Qwen sometimes better than WM guidance; no consistent benefit from WM or RSI-lite | [P65](PHASE65_RESULTS_PHASE66_DESIGN.md) |
| 66 | fivefold Qwen choice / WM hints / verifier | `COMPLETE` | 20 arm folders and 60 task metrics; WM hints reduce invalid actions but not stable MAE/Brier uplift | [P66](PHASE66_RESULTS_PHASE67_PLAN.md) |
| 67 | neural WM and Qwen strategy comparisons | `COMPLETE` | 20 arm reports; nonuniform, weak performance changes; no conclusive clinical uplift | [P67](PHASE67_RESULTS_PHASE68_RUN.md) |
| 68 | multimodal neural WM selection | `COMPLETE` | 20 complete arm folders; full clinical-feature heads not reliably better | [P68](PHASE68_RESULTS_PHASE69_DESIGN.md) |
| 69 | temporal WM Qwen verifier | `COMPLETE` | 20 complete Qwen strategy folders; mixed results and no robust verifier benefit | [P69](PHASE69_RESULTS_PHASE70_DESIGN.md) |
| 70 | horizon-fused time-aware WM | `COMPLETE` | 20 arms; temporal Qwen+WM generally not better than static or plain Qwen; 1/15 verifier accepted | [P70](PHASE70_RESULTS_PHASE71_DECISION.md) |
| 71 | static versus temporal WM mechanism tests | `COMPLETE` | 20 full arms, modest/inconsistent temporal changes | [P71](PHASE71_72_RESULTS_PHASE73_PLAN.md) |
| 72 | advanced WM and RSI proxy | `PARTIAL_OR_INCOMPLETE` | Slow Qwen workload and partial arms; no reliable measured gain from advanced rollout | [P72](PHASE71_72_RESULTS_PHASE73_PLAN.md) |
| 73 | matched continuous-time neural WM Qwen agent | `COMPLETE` | Prodromal12 Ridge MAE .9920 vs CT WM 1.0417; no consistent WM/Qwen win | [P73](PHASE73_RESULTS_PHASE74_PROTOCOL.md) |

### Phase74–79: direct clinical trajectory and probabilistic WM

| Phase | Mechanism / question | State | Key recorded result / warning | Full result source |
|---|---|---|---|---|
| 74 | direct clinical trajectory WM | `PARTIAL_TWO_FOLDS` | 8/20 folders with result files; CT promising but optional input parity mismatch vs baseline | [P74](PHASE74_RESULTS_PHASE75_INPUT_PARITY.md) |
| 75 | input-parity multimodal trajectory WM | `PARTIAL_TWO_FOLDS` | 8/20 result arms; Prod12 Motor CT MAE ~.765, HGB ~.948; two-fold only | [P75](PHASE75_RESULTS_PHASE76_PLAN.md) |
| 76 | reliability-gated multimodal CT WM | `PARTIAL_TWO_FOLDS` | Motor CT Prod12 MAE .76386; verified .70256 on only 2 folds; reliability gate near default | [P76](PHASE76_RESULTS_PHASE77_PROTOCOL.md) |
| 77 | crossfit bias/residual Motor WM | `COMPLETE_AFTER_FIX` | Motor CT Prod12 MAE .87227; FIT OOF median bias .76658 but RMSE tradeoff; initial 20 jobs timed out, later corrected | [P77](PHASE77_COMPLETE_RESULTS_PHASE78_DISTRIBUTIONAL_PLAN.md) |
| 78 | hurdle and categorical probabilistic Motor WM | `COMPLETE` | Categorical median Prod12 MAE .75856; hurdle median Prod24 MAE .84392; mean/median RMSE tradeoff | [P78](PHASE78_RESULTS_PHASE79_PROTOCOL.md) |
| 79 | ordinal proper score / temperature verification | `COMPLETE` | Ordinal RPS gains small; temperature accepted 1/15; use median MAE vs mean RMSE | [P79](PHASE79_RESULTS_PHASE80_AGENT_PROTOCOL.md) |

### Phase80–83: Qwen and clinical information acquisition

| Phase | Mechanism / question | State | Key recorded result / warning | Full result source |
|---|---|---|---|---|
| 80 | real Qwen active baseline test acquisition | `COMPLETE` | Fixed query best Prodromal RPS; 2449 Qwen calls, verifier accepted 2/15; no LLM uplift | [P80](PHASE80_RESULTS_PHASE81_OOF_POLICY.md) |
| 81 | FIT OOF value-of-information abstention | `COMPLETE` | OOF abstains often, Qwen never independently abstained, fixed query RPS better; 1/15 verifier switch | [P81](PHASE81_RESULTS_PHASE82_VALUE_AUDIT.md) |
| 82 | cost-aware value filter and hindsight oracle | `COMPLETE` | 354 guarded Qwen calls, 1/15 verifier acceptance; oracle retrospective headroom ~9%-13%, not realizable | [P82](PHASE82_RESULTS_PHASE83_CONDITIONAL.md) |
| 83 | risk-conditional clinical likelihood + Qwen | `COMPLETE_AFTER_FIX` | 20/20 after 5 Qwen KeyError reruns; 245 Qwen calls, 0/15 verifier accepts, conditional acquisition no stable uplift; inner OOF dependence caveat | [P83](PHASE83_FINAL_FIVEFOLD_RESULTS_AND_RESEARCH_DECISION.md) |

### Phase84–90: reproducible single-split structural WM development

| Phase | Mechanism / question | State | Key recorded result / warning | Full result source |
|---|---|---|---|---|
| 84 | site-consistent future six Motor item convolution | `COMPLETE` | Prod24 independent six-site RPS .023108 vs categorical .024387; aux weight worsened Prod24 | [P84](PHASE84_RESULTS_PHASE85_MIXTURE_WM.md) |
| 85 | correlated K4 six-site mixture, horizon sharing | `COMPLETE_REPLICATED` | Best Prod12 RPS .023103; Prod24 .022844; PD12 separate .067275; four arms repeated with matching metrics | [P85](PHASE85_RESULTS_PHASE86_CONTINUOUS_TIME_WM.md) |
| 86 | continuous-time latent flow and semigroup | `COMPLETE` | Prod12 .023394, Prod24 .023143 both worse than P85; PD12 .064143 improved; semigroup loss negligible | [P86](PHASE86_RESULTS_PHASE87_VISIT_ANCHORED_WM.md) |
| 87 | FIT real next-visit supervision for latent flow | `COMPLETE` | Prod12 RPS .022642 (weight .05), PD12 .063139 (weight .15), Prod24 remains behind P85 | [P87](PHASE87_RESULTS_PHASE88_SELECTIVE_WM.md) |
| 88 | jointly trained static/dynamic gate | `COMPLETE` | Gates near constant, Prod24 and PD underperform independent expert choices; shared-head limitation | [P88](PHASE88_RESULTS_PHASE89_OOF_EXPERT_GATING.md) |
| 89 | independent Phase85/87 experts + FIT OOF gate | `COMPLETE` | Patient gate Prod24 RPS .022783, Prod12 .022932, PD12 .063256; near-constant per cohort gates, limited uplift | [P89](PHASE89_RESULTS_PHASE90_PARTIAL_OBSERVABILITY.md) |
| 90 | full33 partially observed joint future/same-visit model | `COMPLETE` | Future: static full33 Prod12 RPS .022504544 best; Prod24 P85 .022843898 best; PD12 P87 .063139055 best. Same-visit masked recovery is a distinct task. | [P90](PHASE90_ACTUAL_RESULTS_AND_DECISION.md) |

## Do-not-repeat map (same input/split without a *new* scientific test is redundant)

| Attempt family | Already explored | What was learned | What would justify a genuinely new experiment |
|---|---|---|---|
| Unconstrained Qwen test choice, WM expected-KL, hard value filter, OOF abstain, verifier gate | Phase64–83, especially 80–83 | Fixed rule at least as reliable on RPS; thousands of Qwen calls without independent gain | New *prospective* observation budget, grounded clinical utility, actual independent patient cohort |
| Repeated 5fold active acquisition on original frozen TRAIN | 62–83 | Many variants tried, multiple incomplete intermediate stages | A pre-specified confirmatory design rather than further TRAIN re-selection |
| Direct total categorical, hurdle, ordinal loss, residual and temperature calibration | 77–79, 84 | Median-vs-mean tradeoff; calibrated gains limited | New outcome or structural data mechanism, not another minor loss weight |
| Six-site independent vs correlated K4 mixture, sharing 12m/24m | 84–85 | Phase85 strongest long-term Prodromal candidate | New genuinely untested temporal / observation mechanism with a frozen dev protocol |
| Explicit continuous-time flow, semigroup, real adjacent-visit supervision | 86–87 | Unsup flow and semigroup not consistently helpful; real visits helped 12m/PD | Multi-timepoint truth-consistent transition supervision |
| Jointly trained static/dynamic gate, independently trained experts with 2fold OOF gate | 88–89 | Learned near-constant group preference; refit shift and small 24m gain | Gate calibrated on matched training-size experts or distinct uncertainty hypothesis, not another linear gate |
| Unified full33 partial-observation mixture and masked same-visit posterior | 90 | New best Prod12 future RPS; same-visit conditional recovery mixed, sometimes harmful | Explicit calibration of conditionals and proper same-visit matched baselines |

## Protocol comparability and privacy

1. P60–83: historical heterogeneous tasks, most on the original frozen TRAIN with five patient folds. P74–76 only two folds returned; never pool them with complete fold results as if coverage is identical.
2. P84–90: fixed **80/20 patient-global** split; Prodromal12 470 FIT /114 DEV, Prodromal24 307/71, PD12 119/37. Method selection has repeatedly used the exact same DEV; its reported numbers are exploratory optimization guidance.
3. For a probability distribution on future six-site total 0..24, report RPS, NLL, Brier; calculate point MAE from the distribution median and RMSE from its mean. Do not conflate with a same-visit 27-observed recovery task.
4. Proper causal clinical interventions, autonomous self-rewriting RSI, prospective clinical planning, and confirmed externally valid patient-level benefits are **not** established by these outcomes.
5. Never commit patient-level identifiers, raw scores, individual predictions, prompts, unreviewed logs or raw PPMI. Committed data here are aggregate-only.

## Result provenance and automatic record keeping

- Each phase row links to its dated primary protocol/results document. A reference to a protocol file does **not** assert that a missing raw run was independently verified. This register was consolidated from GitHub historical records and the Phase90 result ZIP checked on 2026-11-05.
- Full Phase90 actual results: [PHASE90_ACTUAL_RESULTS_AND_DECISION.md](PHASE90_ACTUAL_RESULTS_AND_DECISION.md). Jobs 342535–342538; completion and all table values checked directly against uploaded summary and metric CSVs.
- Full list for programmatic preflight: [experiment_registry.json](experiment_registry.json); checker: [../scripts/experiment_preflight.py](../scripts/experiment_preflight.py).
- For every new experiment before running: record phase, **hypothesis**, cohort, horizon, observation budget, source/manifest, patient split key, outcome definition, code commit, random seeds, GPU/CPU budget, ablation rationale, and a new unique protocol signature. After run add Slurm job IDs, status, source report ZIP hash or storage reference, all aggregate metrics, caveats and winner criteria.
- Do not silently overwrite historical results. Append Phase91+ as new rows, extend structured registry, and keep README's 'latest ledger' link current.

**Canonical registry update rule:** New experiments go into `experiment_registry.json` with `PLANNED_NOT_RUN` first; completed Slurm job IDs and outcomes are appended only after inspecting real outputs. The numbered `EXPERIMENT_LEDGER_PHASE60_90.md` and `experiment_registry_phase60_90.json` remain historical snapshots and do not auto-update. This living ledger is the authoritative index for Phase60 onward.

**Repository visibility:** GitHub default branch `main` is historically behind. This ledger and Phases60–90 are on `feat/phase62-ppmi-cohort-expansion`, not merged into `main`. To see them online, choose that branch in GitHub. Avoid accidental switching to `main` when looking for newest work.
