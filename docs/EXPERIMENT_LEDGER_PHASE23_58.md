# VERA-PD cumulative experiment ledger — Phase23–58

Last updated: **2026-10-22**. This file indexes findings already
documented throughout the repository; it must not replace original
source reports or retroactively change the frozen Phase23–44 policy.
**Research status: exploratory. Original Phase44 final TEST consumed.**

## Immutable distinctions

- **Task A: onsite V04 masked same-visit observation simulation.**
  Doctor may request recorded V04 *visible* scores only. Hidden six
  scores are ground-truth targets, never observable tool responses.
  A site's real onsite assessment is not demonstrated by this simulation.
- **Task B: genuine earlier remote R01 -> months-later onsite V04.**
  The six hands-on scores are **101/UR at R01**, not measured zero.
  A later V04 score is an observed future outcome, not a
  contemporaneous remote label. The strict Phase56–58 R01 cohort is
  197 subjects, distinct eligibility from earlier Phase36–44 199.
- **Task C: longitudinal next-visit score modelling**, with target
  measured-item masks and irregular observed visit times. Supplying
  the next visit's actual month is retrospective time-conditioned
  forecasting, not prediction of appointment timing.
- **Testing:** Phase44 TEST was finally evaluated; it cannot serve
  as a new independent test for Phase45+ architecture searches.
  Patient-disjoint TRAIN folds are useful for internal engineering
  and comparison, not confirmatory external generalization.

## Study ledger

| Stages | Experiment and data | Observed outcome | What it DOES NOT establish | Source |
|---|---|---|---|---|
| 23–30 | Frozen Prodromal split; Motor BL+12m→24m | Train=1312/Val=281/Test=281; 332 Train three-visit total-score sequences; 50/50 history smoothing MAE **2.908** vs Ridge **3.069** | Learned temporal state superior to history | [Research snapshot](RESEARCH_SNAPSHOT_PHASE23_44.md) |
| 31–38 | Actual onsite complete Motor V04, mask six site-only items | 605 Train V04; always-zero six-item MAE **0.7785** vs 27-score Ridge **0.8477**; simulated 20% visible-sum selection captured 81/206 positives | True onsite referral efficacy | [Research snapshot](RESEARCH_SNAPSHOT_PHASE23_44.md) |
| 39–44 | Genuine R01 (27 measured + six 101/UR) → later V04 | Lock policy and report identification-bounded recall lift; Train/Val/Test outcomes recorded; **TEST consumed** | Same-visit remote hidden truth or new final test | [Reproduction protocol](phase23_44_reproduction.md) |
| 45 | Categorical Naive Bayes observation WM + constrained Doctor/Patient + RSI-lite | TRAIN-only initial software feasibility, q-limited observation, offline mutation/verification | LLM Doctor, modern temporal WM or full code-edit RSI | [Phase45 guide](PHASE45_WM_RSI_PILOT.md) |
| 46–47 | Real Qwen3-4B Doctor; q=4/8/12/16 same-V04 observation rollout | Qwen backend on Dusky L40S, strict tool adapter; NB-guided Doctor could imitate NB planner, with little demonstrated gains | The upper bound of advanced WM guidance or real R01 interactions | [Phase46 guide](PHASE46_QWEN_DOCTOR.md), [Phase47](PHASE47_QUERY_BUDGET_CURVES.md) |
| 48–49 | Neural masked-item Transformer and decision-aligned ordinal/binary loss | Transformer reduced masked-item NLL ~3.5% vs NB in a 100-person Train subset; downstream decision gain not established | Modern longitudinal clinical state-transition WM | [Phase48](PHASE48_NEURAL_WM.md), [Phase49](PHASE49_DECISION_ALIGNED_WM.md) |
| 50–52 | Dreamer/JEPA-inspired BL+12m→24m scalar latent model | 332 Train trajectories. 50/50 baseline MAE **2.90813**; Phase50 WM **2.91681**; Phase52 5fold WM **2.94740**, 10fold **2.93544**; no reliable win | Actual Dreamer4 checkpoint or efficacy of imagination | [Phase52 actual results](PHASE52_5FOLD_10FOLD_RESULTS.md) |
| 51–53 | Real Train longitudinal data capability audits with item masks | Phase53: **2702** usable complete/partial episodes, **1165** BL anchored patients, **1490** anchored transition pairs, **658** patients ≥2 valid visits; real R01 27+6/UR retained | 2702 independent patients or causal intervention transitions | [Phase53 actual results](PHASE53_RESULTS_PHASE54_MASKED_LONGITUDINAL_PROTOCOL.md) |
| 54–55 | Masked 33-item recurrent temporal latent WM vs learned Markov | **658** Train patient-disjoint OOF; temporal WM next-item NLL **0.413298**, tuned Markov **0.410580** (WM−Markov +0.002718, 95% [-0.002875,+0.008620]) | WM superiority over strong statistical temporal control | [Phase55 actual results](PHASE55_ACTUAL_RESULTS_DECISION.md) |
| 56–57 | True earlier R01 → future verified V04 six-item result | Strict **197** Train pairs, **57** future-positive; WM NLL **0.315892** vs Markov **0.314551**; top-39 capture WM **19**, Markov **23**, R01 visible 27 sum **22** | Statistically stable clinical selection win or causal referral effect | [Phase56 & 57 results](PHASE56_ACTUAL_RESULTS_PHASE57_BASELINE.md), [Phase57](PHASE57_ACTUAL_RESULTS_PHASE58_PAIRING.md) |
| **58** | **Ten matched strategies**, 197 same real R01→V04 TRAIN cases | Fixed WM64+Markov 50/50 fusion top-39 **24/57**, AUROC **0.730263**; Markov **23/57**, visible sum **22/57**, neural WM64 **20/57**; fusion−Markov Recall CI **[-0.0541,+0.1063]** | Confirmed superiority (10 policies tested, TRAIN reused); genuine WM planning, LLM RSI | [Phase58 actual report](PHASE58_ACTUAL_RESULTS.md) |

All metrics above were transcribed from the corresponding Phase result
documentation, not recomputed from private patient data in GitHub.
Original private data are restricted to cluster. Some earlier reports
were supplied to the chat as ZIP files; the full original source
archives are **not** committed to the repository.

## Current standing (frozen interpretation)

- **Operationally demonstrated:** real offline Qwen3-4B function
  selection on Dusky L40S; 33-item mask-aware clinical temporal WM;
  fit-only Markov and simple score baselines; nested patient CV;
  constrained offline RSI-lite, separate from Qwen/temporal WM.
- **Scientific candidate, not confirmation:** a *prespecified
  50/50 WM64 + Markov* ensemble may contain complementary information.
  Its point Recall gain is **one of 57 future positives** vs Markov;
  its paired Recall interval crosses zero. Its AUROC difference
  is nominal uncorrected multiple-comparison/exploratory.
- **Not done yet:** one integrated real Qwen Doctor ↔ longitudinal
  WM ↔ actual evidence service ↔ independent verifier ↔ versioned
  coupled RSI system; calibrated multistep imagination; truly new
  external patient validation.
- **Privacy note:** Phase58 historical overlap table contains small
  (<10) overlapping-positive counts; do not publish the original
  unsuppressed table. Later code applies suppression.

## Change control

Never silently overwrite older metrics or rename old subjects.
Each new experiment needs: git commit, immutable config/seeds,
task/eligibility, split lineage, model checkpoint hash, allowed
observation protocol, trained-fit count, inner verifier count,
outer evaluation count, compute/token budget, missingness handling,
per-patient unit of resampling, failure modes, and aggregate-only outputs.

New final claims require genuinely independent patients/cohorts
and a prespecified protocol; do not re-open Phase44 Test.
