# Phase58 completed ten-policy benchmark (Dusky job 330900)

Source: user-uploaded `phase58_ten.zip`, created 2026-10-22
04:55 UTC. Contains ten_strategy_metrics.csv, paired_bootstrap.csv,
topk_overlap.csv, fold_training_compute.csv and summary.json.
Frozen Prodromal TRAIN, real R01 27 scored items -> later complete V04
six-site burden. **197** patients, **57** future positives,
**39** checked at strict 20% quota, five patient-disjoint OOF folds.
Previously consumed original TEST was not read. These remain
repeated exploratory TRAIN analyses.

## Ten strategies in original report

| Method | Positives at k=39 | Recall | AUROC |
|---|---:|---:|---:|
| visible27_sum | 22 | .385965 | .675188 |
| visible27_nonzero | 19 | .333333 | .671742 |
| visible27_ridge | 23 | .403509 | .656892 |
| visible27_logistic | 22 | .385965 | .647243 |
| visible27_naive_bayes | 20 | .350877 | .656391 |
| markov_tuned | 23 | .403509 | .650940 |
| latent_wm_64 | 20 | .350877 | .665226 |
| latent_wm_128 | 20 | .350877 | .667043 |
| **wm_markov_blend** | **24** | **.421053** | **.730263** |
| visible_wm_blend | 22 | .385965 | .674624 |

WM+Markov 50/50 fusion outperforms either component numerically,
suggesting complementary ranking information. Do NOT claim
confirmatory gains: multiple methods were assessed and the same
TRAIN people have been repeatedly explored. Five-fold OOF
does not provide a fresh independent test.

Paired 500-bootstrap contrasts:
- blend minus Markov: recall +.017544,
  CI [-.054083,+.106250]; AUROC +.079323,
  CI [+.003315,+.163330].
- blend minus visible sum: recall +.035088,
  CI [-.068410,+.183586]; AUROC +.055075,
  CI [-.010252,+.119353].
- WM128 minus WM64: recall .000000,
  CI [-.122637,+.074313]; AUROC +.001817,
  CI [-.044881,+.046459].

Bootstrap did not refit neural/Markov models, nor account
for exploratory model selection. The AUROC interval for the
blend versus Markov is only **nominal** and uncorrected
for the ten policy trials. The clinical recall-at-quota
improvement remains uncertain.

## Top-k set complementarity

Both markov_tuned and latent_wm_64 select 39 people but overlap
at only **15** (Jaccard .238095); 9 of the common selections
are future-positive, demonstrating different rankings.
WM+Markov vs Markov overlaps 28 (Jaccard .560000).
WM+Markov vs visible27_sum overlaps 20 (Jaccard .344828).
These are selection-structure clues, not a demonstration of
clinical action benefit.

## Runtime and model size (correcting expectations)

- CPU total **167.65 seconds** across ten methods/five folds,
  including data and nested fitting.
- Five fold walltime components:
  39.72, 29.25, 29.12, 35.93, 33.26 seconds.
- Width64 recurrent WM has **55,205 parameters**.
- Width128 recurrent WM has **192,165 parameters**.
- Selected epochs WM64 [25,10,5,25,25] and
  WM128 [10,5,10,10,2].

This is not GPU under-utilization or a secretly pretrained
billions-of-parameters World Model. The two neural models
are very small and fast on 658 patient trajectories.
WM128 did not improve over WM64 in this evaluation.

## Next experimental gate

Lock the 50/50 WM64+Markov policy as a fixed
**exploratory candidate** and stop searching repeatedly on
these same 197 people. Before claiming WM value:

1. Validate on **genuinely new independent patients**, with
   prespecified quota, positive-label definition, and
   fixed model-selection procedure. Original final TEST is consumed.
2. Compare with an equal-compute/parameter strong discriminative
   ensemble, **not** only either constituent.
3. Test multi-step open-loop forecasts under actual clinical
   measurement masks, rather than only retrospective one-step
   risk sorting.
4. Connect to Qwen Doctor for actual sequential observation
   actions and verifier-gated RSI once the world-model
   representation is demonstrably useful.
5. Use proper patient-level calibration, risk intervals,
   and physician referral burden; a product-of-six marginal
   probability is not a calibrated joint outcome estimate.

## Privacy

The uploaded historical report contains some
`both_capture_positive` counts below 10 in the top-k overlap
table (e.g. 8 and 9). These are small aggregate cells, not
patient identifiers; however they violate our conservative
minimum-cell=10 reporting convention.
The metric export code was subsequently amended on
`feat/parkinson-data-audit` to suppress small overlap-positive
cells. There is no need to re-run the numerical experiment
solely to change historical results. Do not publicly release
the original overlap table without suppression.
