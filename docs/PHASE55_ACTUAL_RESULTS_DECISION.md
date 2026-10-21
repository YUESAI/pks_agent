# Phase55 actual results and decision gate (2026-10-21)

**Source:** user-uploaded aggregate archive `phase55_markov.zip`,
Slurm job **330749**, created at 2026-10-21 23:41 UTC.
Only frozen PPMI Prodromal TRAIN patients were used;
original Phase44 TEST was consumed earlier and has not been reopened.

## Observed metrics, 658 patient-disjoint OOF predictions, 5 folds

| Method | Mean patient next-visit observed-item categorical NLL |
|---|---:|
| Tuned per-item Markov transitions | **0.410580** |
| Mask-aware time-conditioned recurrent latent WM | **0.413298** |
| Fit-TRAIN item marginals | 0.448684 |
| Fixed 0.85 same-score persistence | 0.669542 |

Patient-paired 400-bootstrap WM minus comparator mean NLL intervals:

| Comparator | WM - comparator | 95% bootstrap interval |
|---|---:|---|
| Fixed persistence | -0.256243 | [-0.269142,-0.242533] |
| TRAIN item marginal | -0.035386 | [-0.041978,-0.028154] |
| **Tuned item Markov** | **+0.002718** | **[-0.002875,+0.008620]** |

The neural WM convincingly beats static marginals in this
*exploratory*, repeatedly examined TRAIN cohort. It does
**not show a reliable gain** over tuned Markov; a confidence
interval crossing zero should not be called significant.
Markov's NLL advantage is just 0.002718 (about 0.66% of
Markov NLL). Inference uses the **actual recorded target
visit month**, so these are retrospective time-conditioned
next-visit score forecasts, not prospective appointment forecasting.

## Markov hyperparameter selections

Every outer fold chose **pooled** item-wise transitions:
- fold 0: alpha=2, pooled;
- fold 1: alpha=10, pooled;
- fold 2: alpha=10, pooled;
- fold 3: alpha=10, pooled;
- fold 4: alpha=2, pooled.

No fold selected the three time-gap bins. This result
reflects inner-development selection for these data and
candidate models; it does not establish that disease
time dynamics do not matter in general.

Neural model outer-fit sizes were 526–527; inner
fit for epoch selection was 447–448, inner dev 79.
Chosen epochs 25,10,10,5,25; two reached the
maximum available duration, a training-sensitivity
concern requiring examination but not license for
uncontrolled hyperparameter tuning.

## What is and is not learned

The learned neural recurrent state can condition on other
measured Motor items and past visits. But the observed
gain over static marginals can also be achieved by a
**much simpler per-item conditional transition model**.
Current metrics do not identify unique value attributable
to latent, cross-item or long-horizon representations.

It would be scientifically misleading to characterize
this model as a proven Dreamer-4-like patient simulator
or to claim WM improves Qwen Doctor. No RSI policy
update, multi-step open-loop test, clinical treatment
intervention, independent heldout test or real patient
action is evaluated by this report.

## Decision: do NOT scale the model or plug it into Doctor blindly

Keep `tuned_item_markov` as a compulsory
strong transition baseline for all later World Model
claims. Next experiments should have *different
capability targets*, not merely lower pooled one-step NLL:

1. **Cross-item information transfer:** when an item
   has *never* been observed for a given person, can
   other observed Motor items predict its next actual
   value better than fit-TRAIN priors? Stratify by
   fully observed vs structurally unrateable inputs.
2. **Real R01 -> later V04:** forecast actually
   measurable future onsite six-item burden and
   score distributions from grounded remote Motor
   observations. Reconcile Phase53 197 paired
   people with the earlier 199 frozen training pair
   definition *on cluster*; never silently mix cohorts.
3. **True multi-step open-loop rollout:** start
   from BL or an early visit, advance without reading
   intermediate true future observations, and compare
   per-horizon uncertainty and future measured items
   with correctly rolled-forward Markov baselines.
4. **Uncertainty and calibration:** report patient-paired
   categorical NLL, multiclass Brier, calibration
   curves/coverage and per-horizon performance with
   small-cell suppression. Do not treat transitions
   as independent patients.
5. **Tool-choice relevance:** only if a state model
   demonstrates real added information, introduce
   Doctor-controlled observation actions, which
   update beliefs but do not directly change disease.
   Add strict verifier-guided RSI with versioning and
   rollback, inner TRAIN-only acceptance, and
   equal-tool/LLM-token budget baselines.

All evaluations are further exploratory use of
previously repeatedly analysed TRAIN, and need a new
genuinely independent cohort for publication-level
confirmation. Do not use the consumed original TEST.
