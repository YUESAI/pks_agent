# Phase88 results and Phase89 independent-expert OOF gating

Phase88 user ZIP: phase88_reports.zip, 4 successful jobs
337385..337388, each 3 tasks. The three clinical tasks share
the same patient-global 80% FIT / 20% repeatedly-developed DEV
split and SHA-256 cohort membership fingerprints.

| Method | Prodromal12 N114 RPS | Prodromal24 N71 RPS | PD12 N37 RPS |
|---|---:|---:|---:|
| Phase85 static shared mixture | .023103038 | **.022843899** | .067274773 |
| Phase87 observed-visit flow (.15) | .022642465 | .023180876 | **.063139069** |
| Phase88 horizon gate | **.022638520** | .023219169 | .069825921 |
| Phase88 patient gate | .022644076 | .023246176 | .069785941 |

Phase88 gates have tiny within-task variation:
- horizon gate 12m mean .16979, std .00138;
  24m mean .16255, std .00149; PD mean .20070, std .00007.
- patient gate 12m mean .10658, std .01212;
  24m mean .08741, std .01024; PD mean .19132, std .00432.
Thus they did not learn distinct patient-level dynamic use,
and overall lost the 24m static and PD12 dynamic advantages.
Even gate=0 in Phase88 does NOT reproduce Phase85: experts
share a GRU, decoder, and training and are NOT independent.

Phase89 tests whether the missing factor is **independently
trained specialists plus leakage-controlled gate learning**.
Within the SAME global 80/20 development split:
- FIT-only 2fold patient-global OOF for expert predictions.
  Static Phase85 and visit-supervised Phase87 experts are
  retrained per inner fold only on other FIT people, with
  adjacent-visit supervision likewise restricted to those
  persons. The gating network is trained only on held inner
  predictions and true FIT outcomes, never in-sample predictions.
- Refit two independent full-FIT experts after gate learning;
  use trained gate to combine their 0..24 distributions.
- Optimize directly the OOF proper ranked probability score
  RPS with patient-normalized weights (avoid double counting
  12m+24m observations from the same person). Small fixed
  L2/initial-weight regularization is specified in code.
- Compare a horizon-only linear gate to a patient-state + horizon
  + expert uncertainty/disagreement linear gate. Both gates
  are supervised by the same OOF predictions, and neither
  sees outer DEV labels. These are statistical decision
  networks, not verified biological disease subtype classifiers.

Four ordinary jobs (NOT a fivefold outer benchmark):
0 Phase85 full FIT static expert.
1 Phase87 full FIT transition-supervised flow expert.
2 FIT-OOF horizon-only gate with independently refitted experts.
3 FIT-OOF patient-aware gate with independently refitted experts.

Caveat: Inner OOF experts train on ~half the FIT patients, while
final experts use all FIT. Gate could therefore suffer an
out-of-fold / refit prediction distribution shift, especially
at the tiny PD FIT size. A small gate does not guarantee
better generalization. Both DEV and all prior PPMI data have
already been repeatedly examined. Do not claim confirmatory
significance or touch the consumed final TEST.

Outputs (no patient records or scores): dev_metrics.csv,
coverage.csv, oof_audit.json, progress.json and summary.json.
The OOF audit reports only sizes and aggregate proper scores.

## Run tests then four ordinary CPU jobs

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python \
 -m unittest discover -s tests -p test_phase89_wm.py -v
for arm in 0 1 2 3; do
 sbatch --export=ALL,PHASE89_ARM_ID="$arm" \
   slurm/phase89_dusky_cpu.sbatch
done
```

8 CPU cores allocated, but only 1 for PyTorch/BLAS, MKLDNN
disabled (Phase77 backward bug). 48GB RAM, 12h maximum.
No Slurm arrays, no GPU and no new branch.
