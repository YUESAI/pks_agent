# Phase80 full reports and Phase81 out-of-fold value-based active measurement

phase80_reports.zip: all twenty jobs (IDs 335896–335915)
completed all 3 clinical tasks. 210 method rows, 900 paired rows,
15 independent DEV/AUDIT gate decisions. Five Qwen-only arms
run 4.1 hours each; total 2,449 real CPU Qwen prompts, zero invalid
requested clinical test names.

Deduplicate repeated per-fold references. Patient-weighted metrics:
| Model | Prodromal12 N584 median MAE/RPS | Prodromal24 N378 median MAE/RPS |
|---|---|---|
| Motor no query | .75514/.024486 | .87037/.027573 |
| FIT-priority measured test | .75685/.024324 | **.84656/.027472** |
| FIT expected KL test | .76370/.024815 | .84921/.027782 |
| Real Qwen chosen test | .76370/.024708 | **.84656/.027739** |
| Qwen+audited likelihood alpha | .76199/.024549 | .84656/.027739 |

At Prodromal12 WM gain policy had higher RPS than fixed in
all five outer folds. PD12 no coherent stable gain.
Verifier accepted only 2/15 decisions (Prodromal12 fold4 and
PD12 fold1), both alpha=0, removing evidence updates.

**Interpretation**: Clinically recorded optional tests can sometimes
contribute signal, but active *selection* with unconditional expected
information gain failed to beat a fixed FIT-priority strategy.
The naive channel P(BL test | future severity stratum) may be
conditionally redundant with Motor prior and not calibrated.
A verifier choosing alpha=0 is protection from noisy evidence,
not proof of an autonomous RSI.

Phase81 focuses on information-acquisition utility and the option
to **ABSTAIN**. Phase80 treated every available clinical test as
worth requesting. Real measured baseline optional values are only
released after the action; no patient-level records exported.

## Four protocols, five patient folds = 20 independent CPU jobs

0. Motor q0, fixed q1, WM-information gain q1, deterministic random
   q1, all same person, same FIT-trained distribution.
1. Same no-Qwen reference policies (duplicated by design for
   in-job paired metrics).
2. Add FIT-only **three-way inner out-of-fold** training:
   - Inner WM and smoothed test emission fitted only on other FIT people.
   - Inner evaluation per subject uses actual recorded BL test
     after a hypothetical choice and future endpoint solely for
     training value-of-information estimates.
   - Learn test-specific and Motor-risk-band shrinkage estimates of
     *expected RPS improvement* with uncertainty penalty.
   - Choose the best positive-value test or ABSTAIN.
3. Real offline Qwen Doctor gets available test names, FIT-OOF
   expected gains and explicit ABSTAIN option. It never receives
   patient test values before selection or future outcome.
   DEV and independent AUDIT compare exactly one candidate switch
   (use Qwen instead of data-driven OOF policy); rollback on
   failure of prespecified RPS/MAE/RMSE/Brier guardrails.

Caveats: The technique estimates values of the *same retrospective
BL tests*, with available test names already known. This is NOT
prospective order-set design. Benefit estimated by inner 3-way OOF
FIT only; the global evidence channel for outer EVAL remains trained
on FIT. Calibrated incremental conditional value is an approximation,
not a patient counterfactual/causal acquisition model.

The OOF confidence penalization is heuristic; further independent
new-patient evaluation would be necessary. Actual Qwen can take hours
per arm. Model training uses one CPU thread and disabled MKLDNN,
avoiding the original 12h PyTorch backward stall. No GPU,
no array, 8 cores, 48GB, per-job wall limit 12h.
Task-level output persistence saves completed tasks if Qwen arm
times out.

**20 ordinary CPU jobs, no smoke jobs or new branches**:
```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
 sbatch --export=ALL,PHASE81_ARM_ID="$arm" slurm/phase81_dusky_cpu.sbatch
done
```

Collect `/scratch/ylong66/pks_agent_outputs/phase81_oof_*/` with
method_metrics.csv, paired_bootstrap.csv, coverage.csv,
audit.json, progress.json, summary.json (only fully finished).
