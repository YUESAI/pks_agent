# Phase68 actual results → Phase69 temporal WM and real Qwen verifier

Phase68 archive uploaded 2026-10-26: 25 complete report directories
for 20 unique arm IDs. Five greedy arm IDs (0,4,8,12,16) each have
an earlier duplicate. For the official aggregation we selected the
latest Job ID per arm. Do not count older duplicates as new patients.

Five-fold patient-weighted metrics:
| Cohort/time | Greedy MAE | Plain Qwen MAE | Qwen+WM MAE | Qwen+verified MAE |
|---|---:|---:|---:|---:|
| Prodromal12 n584 | 1.068473 | 1.006711 | 1.068057 | 1.003672 |
| Prodromal24 n378 | 1.216101 | 1.138008 | 1.209433 | 1.179944 |
| PD12 n156 | 2.560040 | 2.641306 | 2.554697 | 2.556999 |

Prodromal12 Brier greedy .214850, plain .216779, WM .214083,
verified .214480. Prodromal24 Brier .233332, .230198, .234927,
.235952 respectively.
Prodromal12 invalid actions: plain 416/2336, WM 50/2336, verified 16/2336.
Prodromal24: plain 246/1512, WM 14/1512, verified 10/1512.
Exactly 4/15 verifier proposals accepted (11 rejected).
Qwen job runtimes were 4.7-11.5h per report. Several jobs approached
the 12h Dusky cap. Slurm exit code not supplied.
Five repeated greedy runs had mildly different numerical outputs,
so variance/reproducibility also requires audit.

**Result**: Phase68 full clinical-feature observation head did not
produce stable superiority over Plain Qwen. RSI still validated
a greedy proxy, not actual Qwen behavior.

## Phase69 methods

1. Train FIT-only (including baseline unlabeled FIT subjects) neural
   masked multimodal observation and future-burden heads.
2. ADD time-gap-conditioned MDS-UPDRS Motor transition head, supervised
   only on actual consecutive FIT-person visits with matching
   medication state, using a 1..36 month gap, and FIT-only supervised
   burden replay. No future visits from EVAL/TUNE/AUDIT persons.
3. Evaluate the same five outer patient folds (same Phase66 original
   hash) x four policies = 20 standalone jobs. Greedy / plain Qwen /
   Qwen+WM / Qwen+WM+revised config.
4. One independent AUDIT of a DEV-selected style/temperature revision,
   using ACTUAL Qwen doctor calls on 8 DEV + 8 AUDIT subjects per
   horizon. This is constrained RSI-lite, not a self-modifying
   neural weight or code-evolution agent.
5. TWO actual questions per person (down from four Phase68) to keep
   15 Qwen CPU tasks feasible within 12-hour time cap. Thus Phase68
   comparison is not compute/information-budget matched; compare
   Phase69 arms to each other under identical q=2.

Caveat: The trained temporal transition head currently provides
a learned *diagnostic* probability API, but Qwen action values are
from the shared masked feature observation/outcome heads, and future
temporal head outputs are not yet fused into decisions. It is a
representation-training ablation, NOT demonstrated temporal
planning or a causal dynamic WM. Only internal previously explored
TRAIN patients; independent external test still required.

## Submit exactly 20 non-array CPU jobs

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
  sbatch --export=ALL,PHASE69_ARM_ID="$arm" slurm/phase69_dusky_cpu.sbatch
done
```

Each requests 8 CPU, 48 GB, 12 hours, no GPU. Collect complete
aggregated reports including fold_metrics.csv, cohort_coverage.csv,
verifier_trace.json and summary.json. Do not export patient-level
clinical records, patient IDs or individual trajectories.
