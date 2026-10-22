# 2026 World Model architecture pivot: Dreamer-style + clinical JEPA

**Date:** 2026-10-20. **Status:** prospective design, not a verified
clinical model. The established Phase23–44 baseline and consumed TEST
remain fixed. This replaces "just tune Phase49 masked Transformer"
as the *method research direction*, but not as a validated winner.

## Verified 2025–2026 literature

| Work | Algorithm that matters | Applicability |
|---|---|---|
| [Dreamer 4, Hafner et al. 2025](https://arxiv.org/abs/2509.24527) | Fast transformer world model, action conditioning, imagined experience for model-based control. Community [PyTorch reproduction, 2026](https://github.com/nicklashansen/dreamer4), not official/full original reproduction. | **Algorithmic inspiration**, not directly reusable image/Minecraft weights |
| [DreamX-World, June 2026](https://github.com/AMAP-ML/DreamX-World) | 5B video-based interactive simulation and event prompts. | Not a pretrained structured PPMI patient model; unsuitable as a direct base |
| [SMB-Structure, Jan 2026](https://arxiv.org/abs/2601.22128) | Joint embedding predictive architecture (JEPA) + token reconstruction learns predictive EHR patient embeddings. [1.7B Qwen-based weights](https://huggingface.co/standardmodelbio/SMB-v1_Qwen3-1.7b_multi-objective) exist, but model is a *patient encoder*, not generative physical simulator. | Clinical latent-pretraining reference; model card is **gated**, **CC-BY-NC-4.0**, requires MEDS format, and trained in oncology. Do not silently download or assume transfer |
| [EHRWorld, Feb 2026](https://arxiv.org/abs/2602.03569) | Long-horizon sequential patient world simulation with explicit intervention history. | Clinical dynamics objective; paper-scale resources are not present in our Prodromal sequences |
| [ChronoMedicalWorld, May 2026](https://arxiv.org/abs/2605.21963) | Joint embedding state, explicit time, recurrent dynamics, closed-loop prefix rollout, multimodal clinical actions. | **Most applicable architectural template** for irregular chronic-PD followup |
| [SepsisAgent, May 2026](https://arxiv.org/abs/2605.14723) | LLM proposes treatment -> clinical WM simulates -> agent refines; SFT/behavior cloning/agent RL. [Public code](https://github.com/FreedomIntelligence/SepsisAgent). | The right *LLM/WM agent-control* comparator, but its intervention semantics and weights concern sepsis/MIMIC, not Parkinson |

## New scientific thesis

**Dreamer-inspired imagination + clinical JEPA + observation-grounded RSI.**

A latent patient's *recorded-state belief* must be updated using
actual measurements, not fabricated LLM output. There are two different
action types:

1. **Measurement/query**: changes the agent's knowledge, not the
   patient's physical disease. Observation kernel
   `p(o_t^a | z_t, modality, mask, a)`; belief update
   `q(z_t | o_{<=t}, m_{<=t}, a_{<t})`.
2. **Therapeutic intervention**: may actually modify future disease
   state; requires recorded, temporally aligned treatment actions.
   A state-transition
   `p(z_{t+Δ} | z_t, Δ, treatment_history)` is not a causal simulator
   merely because retrospective treatment features were present.

The long-term method therefore separates *epistemic* measurement
actions from *physiological* intervention actions. Dreamer-style
imagination may guide safe **selection of recorded follow-up
observations**, but its imagined trajectories remain hypotheses.
An external verifier evaluates only **real measured heldout facts**.

## What we can run right now

A lightweight, **JEPA/RSSM-inspired longitudinal control experiment**,
`scripts/phase50_clinical_latent_wm.py`, uses the Phase29 frozen
Prodromal TRAIN trajectories that have BL, 12m, 24m Motor totals (~332).
It has a latent observation encoder, continuous month-gap transition,
measured-observation belief update, a score distribution head,
teacher-latent matching, and BL->nominal 12->24 **unobserved rollout**.

It forecasts nominal 24m Motor from actual BL+12m history. It must
compare with exact historical baselines:
- last observed (12m) value;
- baseline alone;
- fixed 50/50 averaging BL and 12m (very strong Phase30 baseline);
- linear extrapolation.

Three patient-level outer folds, each with independent TRAIN-only
inner validation for stopping, produce paired outer-fold predictions,
MAE/RMSE, and a patient-bootstrap MAE difference vs fixed 50/50.
No future actual gap enters predictors. No original VAL or consumed
TEST records are accessed, and no individual patient data exported.

**Crucial limitation:** this pilot has just two transitions from
about 332 patients, with no consistently logged treatment-action
sequence and no high-dimensional longitudinal state. The model is
not a full Dreamer 4 architecture, not a pretrained SMB model, not
a calibrated counterfactual simulator and not a clinical trial.
An advanced WM cannot be declared effective solely from this stage.

## Research road map beyond the pilot

1. **Data foundation:** revisit the private PPMI full visit/event
   catalog to build masked multi-measurement trajectories for more
   patients and many visits; release *only* source metadata and
   derived aggregate coverage. Harmonize BioFIND / Fox Insight only
   after verifying visit and measure semantics.
2. **World model:** scale latent temporal dynamics (RSSM or
   transformer SSM), temporal JEPA, probabilistic observation
   decoder, calibration and rollback-able imagination.
3. **Agent:** Qwen3-4B Doctor uses tools to request *actual
   available* observations; compare true q-matched greedy and
   non-LLM policies.
4. **Verifier-guided RSI:** versioned candidate WM+policy changes,
   independent evaluation on exclusively TRAIN-development
   partitions, acceptance/rollback, audits of simulator exploitation;
   separate gains from compute budget and LLM tokens.
5. **Independent evaluation:** original Phase44 TEST is already
   consumed. Require genuinely new external data for new final claims.

## Hardware and networking

Prototype is **CPU-first** using working
`/scratch/ylong66/conda/envs/pks_qwen46_cu118`.
No model download; no GPU allocation necessary. All patient data,
Conda packages and model weights remain in prescribed local paths.
A future SMB/Qwen pretraining evaluation would require download
**only from a networked login node**, model terms acceptance, and
local offline loading in GPU Slurm.

Do not claim we used the actual Dreamer 4 or SMB weights when
we only implemented architecture-level ideas.
