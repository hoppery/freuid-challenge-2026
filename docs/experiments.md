# FREUID Experiment Plan & Log

Primary selection metric: **APCER@1%BPCER** (lower better) on a **domain-holdout** validation split (hold out whole `doc_type`s → mimics the public≠private private-test shift). Secondary: AuDET (=1−AUC), ROC-AUC, and per-group breakdowns. Submit raw attack probabilities (Kaggle computes the official AuDET).

Why domain-holdout, not random split: SOTA (docs/sota-research.md) shows in-domain fit is easy and **generalization decides ranking**. A random-split number will be optimistic; the doc_type-holdout number is the honest proxy for the private leaderboard.

## Stage 0 — Baseline (run first, after data download)
- `configs/baseline.yaml`: ConvNeXt V2 Tiny, pretrained, light aug, 384px, 10 ep.
- Establishes the honest reference APCER@1%BPCER / AuDET. Confirm inference + submission format end-to-end. This is "the baseline" per the directive.

## ⚡ READY-TO-LAUNCH (migrated to RTX PRO 6000 Black ×2, 98 GB VRAM each)
Verify CUDA, then launch the next wave (2 GPUs, queued):
```bash
cd ~/ijcai_freuid_chanllenge && unset PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python3 -c "import torch; torch.zeros(1).cuda(); print('OK')"
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/run_sweep.py \
  --configs exp_heavyaug exp_dinov2 exp_hpf exp_convnext_base baseline_indomain --gpus 0 1
```
Aux data ready: `manifests/fantasyid.parquet` (3,284 imgs: 2,351 GenAI face/text forgeries + 933 bona-fide; 13 country doc-types incl. Arabic/Persian) — for a combined-data experiment in a later wave.

## Stage 1 — Known-technique improvements (총동원), ablate one lever at a time
| Exp | Config | Hypothesis (lever from SOTA) |
|---|---|---|
| heavy-aug | `exp_heavyaug.yaml` | print-scan/moiré/JPEG-recompression aug on BOTH classes → cross-domain ↑ (the #1 cheap generalization win) |
| freq dual | `exp_freq.yaml` | RGB + FFT spectral branch → captures analog-hole moiré/halftone cue |
| freq+heavy | `exp_freq_heavy.yaml` | combine the two top levers |
| DINOv2 | `exp_dinov2.yaml` | DINOv2 foundation backbone generalizes best (IDNet EER 27.86→8.25 precedent); + heavy aug |

Keep epochs/seed/holdout fixed; change ONE factor per run. Track every run's val APCER@1%BPCER + per-doc_type breakdown in the table below.

## Stage 2 — DTC implementation status (2026-06-13: IMPLEMENTED, smoke-verified)
The #1 novel edge is now code (design: docs/winning-strategy.md §3):
- `src/freuid/data/tracemix.py` — TraceMix: self-supervised consistency labels.
  Coherent (y=1): one recapture chain uniformly (or clean). Incoherent (y=0): chain A
  global + contrasting chain B (forced ΔJPEG-QF ≥15, Δmoiré-freq >0.1) inside feathered
  region(s) of 10–40% area. Class-agnostic (applied to both fraud & bona-fide).
- `src/freuid/models/dtc.py` — DTCNet: RGB backbone + HPF trace branch kept spatial →
  8×8 patch embeddings → pairwise-cosine stats (7 dims) → consistency head (aux BCE) +
  3-way fusion [RGB; global trace; stats] → fraud logit. `model_type: dtc`.
- Train loop: `L = BCE_fraud + dtc_lambda·BCE_cons`; eval path unchanged (forward
  returns fraud logit only → infer.py/ensemble compatible).
- Tests: tests/test_dtc.py (9 tests; 31 total passing). Smoke: configs/smoke_dtc.yaml
  ran end-to-end on GPU (1485-img subset, dual loss converging, ckpt saved).
- Ready to run: `configs/exp_dtc_partial.yaml` + ablation `exp_dtc_noaux_partial.yaml`
  (dtc_lambda=0 isolates aux-loss contribution vs architecture alone).

## Stage 2 — Novel / original edges (the winning differentiators)
Ranked by expected payoff (full rationale in docs/sota-research.md §"5 NOVEL edges"):
1. **Analog-hole double-trace consistency** — detect intra-document recapture-trace inconsistency (a digital edit printed+recaptured leaves a layered, locally-inconsistent trace). Turns the analog hole into a signature.
2. **Script-aware glyph-rendering forensics** — OCR-crop text fields, per-glyph stroke/baseline/kerning consistency branch; targets Arabic + <2%-area text inpainting (the documented hard failure mode).
3. **Self-supervised recapture-trace pretraining (FMAG + CLIP-Flow on FREUID)** — masked-frequency MAE on the doc corpus → anomaly head generalizes to unseen private attacks.
4. **Capture-device adversarial disentanglement** — gradient-reversal branch predicting device/substrate → device-invariant fraud features (directly optimizes public≠private).
5. **Physics-guided moiré-phase synthesis + moiré-phase auxiliary** — model the generative physics of recapture, not a memorized texture.

## Stage 3 — Ensembling + calibration
Ensemble complementary inductive biases (freq-heavy + RGB-semantic + recapture-trace); pick the blend by domain-holdout APCER@1%BPCER. Add aux datasets (IDNet/FantasyID/DLC-2021/MIDV-Holo/SIDTD/DocXPand) as pretraining/extra-fraud once their adapters are written.

## Failure protocol
If a lever does NOT improve domain-holdout APCER@1%BPCER: brainstorm root cause (overfit to public domain? augmentation destroying the cue? metric-threshold artifact? class imbalance? doc_type leakage?), form a falsifiable hypothesis, design the next single-factor experiment, run, repeat.

## New capabilities (2026-06-12, RTX PRO 6000 migration)
- `freeze_backbone: true` in any config → `FrozenBackboneClassifier` (frozen timm backbone + LayerNorm+MLP adapter). Enables proper DINOv2-frozen experiments (biggest generalization lever per SOTA).
- `infer.py` now supports `--ckpts A.pt B.pt C.pt` ensemble (score-average) + `--val-manifest` for APCER@1%BPCER calibration readout.
- `scripts/refresh_partial_manifest.py` — updates `manifests/freuid_partial.parquet` as data arrives; run manually during copy.

## Partial-data wave (data copy in progress, ~46% present as of 2026-06-12)
Running on `manifests/freuid_partial.parquet` (~32k imgs, all 5 doc_types). Results will be directionally valid; re-run on full data when copy completes.
| config | GPU | status |
|---|---|---|
| exp_heavyaug_partial | GPU0 | **running** |
| exp_hpf_partial | GPU1 | **running** |

Next (after current runs finish, still during copy):
- exp_dinov2_frozen_partial (GPU0) — frozen DINOv2 + adapter, the key generalization experiment
- exp_dinov2_partial (GPU1) — full fine-tune DINOv2 baseline for comparison

## Results log
Validation = leave-one-doc-type-out (held out MOZAMBIQUE/DL, 13,365 imgs; train on 55,987).
Full-data runs marked with `[full]`; partial-data runs marked with `[partial ~32k]`.
| date | exp | backbone | aug | val APCER@1%BPCER | AuDET | ROC-AUC | notes |
|---|---|---|---|---|---|---|---|
| 2026-06-11 | **baseline** [full] | convnextv2_tiny | light | **0.501** | **0.257** | **0.743** | VERIFIED plausible (beats zero-shot forensic SOTA 0.52–0.62; brackets DeepID OOD winners 0.70–0.73). docs/baseline-verification.md |
| 2026-06-11 | exp_freq [full] | convnextv2_tiny (freq dual) | light | 0.988 | 0.478 | 0.522 | FAIL — global FFT learns doc-type shortcut, fails cross-domain. Fix: local high-freq residual (hpf_dual) |
| 2026-06-12 | exp_heavyaug [partial] | convnextv2_tiny | heavy | — | — | — | running |
| 2026-06-12 | exp_hpf [partial] | convnextv2_tiny (hpf_dual) | heavy | — | — | — | running |
