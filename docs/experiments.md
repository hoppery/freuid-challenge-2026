# FREUID Experiment Plan & Log

Primary selection metric: **APCER@1%BPCER** (lower better) on a **domain-holdout** validation split (hold out whole `doc_type`s → mimics the public≠private private-test shift). Secondary: AuDET (=1−AUC), ROC-AUC, and per-group breakdowns. Submit raw attack probabilities (Kaggle computes the official AuDET).

Why domain-holdout, not random split: SOTA (docs/sota-research.md) shows in-domain fit is easy and **generalization decides ranking**. A random-split number will be optimistic; the doc_type-holdout number is the honest proxy for the private leaderboard.

## Stage 0 — Baseline (run first, after data download)
- `configs/baseline.yaml`: ConvNeXt V2 Tiny, pretrained, light aug, 384px, 10 ep.
- Establishes the honest reference APCER@1%BPCER / AuDET. Confirm inference + submission format end-to-end. This is "the baseline" per the directive.

## Stage 1 — Known-technique improvements (총동원), ablate one lever at a time
| Exp | Config | Hypothesis (lever from SOTA) |
|---|---|---|
| heavy-aug | `exp_heavyaug.yaml` | print-scan/moiré/JPEG-recompression aug on BOTH classes → cross-domain ↑ (the #1 cheap generalization win) |
| freq dual | `exp_freq.yaml` | RGB + FFT spectral branch → captures analog-hole moiré/halftone cue |
| freq+heavy | `exp_freq_heavy.yaml` | combine the two top levers |
| DINOv2 | `exp_dinov2.yaml` | DINOv2 foundation backbone generalizes best (IDNet EER 27.86→8.25 precedent); + heavy aug |

Keep epochs/seed/holdout fixed; change ONE factor per run. Track every run's val APCER@1%BPCER + per-doc_type breakdown in the table below.

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

## Results log
Validation = leave-one-doc-type-out (held out MOZAMBIQUE/DL, 13,365 imgs; train on 55,987).
| date | exp | backbone | aug | val APCER@1%BPCER | AuDET | ROC-AUC | notes |
|---|---|---|---|---|---|---|---|
| 2026-06-11 | **baseline** | convnextv2_tiny | light | **0.501** | **0.257** | **0.743** | VERIFIED plausible (beats zero-shot forensic SOTA 0.52–0.62; brackets DeepID OOD winners 0.70–0.73). docs/baseline-verification.md |
| 2026-06-11 | exp_freq | convnextv2_tiny (freq dual) | light | 0.988 | 0.478 | 0.522 | FAIL — global FFT learns doc-type shortcut, fails cross-domain. Fix: local high-freq residual, not global FFT |
| _GPU2 fault_ | exp_heavyaug | convnextv2_tiny | heavy | — | — | — | crashed (GPU2 hw fault); re-run pending |
| _GPU2 fault_ | exp_dinov2 | dinov2_vitb14 | heavy | — | — | — | crashed (GPU2 hw fault); re-run pending |
