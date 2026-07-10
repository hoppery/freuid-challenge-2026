# SOTA Research — Winning FREUID 2026 (strategy backbone)

Task: bona-fide(0)/fraud(1) ID-doc classification across physical / GenAI-digital / **print-and-capture** attacks, African/Asian docs (Latin+Arabic). Metric **AuDET + APCER@1%BPCER**. Core difficulty: public-val ≠ private test (cross-domain). Closest external prior: **DeepID/FantasyID (ICCV-2025)** — same threat model, **winning aggregate F1 only 0.801** → generalization, not in-domain fit, decides ranking.

## Benchmarks
| Dataset | Composition | Attack focus | Relevance |
|---|---|---|---|
| MIDV-2020 / MIDV-Holo | 72,409 imgs; hologram open-set | holograms, static+dynamic PA | hologram cues absent for print-recapture |
| DLC-2021 | 1,424 clips, 10 EU types | lamination, screen/print | recapture-adjacent |
| SIDTD (Sci Data 2024) | synthetic ID/travel; Crop&Replace+inpaint | composite digital forgery | closest forgery taxonomy |
| IDNet (2024) | 837,060 imgs ~490GB, 20 types, 6 fraud variants | privacy-safe synthetic fraud | largest pretraining corpus |
| FantasyID/DeepID | 262 cards; 786 bona-fide printed+captured + 1,572 manipulated | **print-capture bona-fide + GenAI face-swap/text-inpaint** | **direct analog of FREUID** |
| KID34K | 34,662 Korean IDs, PVC bona-fide | print/screen/PVC | recapture artifacts |

## Architectures & numbers
- **DINOv2 foundation backbones dominate.** Printed 1.28% EER, screen 2.73% EER fine-tuned; **FakeIDet (patch DINOv2) 0% EER internal, holds on unseen DLC-2021**; **DINOv2 fusion cut IDNet EER 27.86→8.25** (clearest "foundation models generalize" datapoint).
- **Off-the-shelf forensic SOTA collapses cross-domain:** TruFor/MMFusion/UniFD/FatFormer ≈ **FNR 50% @ FPR 10%** on FantasyID.
- **TwoHead-SwinFPN** (Swin-L frozen + FPN + UNet decoder + attention, joint detect+localize): det **84.31% acc / 90.78% AUC / F1 88.61%**; loc 57.24% Dice. Failures: **subtle text inpainting <2% area** and **perfectly-blended face swaps (31% of FN)**.
- PAD-ID-Card IJCB: best Track-1 EER 11.34%, Track-2 6.36%; ResNeXt101 6.92% APCER/8.51% BPCER@5%BPCER. Screen-replay hardest.
- IML-ViT (high-res + multiscale SFPN + edge supervision + MAE) tops localization (CASIAv1 0.798, Columbia 0.945) — **architectural lessons transfer; noise-residual feature does NOT survive print-recapture.**

## Forensic features vs the analog hole
Print→rephotograph **destroys digital noise (PRNU/noiseprint/SRM)** and **re-injects recapture trace (halftone/moiré/blur/screen-grid)**.
| Feature | Transfers to docs | Survives recapture | Note |
|---|---|---|---|
| PRNU/Noiseprint/SRM | partial | **No (destroyed)** | basis of CAT-Net/TruFor; dies in analog hole |
| **DCT/FFT/radial spectrum** | **Yes** | **Yes (recapture ADDS peaks)** | **strongest single analog-hole cue** (moiré=periodic FFT peaks) |
| ELA | weak | no | low value |
| Localization nets (CAT-Net/TruFor/MMFusion/ManTra/PSCC/IML-ViT) | yes for *digital* edits | degrade on recapture | use for GenAI-edit axis only |
- **Best-matched method:** Multi-Modal Doc PAD w/ Forensics (arXiv:2404.06663) — **self-supervised recapture-trace disentanglement** (blur-content C / texture T), synthesize+transfer recapture traces → **+7.97% avg AUC/EER** over RGB-only.
- **ForgeNet:** attacker counter (noise compensation + inverse halftoning) defeats noise-based PAD → expect adaptive attacks in private test; don't rely on raw residuals.

## GenAI/diffusion-edit detection (localized)
- FatFormer (CVPR24): CLIP + freq adapter, 98%/95% to unseen GAN/diffusion whole-image — but **fails on tiny local doc edits**.
- Localized benchmarks (COCO-Inpaint, DiffSeg30k, X-Edit) confirm whole-image detectors degrade on small edits → need **patch/pixel supervision**.
- Param-efficient CLIP (C2P-CLIP, LNCLIP-DF tune LayerNorm only 0.03% params) generalize with minimal overfit.

## Domain generalization levers (ranked)
1. **Foundation backbone (DINOv2/CLIP)** — biggest lever.
2. **Frequency-domain learning** (Frequency Masking 2401.06506; synthetic-frequency-injection 2403.13479) — more domain-stable than RGB, aligns with recapture.
3. **Reconstruction/anomaly** (CLIP-Flow 2508.09477) — flags unseen attack types in private test.
4. **Parameter-efficient adaptation** (LayerNorm-only) — preserve pretrained manifold; avoid full fine-tune.
5. **Few-shot/prototypical per doc-geography** (Spain/Chile→Argentina precedent) → maps to Egypt/Guinea/Benin/Mozambique/Mauritius.

## Augmentation recipe (apply to BOTH classes, heavily)
print-halftone → **moiré spectral injection (FMAG)** → Gaussian/motion blur → perspective+lighting → sensor noise → **JPEG recompression chain (2–3 stages QF 60–95)** → downscale/upscale.
- Print-and-Scan aug (ACM IH&MMSec 2024) boosts recapture robustness; recompression *chains* (not single QF) help (Comprint 2211.14079).
- **Critical:** apply recapture/print aug to fraud AND bona-fide so model can't shortcut "recapture⇒fraud".

## Ranked plan
**Tier 1:** (1) DINOv2/CLIP frozen + forgery-aware adapter + dual classify/localize head; (2) two-pathway input RGB + spectral(FFT/DCT) + disentangled recapture-trace; (3) aggressive print-scan/moiré/recompression aug on both classes.
**Tier 2:** (4) anomaly/reconstruction aux (CLIP-Flow) for unseen attacks; (5) few-shot per doc-geography/script; (6) ensemble across freq-heavy + RGB-semantic + recapture-trace, **threshold calibrated to APCER@1%BPCER** on a private-mimicking fold.
**Anti-patterns:** raw PRNU as primary; full backbone fine-tune; recapture-aug on fraud only; optimizing val F1 not APCER@1%BPCER.

## 5 NOVEL edges (beyond method-stacking)
1. **Analog-hole "double-trace" consistency detector.** Genuine capture = one coherent acquisition trace; print-captured forgery = layered history (digital edit residual → halftone print → re-photograph). Detect **intra-document recapture-trace inconsistency/multiplicity** (local blur/halftone/moiré-phase regions inconsistent with the rest) → turns the analog hole into a signature. Unpublished framing.
2. **Script-conditioned glyph-rendering forensics.** GenAI-inpainted text (esp. Arabic cursive+diacritics) leaves per-glyph anomalies (stroke-width spectra, baseline jitter, impossible kerning). OCR-crop text fields → glyph-statistics consistency branch. Targets the <2%-area text-inpaint failure mode + multi-script edge.
3. **Self-supervised recapture-trace pretraining (FMAG + CLIP-Flow on FREUID itself).** MAE/masked-frequency pretrain on doc corpus → head detects deviations from the substrate frequency manifold → anomaly framing generalizes to unseen private attacks.
4. **Capture-device adversarial disentanglement.** Public/private gap ≈ capture-device+substrate shift. Adversarial branch predicts device/substrate + gradient reversal → device-invariant fraud features. Directly optimizes the public≠private objective.
5. **Physics-guided moiré-phase synthesis as hard positives.** Simulate camera-sensor×screen/print-grid interference physics (correct phase/frequency) + moiré-phase-prediction auxiliary → model the generative physics of recapture, not a memorized texture.

## Bottom line
Ceiling is low (~0.80 F1); generalization decides. Win via frozen DINOv2/CLIP + adapter + dual head; frequency + recapture-trace pathways (not raw noise); heavy print-scan/moiré/recompression aug on both classes; anomaly + few-shot DG; calibrate to APCER@1%BPCER. Differentiating edge = **intra-document recapture-trace consistency** + **script-aware glyph forensics**.

Key refs: FantasyID 2507.20808 · DeepID deepid-iccv.github.io · TwoHead-SwinFPN 2601.12895 · ID-Card PAD review 2511.06056 · IDNet 2408.01690 · SIDTD 2401.01858 · MultiModal Doc PAD 2404.06663 · IML-ViT 2307.14863 · FatFormer 2312.16649 · C2P-CLIP 2408.09647 · Frequency Masking 2401.06506 · synthetic-freq-injection 2403.13479 · CLIP-Flow 2508.09477 · FMAG moiré aug (2025) · COCO-Inpaint 2504.18361 · X-Edit 2505.11753 · DocForge-Bench 2603.01433.
