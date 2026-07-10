# FREUID 2026 — Winning Strategy + Novel Method + Paper Plan
(2026-06-12; rev 2026-06-13. ⚠️ Kaggle submission deadline **2026-07-14 11:59** (API-verified;
the site's July 27 is likely the code/report deadline) → ~4.5 weeks. Compress the §6
schedule by ~2 weeks: DTC verdict by 6/22, ensemble lock by 7/06, buffer 7/07–7/14.)

## 0. Public leaderboard reality (2026-06-13, via authenticated Kaggle CLI)
60 teams, 381 submissions. **≥20 teams sit at a PERFECT public score of 0.00000**
(FREUID Score = harmonic mean of converted AuDET + APCER@1%BPCER, lower=better); top
probers have 20–45 submissions. The public slice (7,821 published imgs) is saturated —
in-domain is trivial, exactly the DeepID pattern. Consequences:
- Public LB has ZERO discriminative power at the top; it only verifies pipeline format.
- Final ranking is decided 100% by the 134,997 private images (incl. unseen doc types).
- Submit once for format proof; spend ALL remaining effort on domain-holdout APCER.

## 1. Competition facts (verified from freuid2026.microblink.com + IJCAI page)
- Host: Microblink Fraud Lab; IJCAI-ECAI 2026 Bremen, live event Aug 18–21.
- Timeline: dataset June 8 → **final submission July 27** → results Aug 3 → winners Aug 10.
- Metrics: **AuDET (primary)** + APCER@1%BPCER, on a **private held-out test set**; public val leaderboard for iteration.
- Dataset: **7 document types** (Asian/African, Latin+Arabic scripts), 3 attack axes (physical / GenAI edit / print-and-capture), hybrid synthetic + physically printed-and-captured.
- **Our train manifest has only 5 doc_types** (BENIN/DL, EGYPT/DL, GUINEA/DL, MAURITIUS/ID, MOZAMBIQUE/DL) → the private test almost certainly contains **≥2 unseen document types**. Leave-one-doc-type-out validation is therefore THE correct selection protocol (already in place).
- External data: **any publicly available datasets/pretrained models allowed** (license-compatible, cited) → FantasyID/IDNet/MIDV/DLC-2021 are all legal ammunition.
- **Code release (OSI license) + short technical report REQUIRED to compete** → the paper is not optional extra work; the report is mandatory and the paper is its extension.
- Prizes: $3k/$2k/$1k; top performers get conference entry (workshop/competition track presentation).

## 2. What it takes to win (synthesis of docs/sota-research.md + baseline-verification.md)
- Ceiling is low (closest prior: DeepID winners aggregate F1 0.801, OOD AUC 0.70–0.73). Our vanilla baseline already matches that AUC (0.744). **Ranking will be decided by cross-doc-type generalization, not in-domain fit.**
- Proven levers, in order: (1) foundation backbone frozen + adapter, (2) frequency/recapture-trace pathways (LOCAL, not global FFT — exp_freq failure proved global FFT learns doc-type shortcuts), (3) heavy print-scan/moiré/recompression aug on BOTH classes, (4) ensemble + threshold calibration to the metric.
- Submit early and often to the public leaderboard, but **never select models on it** — select on leave-one-doc-type-out APCER@1%BPCER.

## 3. The novel method (the paper's contribution)
**Name (working): DTC — Double-Trace Consistency learning.**
Premise (sota-research.md novel edge #1, unpublished): a genuine capture has ONE coherent
acquisition trace across the whole image. A print-and-capture forgery of a digitally
edited document carries a LAYERED history (edit residual → halftone print → re-photo),
and local regions (the edited patch vs the rest) carry INCONSISTENT trace statistics.
Instead of asking "does this image contain recapture artifacts?" (which legitimate
recaptured bona-fides also trigger), ask **"is the acquisition trace internally
consistent?"** — turning the analog hole from a nuisance into a signature.

### Architecture = existing model + new logic (per directive)
```
                ┌─ RGB branch (existing: ConvNeXt-V2 / DINOv2-frozen, timm) ─ f_rgb
input ──┤
                └─ HPF residual branch (existing: HighPassResidual + CNN)
                          ├─ global trace feature ───────────────────────── f_trace
                          └─ NEW: patch-grid trace embeddings z_1..z_N
                                     └─ NEW: Consistency Module
                                          · pairwise cosine stats (mean/min/var)
                                          · consistency score s_c + per-patch map
fusion head (NEW): MLP([f_rgb ; f_trace ; consistency stats]) → fraud logit
```
- Base = `HPFDualStream` (src/freuid/models/freq_classifier.py) — already built & validated design.
- New logic = patch-level trace embeddings + consistency module + 3-way fusion. ~few M params on top.

### The self-supervised trick (what makes it a *learning* contribution, label-free)
Generate consistency labels for free from the existing `forensic_aug.py`:
- **Coherent sample** (y_c=1): apply ONE recapture chain uniformly to a bona-fide image.
- **Incoherent sample** (y_c=0): TraceMix — split the image into regions (CutMix-style or
  field-shaped masks), apply DIFFERENT recapture chains (different JPEG QF chain, moiré
  freq/phase, blur) per region. Mimics exactly what a localized edit + reprint produces.
- Loss: `L = BCE_fraud + λ1·BCE_consistency(self-sup) + λ2·patch-contrastive (optional)`
- The consistency head trains on synthetic labels, transfers to real fraud at test time;
  the fraud head benefits via the shared trace encoder + fused consistency stats.
- Critically domain-stable: trace consistency is doc-type-agnostic by construction →
  directly attacks the unseen-doc-type private test.

### Why this is paper-grade novelty
1. Unpublished framing (verified in sota research): existing recapture work detects
   *presence* of recapture; nobody detects *multiplicity/inconsistency* of traces.
2. Self-supervised label generation via augmentation-chain mixing is a reusable recipe.
3. Clean ablation story already staged: baseline → +heavy-aug → +HPF → +DTC (one lever
   per run, all on the same leave-one-doc-type-out protocol).
4. Built-in interpretability figure: per-patch consistency maps localize the edit.

## 4. Experiment plan → paper sections
| Stage | Runs | Paper artifact |
|---|---|---|
| S1 (now, partial→full data) | baseline / heavyaug / hpf / dinov2(-frozen) | Table 1: lever ablation |
| S2 (W2–3) | + DTC module on best S1 base; λ sweep; TraceMix on/off | Table 2: main result + ablation |
| S3 (W4) | + FantasyID/IDNet aux data; DINOv2-frozen + DTC | Table 3: external data & backbone |
| S4 (W5) | ensemble (RGB-semantic + freq-trace + DTC) + APCER@1%BPCER calibration | Table 4: final system |
| Throughout | per-doc-type & per-attack-type breakdowns; consistency-map visualizations | Figs |
- Cross-dataset eval: train FREUID → test FantasyID (manifest ready) = generalization claim.
- Every run: fixed seed/epochs/holdout; one factor changed; logged in docs/experiments.md.

## 5. Paper plan
- Mandatory technical report (competition) = condensed version of the paper. Write the
  paper skeleton FIRST, fill tables as runs finish.
- Title (draft): *"Turning the Analog Hole into a Signature: Self-Supervised
  Trace-Consistency Learning for Cross-Domain ID-Document Fraud Detection"*
- Venues: IJCAI-ECAI 2026 competition/workshop track (winners present, Aug) → extended
  version to IJCB / WACV / TIFS.
- Code release: required anyway (OSI) → keep repo clean, runs reproducible from configs.

## 6. Schedule (deadline 2026-07-27)
- **W1 (6/12–6/18)**: full data lands → full S1 sweep; first Kaggle submission (pipeline
  proof); DTC module implementation + unit tests (TraceMix, consistency head).
- **W2 (6/19–6/25)**: DTC smoke → full DTC runs vs S1 best; λ sweep.
- **W3 (6/26–7/02)**: DINOv2-frozen + DTC; aux-data (FantasyID) joint training.
- **W4 (7/03–7/09)**: ensemble construction; calibration; public-LB reality check.
- **W5 (7/10–7/16)**: final system lock; seeds×3 for variance; technical report draft.
- **W6 (7/17–7/27)**: buffer; final submissions; code-release cleanup; report submission.

## 7. Risks
- Public LB ≠ private test: never model-select on LB (only sanity).
- DTC fails to beat +HPF: fall back per failure protocol (root-cause → falsifiable hypothesis → single-factor rerun). Edge #2 (glyph forensics) is the backup novelty.
- Recapture aug overfit: keep aug applied to BOTH classes (anti-shortcut), monitor bona-fide recaptured FP rate via per-attack-type breakdown.
- Time: DTC must be testable by 6/25 or we ship the best stacked system and keep DTC as paper-only ablation.
