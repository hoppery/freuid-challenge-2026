# FREUID 2026 Technical Report (draft skeleton)
Working title: *Turning the Analog Hole into a Signature: Trace-Consistency Learning
and Cross-Type Ensembling for ID-Document Fraud Detection*

## 0. Final system + honest numbers (as of iter #28, 2026-06-13)
**Deployment single model**: DINOv2 ViT-B (last-2-blocks unfrozen) + DTC + 518px, trained
on ALL 5 doc types (exp_final518_all_s42/s43). In-domain random-val FREUID ≈ 0.000
(saturated, like the 20+ public-LB 0.0 teams — not meaningful).
**Honest cross-type estimate = leave-one-doc-type-out fold mean** (the private test has
unseen types):
| fold | 518px | + conservative TTA |
|---|---|---|
| MAURITIUS | 0.0006 | 0.0005 |
| GUINEA | 0.0025 | 0.0036 |
| MOZ | 0.0188 | **0.0058** |
| EGYPT | 0.117 | **0.071** |
| BENIN | 0.152 | 0.167 |
→ LODO mean: frozen **0.058** → +conservative-TTA **0.050** (TTA validated on ALL 5
folds: strong win on EGYPT/MOZ, neutral on MAURITIUS/GUINEA, −0.015 on BENIN only). Two levers that WORK both change *what the model sees*
(resolution, test-time adaptation); five loss/train tricks failed (§4).

## 1. Method (final system)
1. **Host**: DINOv2 ViT-B/14 (LVD-142M), last 2 blocks + final norm unfrozen, 378px.
   Full freeze fails (doc-type novelty → attack conflation; §4.1); full fine-tune
   destroys the pretrained manifold. Partial plasticity is the sweet spot.
2. **DTC (contribution)**: dual-stream trace branch (high-pass residual → patch
   embeddings → pairwise-cosine consistency statistics) + self-supervised TraceMix
   labels (coherent = one recapture chain uniformly; incoherent = contrasting chains
   per region). Joint loss BCE_fraud + 0.5·BCE_cons.
   - Peak parity with host-only, but **5× more stable under continued training**
     (ep12-14 FREUID 0.13 vs 0.71-0.81 without DTC) → domain-overfit regularizer.
3. **FDA cross-type amplitude swap** (p=0.5): donor of a different doc type provides
   the low-freq amplitude (appearance); phase (forensic content) kept. Stabilizes
   hard-fold training (no collapse) + small gain.
4. **Cross-type (LODO) ensemble**: 5 leave-one-doc-type-out models, score-averaged.
   Each member is "blind" to one type → calibrated to novelty; public 0.2826 (best
   single) → 0.2318 (ensemble). Adding all-types members HURT (0.2365): in-domain
   experts add no OOD diversity.

## 2. Evaluation protocol
- Leave-one-doc-type-out; report ALL five folds (difficulty varies 500×!):
  MAURITIUS 0.0006 · GUINEA 0.0025 · MOZ 0.0188 · EGYPT 0.2115 · BENIN 0.3191.
- Selection metric = official FREUID Score = 1−HM(1−AuDET, 1−APCER@1%BPCER).
- Public LB used as sanity only (5.5% slice; ≥20 teams at probed 0.0).

## 3. Submission trajectory (public)
| v | system | public |
|---|---|---|
| 1 | ConvNeXt-T DTC (46% data) | 0.3574 |
| 2 | DINOv2 partial-unfreeze | 0.2869 |
| 3 | + DTC | 0.2826 |
| 4 | **5-fold LODO ensemble** | **0.2318** |
| 5 | + 2 all-types members | 0.2365 (rejected) |
| 6 | hard folds → FDA members | (pending) |

## 4. Negative results (full failure analyses in docs/iterations.md)
1. Frozen DINOv2 + adapter: unseen-type bona-fides → confident attacks (bona median
   0.9868 vs 0.12 in-domain). Plasticity required.
2. Global-FFT branch: doc-type shortcut. Local residual required.
3. SBD (self-blended pseudo-attacks): blend-artifact shortcut; real attacks lack it.
4. FantasyID joint (3.3k imgs): no effect at 53k scale.
5. Patch-only training: layout destruction ≠ transfer; loses context signal.
6. ViT-L: capacity is not the bottleneck.
7. GRL doc-type adversarial: destabilizes without improving transfer.
8. hflip TTA: negligible (−0.001).
9. Late-epoch collapse is the universal failure mode on hard folds; rank-based metrics
   are calibration-invariant → only ranking changes (features/ensembling) matter.

## 5. Key observations for the community
- Cross-type transfer difficulty is **fold-idiosyncratic** (0.0006→0.32), driven by
  whether a "twin" type remains in training — not by script (Latin BENIN is the
  hardest), not by capacity.
- AuDET/APCER@1% both rank-based → calibration/temperature CANNOT change scores.
- TraceMix consistency supervision = cheap, label-free regularizer against
  domain overfit.

## 3c. Quantitative ablations (FREUID Score, lower=better; LODO unless noted)

### A. Resolution (the dominant lever) — EGYPT/BENIN LODO
| img_size | EGYPT | BENIN | public (all-types) |
|---|---|---|---|
| 378 | 0.2115 | 0.3191 | — |
| 518 | 0.117 | 0.152 | 0.2576 |
| **728** | **0.0343** | **0.1325** | **0.20375** |
Monotone, not saturated. EGYPT 6× from resolution alone.

### B. DTC components (518px / partial-data where noted)
| variant | EGYPT-fold | note |
|---|---|---|
| DINOv2-uf2, no DTC | 0.0142 peak → 0.71–0.81 late | late-epoch collapse |
| + DTC (consistency) | 0.0188 peak → 0.13 late | 5× more stable |
| DTC λ=0 (TraceMix aug only) | 0.4087 | aux loss needed |
| DTC λ=0.5 | 0.4059 | self-sup consistency helps |
DTC = domain-overfit regularizer; TraceMix self-supervised consistency is the active part.

### C. Conservative TENT TTA — 518px LODO frozen→+TTA
MAURITIUS 0.0006→0.0005 · GUINEA 0.0025→0.0036 · MOZ 0.0188→0.0058 ·
EGYPT 0.117→0.071 · BENIN 0.152→0.167. Mean 0.058→0.050.

### D. Rejected components (each regresses ≥1 axis vs FREUID-only)
GRL adv 0.258 · EMA ≈base · bona-compactness collapse · APCER-surrogate 0.23 ·
MixStyle 0.249 · DTC-localization 0.195 · CLIP stream FantasyID-AUC 0.538 ·
external-data union FantasyID-AUC 0.46 · heavy synthetic recapture (hurts both).

## 4b. plan_v3 negative results (external data & arch, 2026-06-14)
- Multi-dataset union (IDNet+DocXPand+BID): fraud-semantics mismatch muddles the boundary
  (FantasyID AUC 0.46, below chance). Class-balance tuning only slides global bias, never
  creates separation. FantasyID xeval is an UNFAIR cross-dataset proxy (ALL archs ~chance)
  → deweight; use FREUID LODO.
- DTC localization head, CLIP semantic stream, heavy synthetic recapture aug: no gain.

## 5b. Headline finding (2026-06-14): RESOLUTION is the dominant lever
378→518→728px monotonically breaks the hard-fold wall (EGYPT 0.21→0.117→0.034, 6×;
BENIN 0.32→0.152→0.133). Architecture/loss innovations plateau; high-res forensic detail
dominates. Deployment = 728px all-types + conservative TTA. Resolution not yet saturated
(896/1036px next).

## TODO before 7/14
- [ ] 728px all-types deployment retrain (running) + 3-axis validation; submit
- [ ] push resolution further (896/1036px)
- [ ] wire TTA into deployment inference; re-run when private images drop
- [ ] code release (OSI) + cite data/models (IDNet/DocXPand/BID/FantasyID, DINOv2)
