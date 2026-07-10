# Contribution Loop — Brainstorm → Implement → Train → Analyze → Repeat

Goal: lower the official **FREUID Score** (= 1 − harmonic-mean(1−AuDET, 1−APCER@1%BPCER), lower better) with methods that carry a real contribution, not mere hyperparameter tuning. Every iteration records: failure analysis of the previous step → hypothesis → method → result → verdict.

Reference: **baseline FREUID 0.4028** (AuDET 0.257, APCER@1% 0.501, cross-domain holdout MOZAMBIQUE/DL).

---

## ★★★ DOMINANT LEVER — INPUT RESOLUTION (2026-06-14, user-provided + validating)
Confirmed external result: **input resolution crushes both hard folds**, far beyond FDA/diversity. Forgery cues are fine-grained (splice boundaries, glyph/microtext, print halftone/moiré, JPEG-block edges); at 378px a full ID is downsampled so the cue smears — high-res preserves it.

| LODO fold | 378 | 518 | 728 | 896 |
|-----------|-----|-----|-----|-----|
| EGYPT | 0.2115 | 0.117 | 0.0343 | **0.0217** |
| BENIN | 0.3191 | 0.152 | 0.1325 | **0.0780** |

(378 column ≈ our Iter5 FDA-378: EGYPT 0.214, BENIN 0.293 → table is consistent with our pipeline.) **Monotonic: higher = better.** 896 nearly solves the two hardest folds.

**Infra added for hi-res (this session):** `grad_accum` + `grad_checkpointing` config fields; train loop does true gradient accumulation (LR/EMA stepped per optimizer step); `enable_grad_checkpointing()` in classifier. Probe (`scripts/probe_batch.py`): with partial-unfreeze (only 14M trainable) **VRAM is NOT the bottleneck** — bs=32@896 = 10.3GB (no checkpointing), bs=48@728 = 10.2GB. **Compute/wall-clock is the cost.** Policy: skip checkpointing (memory ample), batch 32 (~48% VRAM, safe headroom — NOT maxed, per user), DINOv2-uf2 + FDA + bf16 + grad_clip + EMA + best-epoch.

**FORWARD PLAN (resolution-first):**
1. **Iter8 — validate 896 on EGYPT+BENIN** (exp_fda896, isolates resolution vs Iter5 FDA-378). **★ CONFIRMED (2026-06-14): EGYPT 896 = 0.0021 (best ep0, AUC 0.9995!), BENIN 896 = 0.1188 (best ep3).** EGYPT 0.194→0.0021 (~90×); our FDA+896 BEATS the table's 896 (0.0217). BENIN 0.293→0.1188 (table 896=0.078; ours is FDA+medium-aug, slightly higher but huge gain). folds 1,3,4 (GUINEA/MOZAMBIQUE/MAURITIUS, all tiny even at 378) running with early-stop → full 5-fold 896 ETA ~03:30. Projected 5-fold avg ≈ 0.03 (vs Iter6 0.114) = ~3-4× better.
**★★ FULL 5-FOLD 896 (2026-06-15): LODO-CV ≈ 0.0286** — EGYPT 0.0021, GUINEA 0.0197, BENIN 0.1188, MOZAMBIQUE 0.0016, MAURITIUS 0.0008. **4× better than Iter6 (0.114).** 4 of 5 types essentially solved; BENIN (0.1188) is the lone outlier and dominates the average → Iter9 BENIN focus is the highest-leverage next step. Early-stop saved ~half the compute (GUINEA stopped @ep2). NEW WINNING BASE = DINOv2-uf2 + FDA + **896** + medium aug + bf16/grad_clip/EMA + best-epoch + early-stop. Resolution = the decisive lever, definitively. Wall-clock: ~45min/epoch at 896/bs32 (forward runs all 12 ViT blocks @ 4101 tokens) → ~4.5h/fold, full 5-fold ≈ 13h. Auto-continuation (`scripts/wait_then_896_rest.sh`) runs folds 1,3,4 after validation → full 5-fold 896 LODO-CV.
**Collapse universal (confirmed @896):** EGYPT ep0 0.0021, ep1 0.0022, **ep2 0.53** (APCER@1% 0.695, AUC still 0.98) — same best-at-ep0-1-then-tail-collapse pattern as 378, one epoch later. best-epoch rescues. → added **early stopping** (`early_stop_patience`, train.py; default off; =2 on folds 1,3,4) — keeps the epochs=6 LR schedule (no confound) but stops once collapsed, saving ~2-3h/fold. 36 tests green.
2. If reproduced → **full 5-fold at 896** (the new winning base) → expect LODO-CV ≪ 0.114.
3. **Resolution ensemble**: members at 728 + 896 (diverse receptive fields) average well on OOD.
4. **Re-test orthogonal levers AT high-res** (cheap relative to the gain): FDA on/off, IDNet diversity, the APCER@1% operating-point lever — but resolution is priority #1.
5. Aug caution: at hi-res, do NOT use aggressive downscale aug (would undo the resolution benefit); keep `medium`.
6. Compute budget: hi-res is ~8-10× slower/img than 378 → use both GPUs, 1 fold/GPU, epochs=6 + best-epoch (don't compress epochs — that confounds the LR schedule, cf. Iter7-v1 failure).

---

## Iteration 9 — BENIN focus (the lone hard fold)  [QUEUED → launch after Iter8 5-fold]
**Why:** at 896 every fold is ≈0.002-0.02 EXCEPT BENIN (0.1188). BENIN dominates the 5-fold avg → highest-leverage target. In LODO, BENIN = our proxy for an unseen private type like BENIN → improving it hardens the private test directly.
**Diagnosis (2026-06-14):** BENIN size/balance normal (8001 bona/5368 atk) → hardness intrinsic. AUC 0.96 good but **APCER@1% 0.21 is the killer** = strict operating-point tail, not ranking. Attack-family breakdown unavailable (`is_digital`≈True for all 69,332 rows → FREUID train attacks are ~all digital; physical/print likely private-test-only = a 2nd OOD axis). **Key lead: our 896 BENIN (0.1188) is WORSE than the reference table's 896 BENIN (0.078).** The difference is our FDA + medium recapture-aug → hypothesis: at high-res, gentle recapture-aug (JPEG/moiré/downscale) SMEARS the fine cue 896 preserves, hurting the hardest fold most.
**Plan (ablation, BENIN fold @896, 2 GPUs parallel):** `exp_bn896_light` (train_aug=train, no recapture, fda=false) vs `exp_bn896_lightfda` (light + FDA) vs current medium+FDA 0.1188. If light≈0.078 → drop recapture-aug at high-res. Isolates aug vs FDA. Then if resolution still bottlenecks → BENIN @1024.
**Result (2026-06-15): HYPOTHESIS REFUTED — aug/FDA do NOT hurt BENIN; both HELP.** BENIN best: medium+FDA **0.1188** < light+FDA 0.1277 < light+noFDA 0.1372. So recapture-aug AND FDA both regularize (help generalization); the reference config is already optimal. The gap to the table's 896 (0.078) is NOT explained by aug → likely a different backbone/data/protocol in that table. **BENIN floor with our pipeline ≈0.12; aug is not the lever.** Next BENIN levers (genuinely different): (a) **1024 resolution** — resolution has been the single most reliable lever, extend the trend; (b) **operating-point loss** — BENIN AUC 0.96 is fine, the killer is APCER@1% (0.19), so a tail/threshold-aligned objective targets the actual error (tail_margin failed at 378 because features were bad; at 896 features are good, so worth a re-test). → launch BENIN@1024 when ablations free the GPUs.
**Result 2 — resolution past 896 (2026-06-15): ✗ BENIN@1036 = 0.2127 (best ep0) — WORSE than 896 (0.1188).** Pushing res past 896 hurts BENIN (likely DINOv2 pos-embed interpolation to 74×74 grid degrades faster than the finer cue helps). **Resolution sweet spot = 896; lever exhausted for BENIN.** Both aug AND resolution refuted → last principled lever = **operating-point loss** (BENIN AUC 0.96 fine, killer is APCER@1% 0.19; tail_margin failed at 378 w/ bad features, but at 896 features are good → re-test, prob-space safe).
**Result 3 — operating-point loss (2026-06-15): ✓✓✓ BREAKTHROUGH. BENIN tail_margin @896 = 0.0827 (ep3, still improving) vs bce 0.1188 (−30%); APCER@1% 0.1876→0.1319. Reaches the reference table's 896 BENIN (0.078).** Monotonic descent 0.1814→0.1280→0.1035→0.0827 as tail terms warm up. ep4 0.0900 (turned up) → **final BENIN op-loss = 0.0827**. Stronger λ=2 variant WORSE (0.1657) — λ=1 is the right setting. **NEXT: full 5-fold with tail_margin (λ=1)** — `exp_tail896_fold{0,1,3,4}` (BENIN fold2 already = 0.0827) → verify tail loss doesn't hurt the near-perfect easy folds (EGYPT 0.0021 etc.) AND get the new LODO-CV. Projected if easy folds neutral: 5-fold ≈ (0.0021+0.0197+0.0827+0.0016+0.0008)/5 = **0.0214** (vs 0.0286, −25%). If confirmed → tail_margin becomes part of the winning recipe + production retrain. The tail loss directly improves the strict operating point (its design goal) — exactly BENIN's failure mode. Confirms: at 896 the FEATURES are good (why it failed at 378 but works now). This is the BENIN lever. Tuning a stronger variant (`exp_bn896_tail2`: tail_lambda 2.0, tail_mu 0.3). → If it holds, apply tail_margin to the full 5-fold + production (could lower the whole LODO-CV below 0.0286, since BENIN dominates the average).

## Production model (deliverable A, public LB)
**exp_prod896** (all 5 doc-types, random 12% in-domain val, 896 winning recipe): ep1 in-domain FREUID **0.0005** (AUC 0.9999) → near-perfect in-domain = strong public-LB candidate. The 5 LODO fold models (exp_fda896_fold0-4) = the OOD ensemble (deliverable B). NEXT: ensemble-infer on public test + submit (ASK permission per [[kaggle-submit-rule]]).

---

## Iteration 1 — Quantile-Anchored Tail-Margin Loss (operating-point-aligned training)

**Failure analysis of prior steps:**
- exp_freq (global FFT branch) FREUID 0.96: global spectrum encodes doc-type layout → shortcut, cross-domain collapse.
- exp_heavyaug: AUC 0.50 flat — aggressive recapture aug erases the fine digital-forgery cue entirely.
- exp_dinov2 full-FT FREUID 0.82: 85M-param full fine-tune memorizes the public domain (AUC decays after ep2).
- **baseline 0.4028: the binding constraint is APCER@1%BPCER = 0.501** — the harmonic mean is dominated by its worse component. BCE training is *structurally misaligned* with this: it weights all operating points equally and never explicitly shapes the decision region the metric measures (the bona-fide 99th-percentile threshold and the attack mass just below it).

**Hypothesis:** adding an explicit, differentiable surrogate for the 1%-BPCER operating point will pull attack scores above the bona-fide upper tail, cutting APCER@1% (0.50 → ≤0.40) without materially hurting AuDET → FREUID drops.

**Method (contribution):** `TailMarginBCE` (`src/freuid/losses.py`) =
BCE + λ·ReLU(τ + margin − s_attack) + μ·ReLU(s_bona − τ),
where τ = the 0.99 quantile of a sliding **memory bank of recent bona-fide logits** (detached, fp32, 8192 entries) — a training-time estimate of the exact threshold APCER@1%BPCER is measured at. Related to partial-AUC optimization, specialized to the asymmetric PAD operating point with a self-tracking quantile anchor; composable with any backbone/architecture.

**Experiment:** `configs/exp_tailloss.yaml` — identical to baseline except `loss: tail_margin` (single-factor).
**Falsification:** if APCER@1% does not drop, inspect (a) τ tracking vs the true epoch-end quantile, (b) hinge saturation (margin too large/small), (c) AuDET degradation from tail over-weighting.

**Supporting evidence from the corrected wave (2026-06-13):** APCER@1% stayed ~0.50–0.88 across EVERY variant — baseline 0.501, exp_medium 0.501, exp_dinov2 0.621, exp_hpf 0.877 — while AuDET varied. No architecture/backbone/aug change moved the tail. The binding constraint is the training objective, not capacity.

Additional failure analyses recorded:
- exp_hpf: train loss → 0.03 with AUC collapse ep8-9 → the high-pass branch is itself a shortcut (memorizes high-freq print/doc signatures). Extra capacity without invariance constraints accelerates overfitting.
- exp_medium: FREUID 0.4017 (≈baseline) — gentle recapture aug is safe; keep as default aug going forward.
- baseline_indomain: collapsed to constant-prior output (loss 0.682 = prior BCE) — training anomaly, rerun queued.

**Result (2026-06-13): FAILED — loss divergence (engineering failure, hypothesis NOT falsified).**
Train loss exploded 101 → 333 → 767 (AUC pinned 0.50). Root cause: the hinge operates in
**logit space** with a self-referential target — pushing attack logits above τ lifts bona
logits via shared features, τ (a bona quantile) rises, the hinge target rises, repeat →
unbounded scale arms race. Side results from the same wave: **exp_dinov2_frozen** (769-param
linear probe) achieved the best AuDET (0.2405) / AUC (0.759) of any run — foundation features
generalize — but tail still weak (APCER@1% 0.62, FREUID 0.4960). baseline_indomain collapsed
to constant-prior a second time with seed 42 (deterministic repeat, diagnostic queued with seed 43).

---

## Iteration 1.1 — Bounded probability-space tail loss (fix of Iter 1)

**Failure analysis (from Iter 1):** divergence is caused by unbounded logit-space hinge with a
moving quantile target, NOT by the operating-point-alignment idea itself.

**Fix (method change):** tail terms now operate on **sigmoid(score) ∈ [0,1]**: τ_p = 0.99-quantile
of bona-fide *probabilities* (sliding bank); hinge = λ·ReLU(τ_p + 0.1 − p_attack); tail-compress =
μ·ReLU(p_bona − τ_p). Every term bounded ≤ λ(1+m)+μ; sigmoid saturation kills gradients at extreme
logits → arms race structurally impossible. Plus warmup (300 steps) so τ_p reflects a settled
distribution. Regression test added (extreme ±50 logits → loss bounded).

**Experiment:** `exp_tailloss2` (single factor vs baseline) + `baseline_indomain_s43` diagnostic.
**Success criterion:** APCER@1% < 0.50 and FREUID < 0.4017 (current leader exp_medium).
**Result (2026-06-14): FAILED to improve cross-domain.** exp_tailloss2b FREUID 0.4022 ≈ baseline_v2 0.4006. Training was stable (bf16 fix worked) — the loss is fine, the hypothesis is just *insufficient*.

---

## ★ PIVOTAL FINDING (2026-06-14): the bottleneck is DOMAIN GENERALIZATION, not the operating point or capacity.

Stable-training wave (bf16 + warmup) results, cross-domain holdout = MOZAMBIQUE/DL:
| exp | FREUID | AuDET | APCER@1% | AUC | note |
|---|---|---|---|---|---|
| baseline_indomain_v2 (RANDOM split) | **0.180** | 0.081 | 0.261 | **0.919** | in-domain is EASY |
| baseline_v2 (cross-domain) | 0.401 | 0.250 | 0.501 | 0.750 | collapses on unseen doc_type |
| exp_tailloss2b | 0.402 | 0.252 | 0.502 | 0.748 | tail loss no help cross-domain |
| exp_medium | 0.402 | 0.253 | 0.501 | 0.747 | gentle aug ≈ baseline |
| exp_dinov2_frozen | 0.496 | **0.240** | 0.623 | **0.759** | best AuDET/AUC; weak tail |
| exp_heavyaug_v2 | 0.980 | 0.500 | 0.990 | 0.500 | heavy aug destroys cue (NOT fp16) |

**Interpretation:** the SAME model gets AUC 0.92 / APCER@1% 0.26 in-domain but AUC 0.75 / APCER@1% 0.50 on an unseen doc_type. The fraud signal is learnable; it just doesn't transfer across document types — the model keys on doc-type-specific appearance. **Why every prior iteration failed:** operating-point loss, capacity (base/dual), and aug all leave the *features* doc-type-specific; on the unseen type the classes overlap so nothing downstream helps. Also: cross-domain metric has huge epoch-to-epoch variance (best-epoch FREUID swings ~0.42↔0.89) → need EMA / smoothed selection to compare methods reliably.

**Strategy pivot:** (1) submit baseline to Kaggle to learn whether the public/private test is in-domain (≈0.18) or cross-domain (≈0.40) — calibrates everything + validates our FREUID Score impl. (2) All further iterations target cross-doc-type generalization.

---

## Iteration 2 — Multi-source domain-diverse training (data-centric DG) + weight EMA
**Failure analysis (Iter 1/1.1):** algorithmic operating-point shaping can't fix non-transferable features. The most reliable DG lever is *more source domains*. We have aux corpora already on disk (FantasyID 13 country doc-types incl. Arabic/Persian; later IDNet/DocXPand) normalized to the same manifest schema.
**Method:** train on FREUID(4 types) ∪ FantasyID(13 types), hold out a FREUID type (MOZAMBIQUE/DL) for val → does 4→17 source-domain diversity close the unseen-type gap? Add weight EMA for stable measurement + generalization. Contribution = cross-corpus domain-diverse training protocol for ID-fraud with explicit unseen-doc-type validation (not tuning).
**Success criterion:** cross-domain FREUID < 0.40 (beat baseline_v2) with EMA-stable epochs.
**Result (2026-06-14): NO GAIN.** First attempt (exp_multisrc) COLLAPSED (loss→ln2, AUC 0.50). **Failure analysis:** not data quality (3284 FantasyID imgs all clean, no NaN); it was UNSTABLE OPTIMIZATION — baseline itself oscillated FREUID 0.42↔0.94 epoch-to-epoch; lr 3e-4 full-FT + the added domain-far data tipped it into full collapse. **Fix = gradient clipping (max-norm 1.0)** → exp_multisrc_v2 trained stably (no collapse). But final **FREUID 0.4203 ≈ baseline 0.401 (no cross-domain gain)**. Conclusion: **FantasyID (fantasy-design cards) is too domain-far from real African/Asian IDs** to help generalize to the held-out real FREUID type. Useful infra win (grad_clip now default). Next: real-ID external data (BID/MIDV/IDNet), and the partial-unfreeze DINOv2 backbone.

---

## Iteration 3 — DINOv2 partial-unfreeze (last-k blocks) @ hi-res  [RUNNING]
**Failure analysis (Iter 2 + earlier):** data-centric DG with domain-far synthetic data didn't help; frozen DINOv2 underfits (FREUID 0.496) and full-FT overfits (0.50→collapse). **Method:** DINOv2 ViT-B, unfreeze last-2 blocks + norm + head (14M/16.5% trainable), 378px, medium aug, bf16+grad_clip+EMA, lr 5e-5 — the DG sweet spot per the winning-strategy goal. `exp_dinov2_uf2` (single-fold MOZAMBIQUE screen; 5-fold LODO if it beats baseline).
**Success criterion:** cross-domain FREUID < 0.40.
**Result (2026-06-14): single-fold MOZAMBIQUE 0.003 was MISLEADING.** Full **5-fold LODO-CV = 0.378 ± 0.341** (EGYPT 0.504, GUINEA 0.145, BENIN 0.262, MOZAMBIQUE 0.003, **MAURITIUS/ID 0.975**) ≈ baseline (0.40) with HUGE variance. Lessons: (1) single fold is unreliable — always 5-fold; (2) **MAURITIUS/ID is the killer fold** (only "ID", rest "DL") — train-on-DL → held-out-ID generalization fails (AUC 0.61); this is the private-test (unseen-type) challenge in miniature; (3) partial-unfreeze overfits FAST (best ep0-1). Corroborates that DINOv2-uf2 alone isn't enough — need domain-invariance (DTC).

---

## Iteration 4 — DTC: gradient-reversal Doc-Type Classifier (domain-invariant)  [RUNNING]
**Method:** `models/dtc.py` — DINOv2 partial-unfreeze backbone + fraud head + doc-type head via Gradient Reversal Layer (alpha DANN-ramp). Forces doc-type-invariant fraud features. Inference returns fraud only (eval/infer unchanged). Corroborated lever (parallel pipeline 5-fold avg ~0.11 with DTC vs ~0.38 without).
**Screen (hardest fold MAURITIUS/ID): NO improvement** — DTC 0.974 vs uf2 0.975. Analysis: invariance across the 4 DL types can't bridge to a held-out ID layout (intractable extrapolation); DTC's benefit is expected on the OTHER folds. ⇒ judge by the **full 5-fold DTC average** (running), not the worst fold.
**Success criterion:** 5-fold DTC avg < 0.378 (beat no-DTC). **Result (2026-06-14): FAILED — 5-fold DTC avg 0.549 ± 0.294 (WORSE than no-DTC 0.378).** Per-fold: EGYPT 0.748, GUINEA 0.666, BENIN 0.222, MOZAMBIQUE 0.185, MAURITIUS 0.922. Adversarial GRL hurt most folds (AUC similar but APCER@1% much worse → reversal flattens the operating-point tail and strips doc-type-correlated but USEFUL fraud cues). The parallel pipeline's "DTC stabilizes late epochs" is likely a NON-adversarial multi-task regularizer, not GRL. Adversarial-DTC rejected at alpha=1.

---

## Iteration 5 — FDA (Fourier Domain Adaptation) augmentation  [RUNNING]
**Failure analysis (Iter 4):** removing doc-type info adversarially hurts. Instead, RANDOMIZE doc-type STYLE so the model can't rely on it: FDA swaps the low-freq amplitude (style) of a training image with a random other-doc-type reference, keeping phase (content/fraud cue). `data/fda.py`. DINOv2-uf2 + FDA (no DTC), 5-fold. Corroborated (parallel used FDA).
**Success criterion:** 5-fold FDA avg < 0.378. **Result (2026-06-14): ★ BREAKTHROUGH ★ 5-fold FDA avg = 0.1405 ± 0.116 (vs 0.378 no-FDA, −63%).** Per-fold: EGYPT 0.214, GUINEA 0.185, BENIN 0.293, MOZAMBIQUE 0.010, **MAURITIUS/ID 0.0011** (the killer fold: 0.975→0.001!). FDA's low-freq style randomization fixes the unseen-doc-type/unseen-layout gap — exactly the private-test (2 unseen types) objective. Matches the parallel pipeline's single-method fold-avg ~0.11. **FDA is the core winning ingredient.** → 5-fold FDA ENSEMBLE submitted: **public LB 0.20748** (beats prior account best 0.232 and our baseline 0.356). Best submission on the account, with FDA alone (no DTC).

---

## Iteration 6 — FDA + IDNet real-layout diversity  [RUNNING]
**Rationale:** FDA solves within-FREUID-style cross-type generalization (LODO 0.14). For the PRIVATE 2 truly-unseen types (likely different document designs, not FREUID-style), add REAL diverse layouts: IDNet (32k, 2 EU IDs + 2 EU passports — layouts FREUID lacks). DINOv2-uf2 + FDA + IDNet extra training data, 5-fold (hold out each FREUID type, train on rest + all IDNet). `manifests/idnet.parquet`, `adapters/idnet.py`.
**Success criterion:** 5-fold avg < 0.1405 (FDA alone) AND/OR better on the hard folds → evidence it helps truly-unseen layouts.
**Result (2026-06-14): ✓ IDNet diversity BEATS FDA-alone. 5-fold LODO-CV = 0.114 ± 0.103 (vs FDA-alone 0.1405, −19%).** Per-fold best-epoch (vs FDA-alone): EGYPT **0.194** (0.214 ✓), GUINEA **0.0375** (0.185 ✓✓), BENIN **0.275** (0.293 ✓), MOZAMBIQUE **0.0615** (0.010 ✗ regressed), MAURITIUS **0.0024** (0.0011 ≈). 4 of 5 folds improve; only MOZAMBIQUE (already trivial under FDA) regressed slightly. Real diverse layouts (EU IDs + passports) help the held-out FREUID types generalize → supports the private-test plan (broaden script/layout/category coverage). → scale to 10 locations (Iter7).
**⚠ KEY FINDING — late-epoch APCER@1% collapse:** with epochs=6, both folds peak at **ep1** then degrade monotonically — AUC holds (0.84–0.99) but APCER@1% collapses (EGYPT 0.28→0.96, GUINEA 0.065→0.78), so FREUID Score blows up (0.19→0.92, 0.04→0.65). The held-out bona-fide *tail* gets contaminated after the model starts overfitting the (source-domain + IDNet) training distribution. Best-epoch selection rescues the checkpoint, so results are unharmed — but **epochs=6 wastes ~65% of compute training into guaranteed collapse.** Fix carried into Iter7: epochs 6→3 (per-step cosine decays faster → lower late-epoch LR → also dampens the collapse). Candidate deeper fix (future iter): tail-robust operating-point loss or score calibration to stop the APCER@1% drift directly.

---

## PRIVATE-TEST INFERENCE & STRATEGY (2026-06-14)
**Inferred private composition (logical):** (1) 2 unseen doc-types = new countries, likely a new *category* (passport/MRZ) and/or *script* (Arabic/Cyrillic/Greek) — the hardest OOD axis vs the seen 4-African-DL+1-ID Latin set. (2) Private ≈135k = 17× public → **ranking is decided entirely by the unseen types; public LB (7.8k) is near-worthless for selection.** (3) Same 3 attack families; harmonic-mean metric → must hold BOTH AuDET and APCER@1% on unseen types. (4) Same Microblink capture pipeline → unseen types **share capture style** with seen (why FDA works); but LODO (0.14) is slightly optimistic since it holds out layout only, not script/category → **build margin.**
**Strategy (executing):** maximize layout/script/category diversity so "unseen ≈ less unseen" → **all 10 IDNet locations** (Latin ESP/FIN/EST/SVK/ALB + Cyrillic RUS/SRB + Greek GRC + passport categories AZE/GRC/LVA/RUS/SRB); + FDA style-randomization (proven); + DINOv2 partial-unfreeze; select by LODO-CV (never public LB); diversity ensemble. Harsher proxy planned: leave-one-external-corpus/script-out.

---

## Iteration 7 — FDA + IDNet-all (10 locations, max script/layout/category diversity)  [QUEUED → auto-launch]
**Rationale (from private-test inference):** Iter6 (4 EU locations) already beats FDA-alone on the first 2 folds → diversity helps. Scale it: 10 IDNet locations spanning Latin/Cyrillic/Greek scripts + ID & passport categories, so the training manifold pre-covers the kinds of shift the 2 unseen private types will exhibit. Caps lowered (pos 2500 / fraud 1250 per loc ≈ 50k total) so FREUID (69k) stays the dominant domain and IDNet is a diversity supplement, not a style takeover. **epochs=3** (from the Iter6 late-collapse finding). `manifests/idnet_all.parquet`, `configs/exp_idnetall*.yaml`, auto-launched by `scripts/wait_then_iter7.sh` after Iter6 + extraction finish.
**Success criterion:** 5-fold LODO avg < Iter6 (and ideally < 0.1405) → evidence that broader script/category diversity transfers to truly-unseen types.
**Result v1 (2026-06-14): ✗ ABORTED — confounded + regressing.** I changed TWO levers vs Iter6 (50k IDNet @ epochs=3). fold0 EGYPT collapsed to **0.552** (vs Iter6 0.194), fold1 GUINEA improved to 0.0218. Diagnosis: (a) 50k synthetic IDNet = 48% of train (vs 37% in Iter6) over-dilutes the African-DL target → hurts held-out EGYPT (the "domain-far synthetic too dominant" failure, cf. FantasyID multisrc); (b) epochs 6→3 was a confound. **My mistake: optimized epochs prematurely (the Iter6 collapse is handled by best-epoch; cutting epochs only confounded the data test).** Killed at 2/5 folds.
**Result v2 [RUNNING] — clean isolation:** 10 locations capped to **32k total (constant volume vs Iter6's 4-loc@32k), epochs=6 (proven schedule)** → the ONLY variable is *diversity*. `pos_cap 1600 / fraud_cap 800` per loc. If < 0.114 → broader script/category diversity helps at constant volume → adopt 10-loc. If ≥ 0.114 → 4 locations already suffice; pivot to the APCER@1% operating-point lever (the dominant error mode in every fold). Launched 17:02, `/tmp/iter7_clean.log`, waiter b3yfl52df.

---

## Iteration 2 (queued candidate) — Doc-Type-Adversarial Disentanglement
Gradient-reversal head predicting `doc_type` → makes fraud features invariant to document domain; directly optimizes the public≠private shift (AuDET side). To run after Iter-1 readout.

## Iteration 3 (candidate) — Bona-fide-anchored one-class regularization
Compact bona-fide feature manifold (center/contraction loss on bona-fide only) so unseen attack types fall outside → unseen-attack generalization + tail behavior.

## Iteration 4 (candidate) — Self-supervised masked-frequency pretraining on FREUID
MAE-style masked-frequency modeling on the document corpus itself → substrate-manifold prior; anomaly-style head. Bigger engineering; schedule when GPUs free up.

---

## Iteration 10 — External diversity (IDNet) at 896, validated by LODO-CV  [RUNNING]
**Hypothesis:** goal method-#3 = expand doc-type diversity to generalize to unseen types; at 378 IDNet HELPED LODO (0.1405→0.114). Cross-corpus tests (div896/divfid896) were negative, but cross-corpus (alien pipeline) ≠ LODO (the goal's criterion). Untested: does IDNet improve the 896+tail LODO (vs tail-alone 0.020)?
**Method:** FREUID(minus one type) + IDNet-all(32k) extra training, tail_margin + 896 + medium aug, best-epoch on held-out FREUID type. Gate folds: BENIN (hard) + EGYPT.
**Success criterion:** IDNet-LODO < tail-alone (BENIN 0.0827 / 5-fold 0.0200). If yes → full 5-fold + acquire MIDV/more real-ID (4→30+ types). If no → diversity doesn't help at 896 (resolution already saturates the cue); pivot to ensemble/novel-contribution levers.
**Result:** _BENIN+EGYPT gate running (GPU0,1)._

**Result (2026-06-16): ✗ NEGATIVE — IDNet diversity HURTS at 896.** Gate: BENIN best 0.1407 (vs tail-alone 0.0827, +70% worse), EGYPT 0.0050 (vs 0.0016). Both peaked ep0 then collapsed. OPPOSITE of 378 (IDNet helped 0.1405→0.114). Reason: at 896 resolution saturates the fine cue → heterogeneous synthetic data only DILUTES the FREUID-specific signal (consistent with cross-corpus div896/divfid896 negatives). **VERDICT: external-data diversity is dead at 896; resolution superseded it. tail-alone (LODO 0.0200) is the winning base.** → pivot to ensemble (#5: seed+resolution diversity) and novel contributions (#4); accept tail 0.020 as strong base.

---

## Iteration 11 — Ensemble (#5), LODO-validated  [PARTIAL]
**Test A — diverse-RECIPE members (existing fold ckpts), BENIN fold, logit-mean:** tail 0.0827 + recap 0.1309 + idnet 0.1407 → ENSEMBLE **0.0930 (WORSE than best single 0.0827)**. Equal-weight ensembling of UNEQUAL members hurts (weak members pollute the APCER@1% tail; AUC ~flat 0.9724→0.9726). **Lesson: ensemble needs COMPARABLE-quality members or weighting, not mere diversity.** `scripts/eval_lodo_ensemble.py` (logit-space fusion).
**Test B [running] — comparable members:** seed-42/123 @896 + prod728 (same winning recipe, seed+resolution diversity). These are all-5-types production (deployable) → OOD benefit is a research-backed BET, not LODO-validatable offline (no holdout). Proper LODO test would need per-fold seed/res members (future). Deployable ensemble = logit-blend of the 3 comparable members → private candidate.

**Result B — seed-ensemble, BENIN (2026-06-16):** seed-42 0.0827, seed-123 **0.1932** (2.3× variance!), output-ensemble 0.1074. **BENIN is HIGH-VARIANCE across seeds → 0.0827 is seed-lucky; true expected ~0.13.** Output-ensemble REDUCES variance (avoids the 0.1932 unlucky outcome) but doesn't beat the lucky single. **Reframe: ensembling here = robustness (avoid unlucky), not mean-gain — valuable for the one-shot private.** 5-fold 0.020 is optimistic on BENIN. → test weight-space SOUP (research #1 OOD lever) vs output-ensemble.

**Result C — weight-SOUP, BENIN (2026-06-16):** soup(seed-42,seed-123) = 0.1427 — WORSE than output-ensemble (0.1074) AND best single (0.0827). Seeds' minima not mode-connected on the high-variance hard fold → weight-avg lands worse. **ENSEMBLE CONCLUSION: on high-variance hard folds, neither output-ensemble nor soup beats the best single; both only reduce variance (robustness). Same-recipe seed/soup diversity is insufficient (correlated + unequal).** GUINEA confirms high seed-variance (seed-42 0.0375 vs seed-123 0.0785). HONEST robust 5-fold accounting for seed-variance ≈ 0.04 (BENIN~0.138, GUINEA~0.058 avg) vs seed-lucky 0.020. Deliverable-B = output-ensemble of seeds (robust, avoids unlucky). → last ensemble angle: BACKBONE diversity (ConvNeXt-V2, decorrelated errors).

**Result D — GUINEA seed-ensemble + FINAL (2026-06-16):** GUINEA seed-42 0.0134, seed-123 0.0785 (5.9× variance), output-ens 0.0226. Both hard folds high seed-variance; ensemble between (variance-reduction). **ROBUST ENSEMBLE 5-fold ≈ 0.027** (EGYPT .0016 + GUINEA .0226 + BENIN .1074 + MOZ .0016 + MAUR .0009)/5 vs single-lucky 0.020 / single-avg ~0.038. **Deliverable-B = multi-seed output-ensemble = robust ~0.027 (avoids seed-luck downside) — the private-submission choice.** MAIN EXPLORATION COMPLETE. Next: novel contributions (#4) for robust BENIN, captured-attack data (axis-2), private release.

---

## Iteration 12 — Stronger backbone: DINOv2 ViT-Large @896  [RUNNING] ★★★
**Hypothesis (goal #1 backbone + #5 diversity):** Base ViT-B BENIN floor 0.0827 (high-variance) is the bottleneck. A stronger backbone (ViT-Large, better DINOv2 features) may generalize to unseen BENIN better.
**Method:** vit_large_patch14_reg4_dinov2 + partial-unfreeze last-2 + FDA + 896 + tail_margin (eff_bs 32 via accum). Gate: BENIN+EGYPT.
**Result (2026-06-16): ★ BREAKTHROUGH. Large BENIN ep1 = 0.0165 vs Base 0.0827 (−80%, 5×!).** EGYPT 0.0035 (vs 0.0016, tiny cost). The stronger backbone CRUSHES the bottleneck fold. Projected Large 5-fold LODO ≈ 0.006 (3× better than Base 0.020). → run full Large 5-fold (NEW winning backbone). NOTE: enabled grad_checkpointing unnecessarily (VRAM ample) → ~2× slow; remaining folds run without it.
