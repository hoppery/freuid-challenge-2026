# Unseen-Robustness Plan (private test = 2 unseen doc-types, likely unseen attack families)

Research-backed (DomainBed/SWAD/SAGM/CORAL/SSDG/WiSE-FT/Model-Stock/FACT/TFS-ViT/SBI/Ojha/Surgical-FT) + our own experiment history. Public LB is in-domain (seen 5 types) → near-meaningless for ranking. The PRIVATE prize is decided by generalization to **unseen types** (axis 1) and likely **unseen attack families** (axis 2: physical/print — train is ~all digital).

Current honest proxy: tail_margin 896 5-fold **LODO-CV = 0.0200** (BENIN 0.0827 the hard fold). But LODO = same African-DL family → optimistic.

## ⚠ CROSS-CORPUS REALITY CHECK (2026-06-15)
FREUID-trained tail model → **IDNet (alien doc family): AUC 0.525, FREUID 0.987 = essentially RANDOM.** The LODO-0.020 model does NOT transfer to a genuinely different document family. BOTH proxies are biased: LODO over-optimistic (same family/fraud-type); IDNet over-pessimistic (different fraud GENERATOR + SYNTHETIC bona-fide = a real-vs-synth gap the private test won't have). **Private truth is between** — closer to LODO if private shares Microblink pipeline + GenAI fraud style; closer to the cliff if it brings new fraud TYPES (axis-2). **Implication: the model is overfit to FREUID's fraud signature → TRAINING-DATA DIVERSITY (varied fraud types/scripts/layouts) is the highest lever for truly-unseen, above DG losses.** Reprioritize: Tier-2 #9 (SBI synthetic forgery across templates) + Tier-3 (real recapture data + script diversity) move UP.

## ⚠ PUBLIC≠PRIVATE CONFIRMED (2026-06-15 submissions)
- bce-896 production: **public 0.0798**, LODO 0.0286.
- tail-896 production: **public 0.1302 (WORSE)**, LODO **0.0200 (BETTER)**.
→ The operating-point (tail_margin) model wins the UNSEEN proxy but loses the in-domain public LB (its score-reshaping helps the hard tail, hurts the saturated in-domain slice). **Picking the submission by public LB would select the WRONG model for the private prize.** Decision rule: choose the FINAL private entry by LODO (+cross-corpus), NOT public. Open question: is tail's LODO win real-private or proxy-overfit? → keep both candidates; consider a blend; let unseen-robustness work (diversity, ensemble) decide. Kaggle allows 2 final selections — likely submit one public-optimized (bce) + one unseen-optimized (tail/ensemble).

## ⚠ DIVERSITY-VIA-CORPUS = NEGATIVE (2026-06-16)
div896 (FREUID+IDNet 32k) → held-out FantasyID: **AUC 0.559** (vs FREUID-only 0.592 — NO gain, slightly worse) AND in-domain diluted (0.0466 vs 0.0000). → **Adding a heterogeneous synthetic corpus does NOT fix cross-corpus generalization** (different fraud generators + synth/real gap don't bridge). BUT cross-corpus (alien pipeline) is HARDER than the private test (same Microblink pipeline + fraud types, new layouts only) → LODO 0.020 stays the better private proxy; cross-corpus is over-pessimistic. **PIVOT: drop corpus-mixing for diversity; the host-confirmed lever is CAPTURE robustness (axis-2 recapture aug, label-agnostic) on the same family — implement next.** Generic OOD levers (SWAD/soup/WiSE-FT) still apply for same-pipeline LODO. divfid896 (FREUID+FantasyID→IDNet) **CONFIRMED negative: AUC 0.491 (baseline 0.525, below random).** Both directions negative → corpus-mixing for cross-corpus generalization is DEAD. (Caveat: alien-corpus is harder than the same-pipeline private; LODO 0.020 is the realistic proxy.) → committed to axis-2 recap aug + same-pipeline DG levers (SWAD/soup/WiSE-FT).

## DATA to add (research-verified, license-clean)
- **FantasyID** (CC-BY-4.0, ALREADY downloaded `manifests/fantasyid.parquet` 3284): Arabic/CJK/Cyrillic/Indic scripts + REAL print-recapture bona-fide → biggest script-diversity + axis-2 value. Underused — fold into 896 training.
- **DLC-2021** (CC-BY-SA-2.5, was downloading): REAL analog-hole (laminated/copy/screen recapture).
- **MIDV-Holo** (CC-BY-SA-2.5): real print/photo-swap/screen attacks.
- Gated/ambiguous (private-train only): KID34K (Korean+recapture), SIDTD (Cyrillic/MRZ + print subset).
- Bona-fide diversity: MIDV-LAIT (Arabic/Thai/Indic), MIDV-500/2020 (Cyrillic/MRZ).

## ⚠ RECAP AUG GATE RESULT (2026-06-16) — it's a TRADEOFF, not free
recap tier (capture-realistic) vs tail(medium) LODO: EGYPT 0.0026 vs 0.0016 (≈free), **BENIN 0.1309 vs 0.0827 (recap +58% WORSE)**. recap noise hurts the fine digital cue on the hard fold (BENIN peaked ep1 then collapsed, never reached tail's ep3 0.083). → recap COSTS digital-attack perf to bet on captured-attack robustness (host says private leans captured → may be worth it, but **unverifiable without captured validation data**). DECISION: do NOT blindly adopt full recap. Need a CAPTURED validation set (DLC-2021/MIDV-Holo, research-found, CC-BY-SA) to (a) measure if recap actually helps captured, (b) tune the digital↔captured tradeoff, or use recap only as a weighted ENSEMBLE member. Hold full recap 5-fold pending that. Gentler recap (between medium and this tier) also worth trying. → USER DECISION POINT.

## AXIS-2 AUG (print/recapture, 896-safe) — CRITICAL RULE
Apply recapture transforms **label-agnostically to BOTH classes** (never gate on label → else "recapture⇒fraud" shortcut). Gentle ranges, wrap in `A.OneOf` p≈0.4 leaving a clean path; randomize params per-sample; SARE variance penalty across artifact groups; per-family LODO-AUC monitor. (Principled version of "medium helps / heavy collapses".) Families: perspective→gamma/tone→halftone(α0.05-0.18)→mild blur(≤7)→downscale(≥0.5)→ISO/Gauss noise→double-JPEG(Q≥60). AVOID: blur≥15, downscale<0.4, JPEG<40, halftone α>0.35, FIXED moiré freq/angle (=doc-type shortcut).

## Framing facts (from research)
- **DomainBed verdict:** under honest model selection, fancy DG losses rarely beat tuned ERM; the survivors are **flat-minima/weight-averaging (SWAD)**, **CORAL** (ERM-safe), **foundation-frozen features**. → explains our adversarial-GRL (DTC) failure.
- **Our metric (APCER@1%BPCER) is reported nowhere** in the literature (all use AUC/HTER). A method can win AUC and lose at the 1% threshold. → gate EVERYTHING on LODO APCER@1%, not AUC.

## TIER 1 — cheap, reliable, single-run / post-hoc (DO FIRST)
1. **Weight averaging: SWAD + EMA.** Most reliable DG method; flat minima → steadier OOD tail. We have EMA; add SWAD dense averaging over the val-loss window. LayerNorm ViT = no BN recompute (free). ~50 lines.
2. **WiSE-FT / Model-Stock α-interpolation (post-hoc, free, reversible).** Interpolate fine-tuned ↔ DINOv2 init weights; only last-2 blocks moved → clean merge. Sweep α∈[0.3,1.0], pick min LODO APCER@1%.
3. **Greedy model soup** of existing fold/seed checkpoints — OOD gain ~4.5× > in-distribution; provably ≥ best-on-val. Near-free.
4. **Surgical-FT block diagnostic.** Unseen type = appearance shift → research says EARLY blocks may generalize better; our last-2 choice is least-motivated. One run: early-block vs last-2 vs LN-only.

## TIER 2 — one slot each (strong evidence)
5. **SAGM** (sharpness-aware gradient matching) — single-model, domain-count-agnostic, +OOD; stacks with SWAD.
6. **CORAL** — the only ERM-safe alignment; align covariances across doc_type/source groups.
7. **FACT** — upgrade FDA to continuous amplitude-mix + EMA phase-consistency co-teacher → attacks unseen-script/layout (content) shift. Amplitude-only = cue-safe.
8. **TFS-ViT** — token-level feature stylization (the correct MixStyle for a LayerNorm ViT; MixStyle/DSU/EFDMix are BN-CNN-only). Adds style diversity (random tokens, low fraction).
9. **Synthetic-forgery aug (SBI-style) across ALL templates** — self-blend (soft seam≈GenAI) + cross-doc Crop&Replace (hard seam≈physical); statistical mismatch is load-bearing; locality supervision to avoid template shortcut.
10. **SSDG single-side adversarial** (the FIX for our DTC failure): align bona-fide ONLY across domains; push fakes apart; group fakes by attack_type not database.

## TIER 3 — unseen ATTACK families (axis 2)
11. High-res-SAFE recapture/print aug (gentle moiré/halftone/JPEG-double at 896) — extend `medium`; LODO-gated so it can't destroy the cue (heavy aug = our AUC-0.50 failure).
12. Real physical/print ID-attack data if available (research pending: SIDTD/DocXPand/MIDV-Holo).

## TIER 4 — final submission
13. **Diverse ensemble + LOGIT-space fusion + threshold picked on held-out UNSEEN doc-type** (never in-domain). Directly attacks APCER@1% (tail false-accepts cancel across diverse members).
14. **TTA — careful, optional (research-refined recipe).** DINOv2=LayerNorm (no BN) → on the STABLE side of TTA; BN-stat methods N/A. Order by safety×gain: **T3A** (backprop-free prototypes, 2 classes, knob M∈{5,20,50}) and **LAME** (parameter-free output smoothing) FIRST — structurally can't collapse. Then **SAR** (LN-affine only, freeze ViT blocks 9-11, entropy filter E0=0.4·ln2≈0.277, SAM lr1e-3, EMA-reset, **episodic per domain**) as the gradient option. **AVOID bare TENT (collapse) + MEMO.** Collapse guard: monitor predicted fraud-fraction, abort→source if it leaves a sane band; ship per-fold **min-risk(source, adapted)**. CAVEAT: Kaggle scores raw probs & AuDET is rank-only → monotone TTA rescale doesn't change AuDET; TTA only helps if it improves SEPARABILITY. On ~2 unseen domains per-fold deltas ≈ n=1 noise → LODO-gate hard. Threshold (1%BPCER) must be recomputed by scoring labeled bona-fides through the ADAPTED model (only matters for the APCER report, not the AuDET submission).

## SKIP (research-deflated / our failures)
IRM, V-REx, Fish, AND-Mask, MMD, plain PCGrad; adversarial GRL on both classes (our DTC); heavy recapture / RandConv / full AdaIN / EFDMix (cue-destroying); reconstruction one-class (heterogeneous bona-fide); resolution > 896 (interp degrades).

## Validation guardrail
Leave-one-doc-type-out gate on BOTH held-out AUC AND APCER@1%BPCER before promoting anything (would have auto-caught the heavy-aug collapse). Plus cross-corpus (FREUID→IDNet) for a truly-alien read. Any logit rescaling (TTA/whitening/calibration) can move the 1% threshold even when AUC looks fine — always recompute.

## Execution order
- **Now:** cross-corpus eval (true-gap grounding) + tail production retrain (deployable, → submit).
- **Next (Tier 1):** WiSE-FT α-sweep + greedy soup (post-hoc, cheap) → then SWAD (training).
- **Then (Tier 2):** SAGM, FACT one at a time, each LODO + cross-corpus gated.
- **Axis 2:** add high-res-safe print aug + SBI; bring real print/physical data if downloadable.
- **Final:** diverse logit-fusion ensemble, threshold on unseen holdout; TTA only if it passes the gate.
