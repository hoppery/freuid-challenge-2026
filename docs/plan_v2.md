# New Training Plan v2 — targeting the REAL bottleneck (2026-06-13)

## A. Failure analysis of the loop so far (iters #1–#28)

### What we optimized vs what matters
We ran 28 iterations, almost all judged on **leave-one-doc-type-out (LODO)** folds.
LODO measures **doc-TYPE transfer within born-digital data**. But:
- LODO mean ≈ **0.05** (518px+TTA), yet **public = 0.26**. A 5× gap.
- The gap is NOT doc-type: even our hardest fold (BENIN 0.15) is far below public 0.26.
- **Root cause: a domain axis LODO cannot see — capture style.** Train is **99.97%
  born-digital** (only 20/69,352 are physical print-and-capture). The public/private
  test is officially "hybrid synthetic + physically printed-and-captured." The model has
  essentially never seen the print-and-capture domain that dominates the test.

### Why the clever methods failed (re-read through this lens)
- GRL / MixStyle / compactness / APCER-surrogate / EMA — all operated on the **doc-type**
  axis or the loss, none touched the **capture-style** axis → they polished a metric
  (LODO) that doesn't reflect the real gap.
- 518px helped because higher resolution preserves the recapture micro-texture that DOES
  transfer — a partial, accidental hit on the capture axis.
- TTA helped because it adapts to the **test distribution at inference** — the only lever
  that actually touches the real (capture-shifted) test domain. This is why TTA is our
  most reliable win and should be central, not peripheral.

### The under-augmentation bug
`build_dtc_transforms` (the winning DTC path) applies only geometric augs + TraceMix.
It OMITS the heavy `recapture_transforms` block (multi-stage JPEG, moiré, down/up-scale,
sensor noise, blur). So the strongest model is trained on near-clean born-digital inputs
and is unprepared for print-and-capture. **This is the single most likely fixable cause
of the 0.05→0.26 gap.**

## B. The right validation protocol (new)
LODO doc-type folds are necessary but insufficient. Add a **capture-style held-out**:
- **FREUID-train → FantasyID-test** (manifests/fantasyid.parquet: 933 *physical
  printed-and-captured* bona-fides + 2,351 GenAI attacks; 13 doc types incl Arabic).
  This is the closest public analogue of the FREUID test's capture+GenAI shift.
- Judge every new lever on BOTH: (1) BENIN LODO (doc-type), (2) FantasyID cross-dataset
  (capture-style + GenAI). A lever must not regress either.

## C. New levers (ranked by expected payoff on the real gap)
1. **Heavy recapture aug in the DTC path** (fix the under-aug bug). Insert
   recapture_transforms into build_dtc_transforms. Cheap, directly simulates print-and-
   capture on BOTH classes. EXPECT: large cross-dataset (FantasyID) gain.
2. **FantasyID as capture-style data, UPWEIGHTED** (not naive concat like failed #9).
   Its 933 physical bona-fides are the ONLY real print-and-capture examples available;
   oversample them so they aren't diluted at 53k scale. Its GenAI attacks also target
   the BENIN missed-attack family (clean GenAI, 24.8% missed).
3. **TTA wired into inference** (currently eval-only). The validated −0.008…−0.046 LODO
   gain should apply at submission. Make it a first-class inference option.
4. **Capture-style consistency for TraceMix**: bias TraceMix toward stronger, more
   realistic print-and-capture chains (halftone, screen-grid) so the consistency head
   keys on real recapture multiplicity, not synthetic moiré.

## RESULTS LOG (plan_v2)
- **Baseline gap measured**: deployment model (in-domain 0.0, LODO 0.05) scores
  **FantasyID xeval 0.975 (AUC 0.586 = chance)**. TTA doesn't help (0.972).
  → capture-style gap is CATASTROPHIC, not subtle. Confirms thesis.
- **E1 (heavy recapture aug) FAILED both axes**: FantasyID 0.979 (no change), BENIN
  LODO 0.31 (regressed from 0.15). Diagnosis via score distribution: on FantasyID the
  model scores **78% of physical-genuine cards AS ATTACK** (median 0.966) and 88% of
  GenAI attacks as attack — it flags EVERYTHING non-born-digital. This is the
  domain-shortcut "recapture⇒fraud" failure. Synthetic recapture aug can't replicate
  REAL physical cards (halftone/paper/rephoto), so born-digital images + JPEG/moiré
  still don't look physical → shortcut survives.
- **Implication**: the only fix is REAL physical-capture GENUINE data. FantasyID's 933
  physical bona-fides are the unique source. #9's naive concat failed by dilution; E2
  reframes FantasyID as capture-style data and UPWEIGHTS it. Honest measurement needs a
  held-out FantasyID partition (no leakage).

- **E2 (FantasyID-upweighted joint) PARTIAL**: on leakage-free FantasyID-test (6 held-out
  FID types), AUC 0.586→0.688 (upweight) / **0.718 (upweight+heavyaug)**. Real physical
  data + synthetic aug now SYNERGIZE (heavy aug alone failed in E1). BUT FREUID-score
  stays ~0.95: APCER@1% tail still ~0.97 because the 6 held-out FID types are out-of-hull
  (same root cause as the LODO walls). Score dist: physical-bona still 68% flagged as
  attack on held-out types → shortcut broken for SEEN FID types only.
- **Caveats**: (a) FantasyID capture pipeline ≠ FREUID pipeline → this proxy may be
  unfairly hard vs the real private test. (b) TTA on the heavy-aug model produced NaN
  (LN instability under entropy grad) — needs a guard before use.
- **Net**: capture-style gap is confirmed real; real physical data is the lever (AUC
  +0.13); E2+heavyaug is a strictly-better DEPLOYMENT candidate (helps capture-shifted
  images, neutral in-domain). Only a Kaggle submission can confirm on the true
  distribution (awaiting user instruction).

- **E2 Kaggle submission: public 0.2845 — WORSE than FREUID-only single (0.2576).**
  Critical learning: FantasyID physical data + heavy aug HURT the actual public score,
  even though it improved the FantasyID-proxy AUC. → FantasyID's capture/GenAI pipeline
  is OFF-distribution for the FREUID public test; the 20×-upweight + heavy aug pulled the
  model off the FREUID manifold. Either (a) the public slice (7,821 imgs) is
  born-digital-dominant so the capture gap lives only in the unscored private 134,997, or
  (b) FantasyID style ≠ FREUID style. The FantasyID proxy MISLED us about the public dist.
  **Best public single = FREUID-only 518 all-types 0.2576 (unchanged).** Lesson for next:
  if FantasyID helps private at all, the dose must be light, not 20× + heavy aug.

## D. Execution order
- E1: DTC + heavy recapture aug — BENIN LODO + FantasyID cross-eval (vs current recipe).
- E2: E1 + FantasyID-upweighted joint training — same two metrics.
- E3: wire TTA into infer.py; re-measure FantasyID cross-eval with TTA.
- E4: combine winners; full-data deployment retrain; submit (on user instruction).
- Failure protocol unchanged: each E states why prior E failed before the next.
