# Plan v3 — UNSEEN-TYPE generalization (the real prize axis)

## Why this pivot (organizer intel, docs + memory project-organizer-intel)
- Private test = **2 doc types UNSEEN in train AND public** + **non-synthetic CAPTURED**
  examples. Public = born-digital, same 5 types as train → chasing public = overfitting
  the wrong distribution (the 20-teams-at-0.0 trap).
- Our model's core failure: it treats "the 5 FREUID types' appearance" as the genuine
  reference, so unseen-type genuine docs get flagged as attack (FantasyID: 78% of
  physical-genuine flagged). To win, the model must learn **type-INVARIANT** fraud cues.

## The lever: massive document-type diversity in training
Teach "genuine" and "fraud" across as many doc types/countries as possible so neither
class is tied to a specific layout. Available datasets in data/raw (copying):
| dataset | imgs | types | labels | role |
|---|---|---|---|---|
| FREUID | 69,352 | 5 (African/Asian) | genuine + attack | primary |
| **IDNet** | ~big | **20 countries** | **positive + fraud5_inpaint + fraud6_crop_replace** | **genuine+fraud type diversity (crown jewel)** |
| DocXPand-25k | 24,994 | 9 templates | genuine only | genuine-type diversity (+ SBD pseudo-attacks) |
| BID | 57,600 | 3 (Brazilian CPF/RG/CNH) | genuine only | genuine-type diversity (+ SBD) |
| FantasyID | 3,284 | 13 | genuine(physical) + GenAI attack | type + capture diversity |
| DLC-2021 | small | EU | recapture attacks | capture-style |

IDNet's fraud types (inpaint-and-rewrite, crop-and-replace) MATCH the private test's
GenAI/manipulation attacks → directly on-target.

## NEW validation protocol (the key change)
Stop using random-val (leakage→0.0) and public (born-digital). Validate by **leave-whole-
doc-type(s)-out across the full union**, holding out types from a DIFFERENT source than
training when possible — this is the honest analogue of the private "2 unseen types".
- Primary: train on {FREUID + IDNet + DocXPand}, validate on {BID + FantasyID} (entirely
  unseen types AND unseen sources + captured) → strongest unseen-type+capture proxy.
- Secondary: leave-one-FREUID-type-out (as before) for continuity.
- Metric: official FREUID Score on the held-out union.

## Anti-shortcut requirement
Genuine-only datasets (DocXPand, BID) must NOT become "this source = genuine" shortcuts.
Apply SBD self-blended pseudo-attacks (data/sbd.py) to them so each appears in BOTH
classes. Balance per-source genuine/attack counts.

## RESULTS LOG
- Adapters built (external.py): IDNet (positive+fraud, per-country), DocXPand (9 tpl
  genuine), BID (3 Brazilian genuine). manifests/union_v3.parquet = FREUID + IDNet(ready)
  + DocXPand(15k) + BID(15k) = 153,160 imgs, 20 doc_types, 88k genuine / 65k attack.
  FantasyID held out entirely as the unseen-type+capture external test.
- **D2 launched**: DTC-DINOv2 518 & 378, sbd_p=0.2 (pseudo-attacks on genuine-only
  sources), heavy_recapture=true, leave-4-types-out val (EGYPT, idnet_RUS, docxpand_PP,
  bid_CNH). Key metric = FantasyID xeval (FREUID-only baseline = 0.97 / AUC 0.586).
  IDNet extraction continues (more countries → larger union for later runs).

- **D2 (naive union, FREUID+IDNet-full+DocXPand+BID) FAILED**: FantasyID xeval 0.99
  (AUC **0.4645 = below chance**, worse than FREUID-only 0.97/0.586); public dist became
  muddled (22.8% mid-zone vs 0.8%). Root cause: **inconsistent fraud semantics** — FREUID
  attacks (physical/GenAI) + IDNet attacks (synthetic inpaint/crop) + SBD (blend) are 3
  different manipulation kinds → no coherent fraud boundary. Same failure mode as E2.
  Also: heterogeneous held-out val (EGYPT+RUS+docxpand+bid) didn't correlate with
  FantasyID xeval → bad checkpoint selection.
- **D3 (genuine-only-external) launched**: external datasets used as GENUINE only
  (IDNet-positive + DocXPand + BID), fraud = FREUID real attacks + SBD on ALL genuine
  (consistent synthetic fraud). 129k imgs, 22 types, EGYPT held out for ckpt selection
  (consistent semantics). Tests: does broadening the GENUINE manifold stop unseen-type
  genuine from being flagged (the FantasyID 78% failure) WITHOUT muddling fraud?

- **D3 (genuine-only-external) PARTIAL→reject**: broke the genuine shortcut (FantasyID
  physical-genuine flagged 78%→32%!) but LOST attack recall (GenAI attacks detected only
  23%); FantasyID AUC 0.44. Genuine-biased.
- **D4 (attack-upweighted) reject**: swung back — physical-genuine 99% flagged, GenAI
  98% detected, AUC 0.51. Attack-biased. EGYPT 0.2515 (vs FREUID-only 0.117).
- **KEY CONCLUSION (D2/D3/D4)**: class-balance tuning only SLIDES the global bias; it does
  NOT create separation (FantasyID AUC stuck ~0.5 across all). External data
  (IDNet/DocXPand/BID) has different capture/layout/attack distributions → mixing pulls
  the model between distributions, never beats FREUID-only on any validation axis. **Thrust
  A (massive external data) is a dead end.**
- **PIVOT → Thrust B (FREUID-centric capture robustness)**: private attacks are FREUID's
  own style on 2 unseen FREUID-family types (FREUID-only already separates these on LODO,
  EGYPT AUC~0.95); the real private risk is the CAPTURE shift. D5 = FREUID-only 518 +
  heavy_recapture (no external), EGYPT fold, vs FREUID-only 518 EGYPT baseline 0.117.

- **D5 (FREUID-only 518 + heavy_recapture) reject**: hurt BOTH axes — EGYPT 0.117→0.2383,
  FantasyID AUC 0.586→0.52. Synthetic recapture aug can't mimic real physical capture and
  only adds noise to clean data (confirms E1). Thrust B (synthetic capture aug) dead too.
- **CONVERGENCE (after D2–D5 + all earlier)**: NOTHING beats FREUID-only DTC-DINOv2-518 on
  the validation axes. Only 518px + conservative TTA help. External data, heavy aug, and
  loss tricks all regress ≥1 axis. → Deployment model = FREUID-only 518 all-types + TTA.
  Next genuine lever = strengthen the DTC novel method itself (thrust D), not more data/aug.
- **D6 (DTC localization head)**: add a spatially-grounded trace-inconsistency head
  supervised by TraceMix paste masks → more discriminative type-invariant signal + paper
  localization figure. Test on EGYPT vs 0.117 baseline.

- **D6 (DTC localization head) no-gain**: EGYPT 0.195 (vs FREUID-only 0.117), FantasyID
  AUC 0.557 (vs 0.586). Paper-figure value but no detection gain. Confirms convergence.
- **Key cross-experiment fact**: ALL models (FREUID-only + D2–D6) score FantasyID AUC
  0.44–0.59 ≈ chance → our DINOv2+trace forensic features CANNOT separate FantasyID's
  clean GenAI face-swap attacks. Private has GenAI attacks too → real risk.
- **D7 (frozen CLIP semantic stream) launched**: add a frozen CLIP ViT-B/16 stream
  (FatFormer-lineage: CLIP generalizes to unseen GAN/diffusion) to catch GenAI
  manipulation the forensic features miss. The one orthogonal axis untried. Tests:
  does CLIP raise FantasyID AUC above ~0.5 (separate GenAI attacks)? + EGYPT non-regress.

- **D7 (CLIP semantic stream) no-gain**: FantasyID AUC 0.538 (best separation +0.069 but
  still ~chance, below FREUID-only 0.586); EGYPT 0.224. CLIP doesn't crack FantasyID's
  GenAI attacks either.
- **FINAL CONVERGENCE (after D2–D7 + all earlier ~30 iterations)**: NOTHING beats
  FREUID-only DTC-DINOv2-518 + conservative TTA on the validation axes. FantasyID xeval is
  an UNFAIR cross-dataset proxy (ALL architectures ~chance AUC 0.5 → unbridgeable pipeline
  gap, not a fixable signal). The fair private proxy is FREUID LODO, where FREUID-only 518
  is clearly best (EGYPT 0.117 / AUC 0.95, BENIN 0.152). DEPLOYMENT = FREUID-only 518
  all-types (exp_final518_all_s42, public 0.2576) + conservative TTA. RESERVE = D3
  genuine-diverse model (in case private penalizes physical-genuine false-flagging:
  78%→32%). Search space exhausted → consolidate to deployment + paper + await private.

- **D8 (728px) BREAKTHROUGH — convergence was PREMATURE**: EGYPT 0.117(518)→**0.034**(728,
  ep3, AUC 0.994). Resolution lever NOT saturated; pushing 378→518→728 keeps breaking the
  EGYPT wall (0.21→0.117→0.034). Confirms the meta-lesson: "what the model sees" (resolution)
  is THE lever. BENIN 728 launched to confirm generalization. If it holds, 728 = new
  deployment recipe. Next: 728 all-types deployment retrain + maybe 896/1036px.

## Execution
- D1: build adapters (idnet, docxpand, bid) → unified manifest with `doc_type`,`source`.
- D2: leave-source-out split; train DTC-DINOv2-518 on the union; measure on held-out
  {BID+FantasyID}. Compare to FREUID-only (which scored FantasyID xeval 0.97).
- D3: add IDNet fraud (real manipulation attacks across 20 countries) → measure unseen
  fraud detection.
- D4: ablate which sources help the held-out unseen-type score; lock the recipe.
- Failure protocol per iteration (why prior failed before next).
- Submissions only on user instruction; public is NOT the selection metric anymore.
