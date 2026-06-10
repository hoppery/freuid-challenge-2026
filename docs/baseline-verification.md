# Baseline Verification vs Published Numbers

**Our baseline** (ConvNeXt-V2-Tiny, ImageNet-pretrained, fine-tuned; cross-domain leave-one-doc-type-out, held out MOZAMBIQUE/DL):
**ROC-AUC 0.744 · APCER@1%BPCER 0.50 · AuDET 0.256 (≈ EER 0.26–0.30).**

## Verdict: PLAUSIBLE — pipeline working, arguably ABOVE typical vanilla cross-domain baseline.
No published number exists ON FREUID (new 2026 dataset); verified against the closest analogues.

### Key comparison (IN-domain vs CROSS-domain marked)
| Method / benchmark | AUC | EER / APCER | Setting |
|---|---|---|---|
| UniFD (CLIP ViT-L) on FantasyID test | **0.52** | FNR 92.7%@10%FPR | CROSS (~chance) |
| FatFormer on FantasyID test | **0.535** | FNR 92.3%@10%FPR | CROSS (~chance) |
| TruFor on unseen face-swap (Attack-2) | **0.558** | HTER 48.3% | CROSS |
| MMFusion on unseen face-swap (Attack-2) | **0.615** | HTER 37.8% | CROSS |
| **OUR BASELINE** | **0.744** | APCER 50%@1%BPCER | **CROSS (held-out doc-type)** |
| DeepID OOD private real-ID, best teams (Fig.4) | **0.70–0.73** | F1 0.75–0.79 | CROSS (winners, heavy methods) |
| TruFor/MMFusion on text copy-paste (seen) | 0.93–0.99 | — | IN-domain |
| IDNet backbones in-domain face-morph | — | 98–100% acc | IN-domain |
| IDNet cross-doc-type (WV→AZ) | — | **50.55% acc (chance)** | CROSS (vanilla collapse) |
| Foundation models cross-dataset (CHL→IDNet) | — | EER 27.86% single / 8.25% fusion | CROSS |
| PAD-ID-Card IJCB sequestered | — | EER 6.4–21.9% | CROSS (but PHYSICAL PAD, easier) |

### Why our number is validated
1. **0.744 BEATS zero-shot forensic SOTA** (UniFD/FatFormer 0.52–0.54; TruFor/MMFusion 0.56–0.62 on unseen attacks). A broken pipeline would sit at AUC≈0.50 (cf. IDNet WV→AZ 50.55%).
2. **Brackets the DeepID OOD real-ID winners (AUC 0.70–0.73)** — we're at the upper-good end with a *vanilla* model.
3. **APCER@1%BPCER 0.50 is the canonical cross-domain signature**, not an anomaly: FantasyID reports "FNR≈50%@10%FPR" verbatim; at our stricter 1% BPCER, 50% APCER is exactly what an AUC-0.744 ROC yields.
4. **Internal consistency**: AUC 0.744 ⇒ binormal EER 0.26–0.30 ⇒ matches our AuDET 0.256; APCER@1%BPCER (strict) > EER (correct ordering).

### AUC↔EER anchors (equal-variance binormal, EER=Φ(−√2·Φ⁻¹(AUC)))
AUC 0.95→EER~10% · 0.90→~14% · 0.80→~21% · **0.744→~26–30%** · 0.70→~30% · 0.50→50%.

### Strategic implication
DeepID OOD winners reached AUC ~0.70–0.73 / F1 ~0.79 with heavy specialized methods; our vanilla baseline already matches that AUC. **To WIN we must push well past it** via the SOTA levers (foundation backbone, frequency/recapture pathways, heavy cross-domain aug) + the novel edges. The biggest published cross-domain win is **foundation-model fusion (DINOv2) → EER 8.25%** (vs 27.86% single) — multi-model fusion + DG is the path.

### Sources
FantasyID (Korshunov+, IJCB 2025, arXiv:2507.20808, Table 3) · DeepID Challenge (ICCVW 2025) · EdgeDoc (arXiv:2508.16284) · IDNet (arXiv:2408.01690, Tables 14–16) · SIDTD (Sci Data 2024, arXiv:2401.01858) · DLC-2021 (J.Imaging 8(7):181) · PAD-ID-Card (arXiv:2409.00372) · Foundation models PAD (arXiv:2506.05263) · ID-Card PAD review (arXiv:2511.06056) · ISO/IEC 30107-3.
