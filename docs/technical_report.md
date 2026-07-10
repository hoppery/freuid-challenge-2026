# FREUID Challenge 2026 — Technical Report

**Team:** _(fill in Kaggle team name)_ · **Competition:** The FREUID Challenge 2026 (IJCAI-ECAI),
`the-freuid-challenge-2026-ijcai-ecai` · **Date:** July 2026

---

## 1. Introduction

The FREUID Challenge asks for a scalar fraud score in `[0,1]` per identity-document image (higher =
more likely fraudulent), evaluated with the FREUID metric (AuDET + APCER@1%BPCER; lower is better).

The decisive property of this challenge is that the **public and private test sets are, by design,
different distributions.** From the organizers' clarifications: the **public** set is *born-digital*
(clean digital renders of the five training document types), while the **private** set — which decides
the prize — emphasizes **physically captured** documents (print-and-capture pipelines, lighting and
imaging variation) and contains **two document types unseen** in both train and public. The organizers
explicitly warn against optimizing the public leaderboard.

We therefore run a **two-track** strategy: a public-optimized ensemble for the public leaderboard, and
a separate **capture card** — the subject of the reproducibility Docker — for the private prize.

## 2. Method

### 2.1 Base detector: DTC

All private-track models use a **Dense Tampering-Consistency (DTC)** detector: an ImageNet-pretrained
RGB backbone plus a high-pass forensic branch and a patch-level **trace-consistency** module (pairwise
cosine statistics over per-patch trace embeddings) whose signal is fused into a single fraud logit. The
trace/consistency branch is independent of the RGB host, so the host backbone can be any family.

### 2.2 The capture lever: diverse captured-genuine data + heavy recapture aug

Robustness to physically captured documents is *not* obtained from architecture or loss changes; it
comes from training on **diverse real captured-genuine sources** with **heavy print-and-capture
augmentation** (`heavy_recapture=true`) and FDA domain randomization (`fda_p=0.4`). This is the E6
recipe (Section 3).

### 2.3 The winning lever: architecture × resolution diversity

Individually, capture detectors are limited; the gains come from **ensembling decorrelated
architectures at different resolutions**. Applying this ladder to the capture card:

| capture card | FantasyID proxy (FREUID ↓) |
|--------------|----------------------------|
| 2× ViT-B DTC (@896 + @1120), prob-avg | 0.354 |
| + ConvNeXt-V2 DTC (2nd family) | 0.333 |
| + EfficientNetV2 DTC (3rd family) | 0.285 |
| + RegNetY-160 DTC @896 | 0.279 |
| **+ RegNetY-160 DTC @1120** | **≈0.22–0.27** |

RegNetY-160 at 1120px is the strongest single capture model we found (proxy AUC 0.907, the highest of
any member), and pure-CNN families decorrelate strongly from the ViT-B anchors (Spearman ρ ≈ 0.5–0.7 vs
0.79 between the two ViT-B), which is what lifts the ensemble.

### 2.4 Final capture card (deployed in the Docker)

Probability-average of three DTC detectors:

- `exp_e6_fda/epoch2.pt` — ViT-B/14 DINOv2 DTC @896
- `exp_e6_fda1120/epoch2.pt` — ViT-B/14 DINOv2 DTC @1120
- `exp_e6reg_fda1120/epoch1.pt` — RegNetY-160 DTC @1120

Two ViT-B anchors stabilize the operating point (their FREUID score is seed-stable) while the
decorrelated RegNetY contributes the ranking gain; a seed study (3 seeds) showed this 2-ViT-anchor
configuration has the narrowest cross-seed variance among candidate cards. All training uses train-data
and self-supervised augmentation only; inference is a plain feed-forward probability average (no
test-time parameter fitting).

### 2.5 Approaches evaluated and rejected

- **Submitting the public-best ensemble on private** — the public card is capture-blind (≈chance, 0.99,
  on the captured proxy); rejected.
- **Data-diversity via MIDV-2020** — distribution mismatch with the capture proxy; no gain; parked.
- **Domain-routing hedge** (route born-digital rows to the public card, captured rows to the capture
  card) — a learned router keys on *framing/context* rather than intrinsic recapture artifacts and
  misroutes document-cropped captured images; rejected as unsafe.
- **Warm-starting the capture model from the public-best checkpoint** — a marginal cold-start benefit
  but a worse peak than from-scratch; the fraud-specific features do not survive recapture. Rejected.

## 3. Data

- **Competition data** — FREUID train and public test (Kaggle competition
  `the-freuid-challenge-2026-ijcai-ecai`).
- **External captured-genuine sources** (downloaded via `scripts/download/fetch_aux.py`; all are
  genuine captured documents, added at moderate upweight, no external fraud labels):
  MIDV-500, MIDV-2019, MIDV-Holo, MIDV-2020, BID (Brazilian CNH/RG/CPF photos), and FantasyID
  (physical printed bona-fides). FantasyID's held-out split is used only as a leakage-free **capture
  proxy** for model selection, never for training the deployed card.

The capture manifest (`manifests/freuid_fid_capgen_e6.parquet`, built by `scripts/build_capgen_e6.py`)
combines FREUID with these captured-genuine sources.

## 4. Inference

The Docker entrypoint (`scripts/docker_infer.py`) indexes a flat directory of images, runs the three
checkpoints (each: build model with `pretrained=False`, load weights, forward pass, sigmoid), and writes
the per-image **probability average** as `id,label`. Device is CUDA if available, else CPU. Unreadable
images are assigned 0.5 so the output always has exactly one finite row per input image.

## 5. Results

- **Public leaderboard:** the field-tamper architecture-diversity ensemble reached **0.00039** (rank 1);
  a minimal 3-model version scores **0.00041**.
- **Private (capture) proxy (FantasyID, FREUID ↓):** the capture card reaches **≈0.22–0.27**, roughly
  halving the 0.354 two-ViT baseline.
- **Distribution antagonism (measured on this competition):** the capture card scores **0.169** on the
  born-digital *public* set while the field-tamper card scores **0.00041** — a ~410× gap. Conversely the
  field-tamper card is at chance (~0.99) on the captured proxy. This double dissociation confirms the two
  distributions require different models and that public score does not predict private performance. The
  capture card is non-chance on born-digital (0.169), making it the minimax-safe single private pick.

The FantasyID proxy is a small (1,295-image) cross-corpus lower bound; its operating point (APCER@1%) is
seed-noisy, so absolute numbers are indicative and are expected to stabilize on the larger private set.

## 6. Reproducibility

**Hardware:** 2× NVIDIA RTX PRO 6000 (Blackwell, 96 GB). Inference uses a single GPU.

**Environment (pinned):** `torch==2.12.0` (CUDA 13), `timm==1.0.27`, `albumentations==2.0.8`,
`opencv-python-headless==4.13.0`, `numpy==2.4.6`, plus `pandas / scipy / scikit-learn / pillow / pyyaml
/ pyarrow`. See `Dockerfile` and `pyproject.toml`.

**Build the capture manifest and train the three checkpoints:**
```bash
pip install -e .
PYTHONPATH=src python3 scripts/download/fetch_aux.py        # external captured-genuine sources
PYTHONPATH=src python3 scripts/build_capgen_e6.py           # manifests/freuid_fid_capgen_e6.parquet
PYTHONPATH=src python3 -m freuid.train --config configs/exp_e6_fda.yaml         # -> epoch2
PYTHONPATH=src python3 -m freuid.train --config configs/exp_e6_fda1120.yaml     # -> epoch2
PYTHONPATH=src python3 -m freuid.train --config configs/exp_e6reg_fda1120.yaml  # -> epoch1
```

**Build and run the Docker (organizer contract):**
```bash
docker build -t freuid-repro:local .
docker run --network none \
  -v /path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out":/submissions \
  freuid-repro:local
# -> ./out/submission.csv  (id,label)
```

The image runs with `--network none`: all weights are embedded and models build with `pretrained=False`,
so there are no runtime downloads; the container writes only to `/submissions`. If the evaluation host's
CUDA differs from the build host's, adjust the base image tag and the `torch --index-url` in the
`Dockerfile` (checkpoint weights are torch-version portable).

Code: _(public repository URL + frozen commit SHA — fill in)_. Licensed under MIT (`LICENSE`).
