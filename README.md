# FREUID Challenge 2026 — ID-Document Fraud Detection

Reproducibility package for our submission to **The FREUID Challenge 2026 (IJCAI-ECAI)**.

The task: given an image of an identity document, output a fraud score in `[0, 1]`
(higher = more likely fraudulent). Scored by the FREUID metric (AuDET + APCER@1%BPCER).

## Strategy (two tracks)

The public and private test sets are, by the organizers' design, **different distributions**:

| Track | Test distribution | Our card |
|-------|-------------------|----------|
| **Public** leaderboard | born-digital renders, seen document types | field-tamper architecture-diversity ensemble (public rank-1) |
| **Private** prize | **physically captured** + **2 unseen** document types | **CAPTURE card** (this Docker) |

The prize is decided on the private (captured/physical) set, so the **Docker ships the CAPTURE card**.
We empirically confirmed the antagonism on this competition: the same 3-model capture card scores
**0.169** on the born-digital public set while the field-tamper card scores **0.00041** — a model that is
optimal on one distribution fails on the other (recapture destroys the digital forgery artifacts the
public model relies on). The capture card is the minimax-safe private pick (strong on captured, and
still non-chance on born-digital).

### The CAPTURE card (what the Docker runs)

Probability-average of three DTC (dense-tampering-consistency) detectors trained on FREUID +
diverse captured-genuine sources with heavy print-and-capture augmentation:

| checkpoint | backbone | resolution |
|------------|----------|------------|
| `checkpoints/exp_e6_fda/epoch2.pt`        | ViT-B/14 DINOv2 (DTC) | 896 |
| `checkpoints/exp_e6_fda1120/epoch2.pt`    | ViT-B/14 DINOv2 (DTC) | 1120 |
| `checkpoints/exp_e6reg_fda1120/epoch1.pt` | RegNetY-160 (DTC)     | 1120 |

Two ViT-B anchors (stable operating point) + a decorrelated RegNetY (architecture diversity, the
lever that roughly halved our capture-proxy error) give the most seed-robust card.

### Model weights (GitHub Release)

The three checkpoints (~1 GB total) exceed GitHub's 100 MB/file limit, so they are hosted as
**GitHub Release assets** and fetched by the `Dockerfile` at **build time** (SHA-256 verified) and
baked into the image — so `docker run --network none` needs no network. Release `weights-v1` assets:

| Release asset | → in image | SHA-256 (first 12) |
|---------------|------------|--------------------|
| `e6_fda_ep2.pt`        | `checkpoints/exp_e6_fda/epoch2.pt`        | `990703a5bee2` |
| `e6_fda1120_ep2.pt`    | `checkpoints/exp_e6_fda1120/epoch2.pt`    | `c61d1bd76080` |
| `e6reg_fda1120_ep1.pt` | `checkpoints/exp_e6reg_fda1120/epoch1.pt` | `e30b6bcfa763` |

If your fork uses a different owner/repo/tag, pass `--build-arg WEIGHTS_BASE=https://github.com/<owner>/<repo>/releases/download/<tag>`.

## Environment

Python ≥ 3.11. Pinned versions used for training and evaluation (see `Dockerfile`):
`torch==2.12.0` (CUDA 13), `timm==1.0.27`, `albumentations==2.0.8`, `opencv-python-headless==4.13.0`,
`numpy==2.4.6`, `pandas`, `scipy`, `scikit-learn`, `pillow`, `pyyaml`, `pyarrow`.

```bash
pip install -e .            # installs the `freuid` package (see pyproject.toml)
```

Hardware used: 2× NVIDIA RTX PRO 6000 (Blackwell, 96 GB). Inference runs on a single GPU; the Docker
entrypoint falls back to CPU automatically if no GPU is present.

## Data

- **Competition data** (FREUID train + public test): download per the Kaggle competition
  `the-freuid-challenge-2026-ijcai-ecai` into `data/raw/freuid/`.
- **External captured-genuine sources** used to build the capture manifest (downloaded by
  `scripts/download/fetch_aux.py`): MIDV-500, MIDV-2019, MIDV-Holo, MIDV-2020, BID (Brazilian IDs),
  and FantasyID (physical printed + GenAI). All are genuine captured documents added at moderate
  upweight; no external fraud labels are used.

Build the capture training manifest:
```bash
PYTHONPATH=src python3 scripts/build_capgen_e6.py     # -> manifests/freuid_fid_capgen_e6.parquet
```

## Training (reproduce the three capture checkpoints)

Each is a 3-epoch run; we select the epoch noted above (post-hoc, by the FantasyID capture proxy).

```bash
PYTHONPATH=src python3 -m freuid.train --config configs/exp_e6_fda.yaml         # ViT-B DTC @896  -> epoch2
PYTHONPATH=src python3 -m freuid.train --config configs/exp_e6_fda1120.yaml     # ViT-B DTC @1120 -> epoch2
PYTHONPATH=src python3 -m freuid.train --config configs/exp_e6reg_fda1120.yaml  # RegNetY DTC @1120 -> epoch1
```

Recipe (identical across the three, see the config files): `model_type=dtc`, `heavy_recapture=true`,
`fda_p=0.4`, `dtc_lambda=0.5`, `tracemix_p=0.5`, ImageNet-pretrained backbone, full/partial fine-tune.

## Inference

Local (flat directory of images → CSV):
```bash
FREUID_DATA_DIR=/path/to/images FREUID_OUT_DIR=/path/to/out \
  PYTHONPATH=src python3 scripts/docker_infer.py
```

## Docker (reproducibility contract)

```bash
docker build -t freuid-repro:local .
docker run --network none \
  -v /path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out":/submissions \
  freuid-repro:local
# -> ./out/submission.csv  (columns: id,label)
```

- `/data` — flat directory of images (`.jpeg .jpg .png .webp .bmp .tif .tiff`); `id` = filename stem.
- `/submissions/submission.csv` — one row per input image, `label` ∈ [0, 1] (higher = more fraudulent).
- Runs with **`--network none`**: all weights are baked into the image and models build with
  `pretrained=False`, so there are **no runtime downloads**. The container writes only to `/submissions`.
- If the evaluation host's CUDA differs from the build host's, adjust the base image tag and the
  `torch` `--index-url` in the `Dockerfile` accordingly (checkpoint weights are torch-version portable).

## Repository layout

```
src/freuid/           training + inference package (config, data, models, train, infer, metrics)
configs/              experiment YAMLs (the three capture configs are listed above)
scripts/              manifest builders, capture-proxy evals, docker_infer.py entrypoint
checkpoints/          model checkpoints (the three capture ckpts are embedded in the Docker image)
Dockerfile            reproducibility inference image
```

Licensed under the MIT License (`LICENSE`).
