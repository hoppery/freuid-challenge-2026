# FREUID Challenge 2026 — ID-Document Fraud Detection

Reproducibility package for our submission to **The FREUID Challenge 2026 (IJCAI-ECAI)**.

The task: given an image of an identity document, output a fraud score in `[0, 1]`
(higher = more likely fraudulent). Scored by the FREUID metric (AuDET + APCER@1%BPCER).

> **⚠️ Building/running the Docker image?** It targets **CUDA 13** and needs a host **NVIDIA driver
> ≥ 580.65.06** to use the GPU. Pass **`--gpus all`** and confirm the run log shows `device=cuda`.
> Full details in **[Host runtime requirements](#docker-reproducibility-contract)** below.

## Strategy

The private test set is **born-digital identity documents that include document types not present in the
training set**. Our submitted model is built for exactly that: it must generalize across document types
— including unseen ones — rather than lean on the appearance of the types it was trained on.

Two design choices target cross-type generalization:

- **Fourier domain adaptation (FDA)** during fine-tuning. Randomizing the low-frequency (global colour /
  layout) content decouples the fraud signal from document-type appearance, so an unseen-type genuine
  document is not flagged just for looking unfamiliar.
- **Architecture diversity.** A transformer and a convolutional backbone rely on different cues and fail
  on different document types; averaging the two holds up on unseen types where either alone would slip.

### The unseen-FDA card (what the Docker runs)

Probability-average of two FDA-fine-tuned detectors:

| checkpoint | backbone | resolution |
|------------|----------|------------|
| `checkpoints/exp_fdab12_all/epoch2.pt` | ViT-L/14 DINOv2 (FDA) | 896 |
| `checkpoints/exp_cnxfda_all/epoch2.pt` | ConvNeXt-V2-L (FDA)   | 896 |

Both are full fine-tunes with FDA augmentation and a tail-margin operating-point loss (which optimizes
the low-BPCER tail the FREUID metric scores). The ViT-L (DINOv2) and ConvNeXt-V2-L backbones are
decorrelated, so the pair stays robust on document types neither has seen.

### Model weights (GitHub Release)

The two checkpoints (~1.9 GB total) exceed GitHub's 100 MB/file limit, so they are hosted as **GitHub
Release assets** and fetched by the `Dockerfile` at **build time** (SHA-256 verified) and baked into the
image — so `docker run --network none` needs no network. Release `weights-v1` assets:

| Release asset | → in image | SHA-256 (first 12) |
|---------------|------------|--------------------|
| `fdab12_all_ep2.pt` | `checkpoints/exp_fdab12_all/epoch2.pt` | `e44aa6b10701` |
| `cnxfda_all_ep2.pt` | `checkpoints/exp_cnxfda_all/epoch2.pt` | `dd4f17385835` |

Full SHA-256 in `WEIGHTS.md`. If your fork uses a different owner/repo/tag, pass `--build-arg WEIGHTS_BASE=https://github.com/<owner>/<repo>/releases/download/<tag>`.

## Environment

Python ≥ 3.11. Pinned versions used for training and evaluation (see `Dockerfile`):
`torch==2.12.0` (CUDA 13), `timm==1.0.27`, `albumentations==2.0.8`, `opencv-python-headless==4.13.0.92`,
`numpy==2.4.6`, `pandas`, `scipy`, `scikit-learn`, `pillow`, `pyyaml`, `pyarrow`.

```bash
pip install -e .            # installs the `freuid` package (see pyproject.toml)
```

Hardware used: 2× NVIDIA RTX PRO 6000 (Blackwell, 96 GB). Inference runs on a single GPU; the Docker
entrypoint falls back to CPU automatically if no GPU is present.

## Data

- **Competition data** (FREUID train + public test): download per the Kaggle competition
  `the-freuid-challenge-2026-ijcai-ecai` into `data/raw/freuid/`.
- The submitted model is trained on the **FREUID train split only** (`manifests/freuid.parquet`).
  No external datasets are used.

## Training (reproduce the two checkpoints)

Each is a 6-epoch run; we select **epoch 2** of each.

```bash
PYTHONPATH=src python3 -m freuid.train --config configs/exp_fdab12_all.yaml  # ViT-L/14 DINOv2 FDA @896 -> epoch2
PYTHONPATH=src python3 -m freuid.train --config configs/exp_cnxfda_all.yaml  # ConvNeXt-V2-L FDA @896  -> epoch2
```

Recipe (both, see the config files): `model_type=rgb`, `fda=true` (Fourier domain adaptation,
`fda_beta=0.12`), `loss=tail_margin` (tail-quantile 0.99), full fine-tune (`unfreeze_last_k=2` on the
ViT), DINOv2 / ImageNet-pretrained backbone, `img_size=896`, `lr=5e-5`.

## Inference

Local (flat directory of images → CSV):
```bash
FREUID_DATA_DIR=/path/to/images FREUID_OUT_DIR=/path/to/out \
  PYTHONPATH=src python3 scripts/docker_infer.py
```

## Docker (reproducibility contract)

> ## ⚠️ HOST RUNTIME REQUIREMENTS — please read before building
>
> The image is built for **CUDA 13** (`torch` cu130). To run inference **on the GPU**, the evaluation
> host must provide:
>
> | Requirement | Value | Why it matters |
> |-------------|-------|----------------|
> | **NVIDIA driver** | **≥ 580.65.06** | The CUDA 13 minimum. On an **older driver the container cannot use the GPU** and falls back to CPU — too slow for the 6 h limit. Check with `nvidia-smi` (must show `CUDA Version: 13.x` or higher). |
> | **GPU passed to container** | `--gpus all` | …or `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all`, or `--device nvidia.com/gpu=all` on a CDI setup. Confirm the run log prints **`device=cuda`**. |
> | **Base image** | `nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04` | Provides the in-container CUDA 13 runtime. |
> | `--shm-size` | **not needed** | The loader decodes in a background thread (no multiprocessing), so the default 64 MB `/dev/shm` is fine. |
>
> A single **A100** finishes the hidden test in roughly **2–5 h** on the GPU. **If your eval host runs an
> older CUDA/driver**, change the base image tag and the `torch` `--index-url` in the `Dockerfile` to match
> (checkpoint weights are torch-version portable).

```bash
docker build -t freuid-repro:local .
# GPU: if your host rejects `--gpus all` (e.g. a CDI setup), swap it for `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all` or `--device nvidia.com/gpu=all` (see the requirements table above).
docker run --gpus all --network none \
  -v /path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out":/submissions \
  freuid-repro:local
# -> ./out/submission.csv  (columns: id,label)
```

- `/data` — flat directory of images (`.jpeg .jpg .png .webp .bmp .tif .tiff`); `id` = filename stem.
- `/submissions/submission.csv` — one row per input image, `label` ∈ [0, 1] (higher = more fraudulent).
- Runs with **`--network none`**: all weights are baked into the image and models build with
  `pretrained=False`, so there are **no runtime downloads**. The container writes only to `/submissions`.
  (The entrypoint auto-detects the GPU and falls back to CPU only if none is provided.)

## Repository layout

```
src/freuid/           training + inference package (config, data, models, train, infer, metrics)
configs/              experiment YAMLs (the two unseen-FDA configs are listed above)
scripts/              docker_infer.py (the Docker inference entrypoint)
checkpoints/          model checkpoints (the two unseen-FDA ckpts are embedded in the Docker image)
Dockerfile            reproducibility inference image
```

Licensed under the MIT License (`LICENSE`).
