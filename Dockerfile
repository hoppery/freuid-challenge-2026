# FREUID Challenge 2026 — reproducibility inference image (CAPTURE card, private track).
# Contract:  docker run --network none -v <flat_images>:/data:ro -v "$(pwd)/out":/submissions freuid-repro:local
#   reads /data (flat dir of images) -> writes /submissions/submission.csv (id,label = P(fraud)).
# All weights are BAKED IN and the model builds with pretrained=False, so NO network is used at run time.
#
# Base matches the training/eval environment (torch 2.12 / CUDA 13, Blackwell-class GPUs). If the evaluation
# host uses a different CUDA, change the base tag + the torch --index-url below to that CUDA (weights are
# torch-version portable). CPU-only also works (docker_infer.py auto-falls back), just slow.
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip libglib2.0-0 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- deps (pinned to the training environment). Network is available at BUILD time only. ---
RUN pip3 install --no-cache-dir --break-system-packages \
        torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
RUN pip3 install --no-cache-dir --break-system-packages \
        timm==1.0.27 \
        albumentations==2.0.8 \
        opencv-python-headless==4.13.0.92 \
        "numpy==2.4.6" \
        pandas==3.0.3 \
        scipy==1.17.1 \
        scikit-learn==1.9.0 \
        pillow==12.2.0 \
        pyyaml==6.0.3 \
        pyarrow==24.0.0

# --- source ---
COPY src/ /app/src/
COPY scripts/docker_infer.py /app/scripts/docker_infer.py

# --- The 3 CAPTURE-card checkpoints (~1 GB), fetched at BUILD time (network is allowed during build) and
#     baked into the image — so the RUN stage needs no network (--network none). Weights exceed GitHub's
#     100 MB/file limit, so they are hosted as GitHub Release assets. SHA-256 is verified, so the build
#     FAILS LOUDLY on any wrong, missing, or corrupt download.
#     If your owner/repo/tag differ, override WEIGHTS_BASE:
#       docker build --build-arg WEIGHTS_BASE=https://github.com/<owner>/<repo>/releases/download/<tag> -t freuid-repro:local .
ARG WEIGHTS_BASE=https://github.com/hoppery/freuid-challenge-2026/releases/download/weights-v1
RUN set -eu; mkdir -p /app/checkpoints/exp_e6_fda /app/checkpoints/exp_e6_fda1120 /app/checkpoints/exp_e6reg_fda1120; \
    curl -fSL "$WEIGHTS_BASE/e6_fda_ep2.pt"        -o /app/checkpoints/exp_e6_fda/epoch2.pt; \
    curl -fSL "$WEIGHTS_BASE/e6_fda1120_ep2.pt"    -o /app/checkpoints/exp_e6_fda1120/epoch2.pt; \
    curl -fSL "$WEIGHTS_BASE/e6reg_fda1120_ep1.pt" -o /app/checkpoints/exp_e6reg_fda1120/epoch1.pt; \
    { echo "990703a5bee241747b527fb4b3aa202f94e5871c100ca6cd1d2cd51308580e48  /app/checkpoints/exp_e6_fda/epoch2.pt"; \
      echo "c61d1bd76080e4946b187b4330339062ad9107fb8847f73864412d497982714f  /app/checkpoints/exp_e6_fda1120/epoch2.pt"; \
      echo "e30b6bcfa7636cefa7ae633884ceed290cbee11bcedbff57a77413ffc35699d0  /app/checkpoints/exp_e6reg_fda1120/epoch1.pt"; \
    } > /tmp/w.sha256; \
    sha256sum -c /tmp/w.sha256; rm -f /tmp/w.sha256

# Run as a non-root user (uid 1000), matching the organizer's reference container. The baked weights
# and source are world-readable; the container writes only to the /submissions mount, so this also
# enforces the "no writes outside /submissions" verification requirement.
RUN useradd --create-home --uid 1000 runner
USER runner

ENTRYPOINT ["python3", "/app/scripts/docker_infer.py"]
