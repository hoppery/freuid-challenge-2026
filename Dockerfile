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
        opencv-python-headless==4.13.0 \
        "numpy==2.4.6" \
        pandas==3.0.3 \
        scipy==1.17.1 \
        scikit-learn==1.9.0 \
        pillow==12.2.0 \
        pyyaml==6.0.2 \
        pyarrow

# --- source ---
COPY src/ /app/src/
COPY scripts/docker_infer.py /app/scripts/docker_infer.py

# --- Frozen candidate weights, fetched at BUILD time (network is allowed during build) and baked into the
#     image — so the RUN stage needs no network (--network none). Weights exceed GitHub's 100 MB/file limit,
#     so they are hosted as GitHub Release assets. SHA-256 is verified, so the build FAILS LOUDLY on any
#     wrong, missing, or corrupt download.
#
#     CARD selects which frozen candidate the image runs (weights are unchanged either way — swapping the
#     card is a packaging change the rules allow after the freeze; pick after the private set is released):
#       capture (default) = 3-model DTC ensemble (~1 GB)  — private captured/physical bet
#       unseen            = 2-model FDA ensemble (~1.9 GB) — born-digital + 2 unseen document types hedge
#         docker build --build-arg CARD=unseen -t freuid-repro:unseen .
#     If your owner/repo/tag differ, also override WEIGHTS_BASE:
#       docker build --build-arg WEIGHTS_BASE=https://github.com/<owner>/<repo>/releases/download/<tag> -t freuid-repro:local .
ARG CARD=capture
ENV FREUID_CARD=${CARD}
ARG WEIGHTS_BASE=https://github.com/hoppery/freuid-challenge-2026/releases/download/weights-v1
RUN set -eu; \
    if [ "$FREUID_CARD" = "capture" ]; then \
        mkdir -p /app/checkpoints/exp_e6_fda /app/checkpoints/exp_e6_fda1120 /app/checkpoints/exp_e6reg_fda1120; \
        curl -fSL "$WEIGHTS_BASE/e6_fda_ep2.pt"        -o /app/checkpoints/exp_e6_fda/epoch2.pt; \
        curl -fSL "$WEIGHTS_BASE/e6_fda1120_ep2.pt"    -o /app/checkpoints/exp_e6_fda1120/epoch2.pt; \
        curl -fSL "$WEIGHTS_BASE/e6reg_fda1120_ep1.pt" -o /app/checkpoints/exp_e6reg_fda1120/epoch1.pt; \
        { echo "990703a5bee241747b527fb4b3aa202f94e5871c100ca6cd1d2cd51308580e48  /app/checkpoints/exp_e6_fda/epoch2.pt"; \
          echo "c61d1bd76080e4946b187b4330339062ad9107fb8847f73864412d497982714f  /app/checkpoints/exp_e6_fda1120/epoch2.pt"; \
          echo "e30b6bcfa7636cefa7ae633884ceed290cbee11bcedbff57a77413ffc35699d0  /app/checkpoints/exp_e6reg_fda1120/epoch1.pt"; \
        } > /tmp/w.sha256; \
    elif [ "$FREUID_CARD" = "unseen" ]; then \
        mkdir -p /app/checkpoints/exp_fdab12_all /app/checkpoints/exp_cnxfda_all; \
        curl -fSL "$WEIGHTS_BASE/fdab12_all_ep2.pt" -o /app/checkpoints/exp_fdab12_all/epoch2.pt; \
        curl -fSL "$WEIGHTS_BASE/cnxfda_all_ep2.pt" -o /app/checkpoints/exp_cnxfda_all/epoch2.pt; \
        { echo "e44aa6b107011f5663849ac655ef1aa197d1e82bac4be21eb875e69cd5a86b34  /app/checkpoints/exp_fdab12_all/epoch2.pt"; \
          echo "dd4f1738583557f2fcef2f3f33f767f5ac129a6ae403323716cfc5a90231632b  /app/checkpoints/exp_cnxfda_all/epoch2.pt"; \
        } > /tmp/w.sha256; \
    else echo "unknown CARD='$FREUID_CARD' (use capture|unseen)" >&2; exit 1; fi; \
    sha256sum -c /tmp/w.sha256; rm -f /tmp/w.sha256

ENTRYPOINT ["python3", "/app/scripts/docker_infer.py"]
