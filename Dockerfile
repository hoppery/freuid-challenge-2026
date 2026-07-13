# FREUID Challenge 2026 — reproducibility inference image (unseen-FDA card, private track).
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
    OMP_NUM_THREADS=4 \
    NO_ALBUMENTATIONS_UPDATE=1
# ^ albumentations otherwise tries to fetch its latest version at import; under --network none that
#   DNS lookup fails and prints a harmless UserWarning. Disabling it keeps the run log clean.

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

# --- The 2 unseen-FDA-card checkpoints (~1.9 GB), fetched at BUILD time (network is allowed during build)
#     and baked into the image — so the RUN stage needs no network (--network none). Weights exceed GitHub's
#     100 MB/file limit, so they are hosted as GitHub Release assets. SHA-256 is verified, so the build
#     FAILS LOUDLY on any wrong, missing, or corrupt download.
#     If your owner/repo/tag differ, override WEIGHTS_BASE:
#       docker build --build-arg WEIGHTS_BASE=https://github.com/<owner>/<repo>/releases/download/<tag> -t freuid-repro:local .
ARG WEIGHTS_BASE=https://github.com/hoppery/freuid-challenge-2026/releases/download/weights-v1
RUN set -eu; mkdir -p /app/checkpoints/exp_fdab12_all /app/checkpoints/exp_cnxfda_all; \
    curl -fSL "$WEIGHTS_BASE/fdab12_all_ep2.pt" -o /app/checkpoints/exp_fdab12_all/epoch2.pt; \
    curl -fSL "$WEIGHTS_BASE/cnxfda_all_ep2.pt" -o /app/checkpoints/exp_cnxfda_all/epoch2.pt; \
    { echo "e44aa6b107011f5663849ac655ef1aa197d1e82bac4be21eb875e69cd5a86b34  /app/checkpoints/exp_fdab12_all/epoch2.pt"; \
      echo "dd4f1738583557f2fcef2f3f33f767f5ac129a6ae403323716cfc5a90231632b  /app/checkpoints/exp_cnxfda_all/epoch2.pt"; \
    } > /tmp/w.sha256; \
    sha256sum -c /tmp/w.sha256; rm -f /tmp/w.sha256

# Runs as root (the image default). The entrypoint writes ONLY to the /submissions mount, which
# satisfies the "no writes outside /submissions" requirement; and root can always read the mounted
# read-only /data regardless of how the evaluation host owns/permissions those files. (A non-root uid
# can silently fail to read a restrictively-permissioned /data mount and then emit all-fallback scores,
# which is a worse, silent failure than any benefit of dropping privileges here.)
ENTRYPOINT ["python3", "/app/scripts/docker_infer.py"]
