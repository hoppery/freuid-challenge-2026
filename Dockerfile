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
        python3 python3-pip libglib2.0-0 && \
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

# --- source + the 3 CAPTURE-card checkpoints (embedded; ~1 GB total) ---
COPY src/ /app/src/
COPY scripts/docker_infer.py /app/scripts/docker_infer.py
COPY checkpoints/exp_e6_fda/epoch2.pt        /app/checkpoints/exp_e6_fda/epoch2.pt
COPY checkpoints/exp_e6_fda1120/epoch2.pt    /app/checkpoints/exp_e6_fda1120/epoch2.pt
COPY checkpoints/exp_e6reg_fda1120/epoch1.pt /app/checkpoints/exp_e6reg_fda1120/epoch1.pt

ENTRYPOINT ["python3", "/app/scripts/docker_infer.py"]
