#!/bin/bash
# RELEASE-DAY 2-track deployment inference.
# On release: export the private test location, then run this.
#   export FREUID_SAMPLE_SUB=data/raw/freuid/<private_sample_submission.csv>
#   export FREUID_TEST_DIR=data/raw/freuid/<private_images_dir>   # dir holding <id>.jpeg
#   bash scripts/deploy_infer.sh
# Produces:
#   submission_deploy_capture.csv  = CAPTURE card (ViT-B DTC @896 + @1120 + RegNetY@1120, ROOT DTC, prob-avg).
#     PRIMARY for captured/physical private. RegNetY@1120 = arch-diversity breakthrough (2026-07-09): AUC 0.9075,
#     ~doubles capture-proxy vs old 2-ViT card (0.354->~0.27 robust). 2 ViT anchors stabilize RegNet op-point seed-var.
#   submission_deploy_ftens.csv    = field-tamper 12-model ensemble (public rank-1 line). For any born-digital rows.
# Assembly of the FINAL file depends on the organizer's "exact flow" (public vs private id split) — see
# docs/private_release_playbook.md. Default private pick = capture card.
set -u
cd /home/hoppery/ijcai_freuid_chanllenge
export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-}"

echo "[deploy] TEST_DIR=${FREUID_TEST_DIR:-<default public_test>}  SAMPLE_SUB=${FREUID_SAMPLE_SUB:-<default>}"

# --- CAPTURE card (private-primary): ROOT DTC, prob-avg ViT-B@896 + ViT-B@1120 + RegNetY@1120(s42 ep1) ---
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=2 python3 -m freuid.infer \
  --ckpts checkpoints/exp_e6_fda/epoch2.pt checkpoints/exp_e6_fda1120/epoch2.pt \
          checkpoints/exp_e6reg_fda1120/epoch1.pt \
  --out submission_deploy_capture.csv --existing-only --fill-missing 0.5   # --existing-only makes --fill-missing work (skip absent imgs → 0.5); on complete private set nothing is filled
echo "[deploy] capture card -> submission_deploy_capture.csv"

# --- FIELD-TAMPER 12-model ensemble (public line): _transfer, hflip TTA, prob-avg ---
# MINIMAL 3-model public card (0.00041, rank1): ConvNeXt@896 + ViT-Giant + ViT-L@1120 (arch+capacity+res axes)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=_transfer/src OMP_NUM_THREADS=2 python3 scripts/infer_tta_ens_transfer.py \
  --ckpts checkpoints/exp_ftcnx_large/epoch2.pt checkpoints/exp_ftgiant/epoch2.pt \
          checkpoints/exp_fthires1120_s43/epoch2.pt \
  --out submission_deploy_ftens.csv
echo "[deploy] field-tamper ensemble -> submission_deploy_ftens.csv"
echo "[deploy] DONE. Assemble final per docs/private_release_playbook.md; DO NOT submit without user confirm."
