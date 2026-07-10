#!/usr/bin/env bash
# Auto-submit union models (user authorized). Submit each model's best.pt when training finishes.
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
TOK="${KAGGLE_API_TOKEN:?export your Kaggle API token before running}"
VL=3567; DTC=3568

submit(){ KAGGLE_API_TOKEN="$TOK" kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai -f "$1" -m "$2" 2>&1 | grep -iE "success|error" | tail -1; }

until ! ps -p $VL >/dev/null 2>&1; do sleep 180; done
echo ">>> union ViT-L done; log:"; cat checkpoints/exp_union_vitl/log.csv 2>/dev/null
if [ -f checkpoints/exp_union_vitl/best.pt ]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=_transfer/src OMP_NUM_THREADS=2 python3 _transfer/scripts/ensemble_infer.py --ckpts checkpoints/exp_union_vitl --out submission_union_vitl.csv 2>&1 | grep -E "wrote|predicted" | tail -1
  submit submission_union_vitl.csv "union ViT-L BCE+FDA + idnet/docxpand diverse synthetic (186k) — public-gap diverse-data bet"
else echo "  NO best.pt (run died?)"; fi

until ! ps -p $DTC >/dev/null 2>&1; do sleep 180; done
echo ">>> union DTC done; log:"; cat checkpoints/exp_union_dtc/log.csv 2>/dev/null
if [ -f checkpoints/exp_union_dtc/best.pt ]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=2 python3 -m freuid.infer --ckpt checkpoints/exp_union_dtc/best.pt --config checkpoints/exp_union_dtc/config.yaml --out submission_union_dtc.csv 2>&1 | grep -iE "wrote|rows" | tail -1
  submit submission_union_dtc.csv "union ViT-B DTC two-stream + idnet diverse synthetic — research ID-PAD pattern for public-gap"
else echo "  NO best.pt"; fi

sleep 90
echo "===== public scores ====="
KAGGLE_API_TOKEN="$TOK" kaggle competitions submissions the-freuid-challenge-2026-ijcai-ecai 2>/dev/null | grep -iE "union|fileName" | head -4
echo "===== DONE ====="
