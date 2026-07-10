#!/usr/bin/env bash
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
VL=423278; DTC=423939
until ! ps -p $VL >/dev/null 2>&1 && ! ps -p $DTC >/dev/null 2>&1; do sleep 180; done
echo "===== union runs done — in-domain logs ====="
echo "--- union ViT-L (BCE+FDA+union) ---"; cat checkpoints/exp_union_vitl/log.csv 2>/dev/null
echo "--- union DTC (two-stream+union) ---"; cat checkpoints/exp_union_dtc/log.csv 2>/dev/null
echo ">>> generating public submission CSVs (await user auth to submit)"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=_transfer/src OMP_NUM_THREADS=2 python3 _transfer/scripts/ensemble_infer.py --ckpts checkpoints/exp_union_vitl --out submission_union_vitl.csv 2>&1 | grep -E "wrote|predicted" | tail -1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=2 python3 -m freuid.infer --ckpt checkpoints/exp_union_dtc/best.pt --config checkpoints/exp_union_dtc/config.yaml --out submission_union_dtc.csv 2>&1 | grep -iE "wrote|rows" | tail -1
echo "===== DONE — CSVs ready ====="
