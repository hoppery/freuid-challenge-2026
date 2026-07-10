#!/usr/bin/env bash
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
L16=42522; B8=42523
until ! ps -p $L16 >/dev/null 2>&1 && ! ps -p $B8 >/dev/null 2>&1; do sleep 180; done
echo "=== RINE variants val FREUID ==="
echo "--- L16 ---"; grep -E "^ep|saved" .reports/rine_L16.out 2>/dev/null | tail -4
echo "--- B8 ---"; grep -E "^ep|saved" .reports/rine_B8.out 2>/dev/null | tail -4
[ -f checkpoints/rine_L16/best.pt ] && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=2 python3 scripts/rine.py --mode infer --ckpt checkpoints/rine_L16/best.pt --out submission_rine_L16.csv 2>&1 | grep -i wrote | tail -1
[ -f checkpoints/rine_B8/best.pt ] && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=2 python3 scripts/rine.py --mode infer --ckpt checkpoints/rine_B8/best.pt --out submission_rine_B8.csv 2>&1 | grep -i wrote | tail -1
# RINE-only ensemble (3 RINE variants) + RINE-ensemble fused with best diverse archs
python3 scripts/rank_ensemble.py --out submission_rine_ens3.csv submission_rine.csv submission_rine_L16.csv submission_rine_B8.csv 2>&1 | tail -1
python3 scripts/rank_ensemble.py --out submission_ens_max.csv submission_rine.csv submission_rine_L16.csv submission_rine_B8.csv submission_vitb_fda.csv submission_vitb_rgb_fda.csv submission_vitL_ensemble.csv submission_convnext_hpf.csv 2>&1 | tail -1
echo "===== RINE-ensemble CSVs ready (next-slate candidates) ====="
ls submission_rine_ens3.csv submission_ens_max.csv 2>/dev/null
echo "===== DONE ====="
