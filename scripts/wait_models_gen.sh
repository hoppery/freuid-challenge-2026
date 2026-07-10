#!/usr/bin/env bash
# When during-wait models finish, generate their submission CSVs for tomorrow's slate.
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
VITB=27078; RINE=28095
until ! ps -p $VITB >/dev/null 2>&1; do sleep 180; done
echo ">>> ViT-B rgb FDA done; log:"; cat checkpoints/exp_vitb_rgb_fda/log.csv 2>/dev/null
if [ -f checkpoints/exp_vitb_rgb_fda/best.pt ]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=_transfer/src OMP_NUM_THREADS=2 python3 _transfer/scripts/ensemble_infer.py --ckpts checkpoints/exp_vitb_rgb_fda --out submission_vitb_rgb_fda.csv 2>&1 | grep -E "wrote|predicted" | tail -1
fi
until ! ps -p $RINE >/dev/null 2>&1; do sleep 180; done
echo ">>> RINE done; log:"; cat .reports/rine_train.out 2>/dev/null | grep -E "^ep|saved" | tail -6
if [ -f checkpoints/rine/best.pt ]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=2 python3 scripts/rine.py --mode infer --ckpt checkpoints/rine/best.pt --out submission_rine.csv 2>&1 | grep -iE "wrote|predicted" | tail -1
fi
echo "===== during-wait models CSVs ready ====="
ls -la submission_vitb_rgb_fda.csv submission_rine.csv 2>/dev/null
echo "===== DONE ====="
