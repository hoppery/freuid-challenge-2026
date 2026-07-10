#!/usr/bin/env bash
# When pubgen v1/v2 finish, generate per-epoch submission CSVs (NO auto-submit).
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
V1=5122; V2=5335
until ! ps -p $V1 >/dev/null 2>&1 && ! ps -p $V2 >/dev/null 2>&1; do sleep 180; done
for V in pubgen_v1 pubgen_v2; do
  echo "=== $V log (in-domain) ==="; cat checkpoints/exp_$V/log.csv 2>/dev/null
  [ -f checkpoints/exp_$V/best.pt ] && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=2 python3 -m freuid.infer --ckpt checkpoints/exp_$V/best.pt --out submission_$V.csv --existing-only --fill-missing 0.5 2>&1 | grep -i wrote | tail -1
done
# ensemble of the two hardened variants + the proven ViT-B FDA (all good archs)
python3 scripts/rank_ensemble.py --out submission_pubgen_ens.csv submission_pubgen_v1.csv submission_pubgen_v2.csv submission_vitb_fda.csv 2>&1 | tail -1
echo "===== pubgen CSVs ready (NO submit — awaiting instruction) ====="
ls submission_pubgen_v1.csv submission_pubgen_v2.csv submission_pubgen_ens.csv 2>/dev/null
echo "===== DONE ====="
