#!/usr/bin/env bash
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
PROXY=manifests/fantasyid_test.parquet
CAP=375037; PROD=375038
rdy(){ [ "$(tail -n +2 checkpoints/$1/log.csv 2>/dev/null | wc -l)" -ge "$2" ]; }

# --- Stage 1: wait for capture-SAM (3ep), eval proxy vs E6 0.423 ---
for i in $(seq 1 300); do
  rdy exp_e6sam_capgen 3 && break
  ps -p $CAP >/dev/null 2>&1 || break
  sleep 120
done
echo "===== capture-SAM proxy (fantasyid_test) vs E6 0.423 ====="
cat checkpoints/exp_e6sam_capgen/log.csv 2>/dev/null
for ck in epoch0.pt epoch1.pt epoch2.pt; do
  p="checkpoints/exp_e6sam_capgen/$ck"; [ -f "$p" ] || continue
  out=$(CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=2 python3 scripts/xeval_fantasyid.py --ckpt "$p" --manifest "$PROXY" 2>/dev/null | tail -1)
  echo "  $ck :: $out"
done

# --- Stage 2: capture-SAM done -> launch GUINEA SAM on GPU0 (free now) ---
until ! ps -p $CAP >/dev/null 2>&1; do sleep 60; done
echo ">>> launching GUINEA SAM on GPU0"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=_transfer/src OMP_NUM_THREADS=2 nohup python3 -m freuid.train --config _transfer/configs/exp_sam_fold1.yaml > .reports/sam_guinea.out 2>&1 &
GUI=$!
echo "GUINEA SAM PID $GUI"

# --- Stage 3: wait for GUINEA (3ep) + production-SAM (done), report ---
for i in $(seq 1 300); do
  g=0; rdy exp_sam_fold1 3 && g=1
  ps -p $PROD >/dev/null 2>&1 || ps -p $GUI >/dev/null 2>&1 || { [ "$g" = 1 ] && break; }
  { [ "$g" = 1 ] && ! ps -p $PROD >/dev/null 2>&1; } && break
  sleep 150
done
echo "===== SAM 5-fold FINAL + production ====="
echo "DINOv2 5-fold mean 0.0214 | SAM so far: EGYPT 0.00211 MAURITIUS 0.0024 MOZ 0.0217 BENIN 0.0494"
echo "--- GUINEA SAM (vs DINOv2 0.0216) ---"; cat checkpoints/exp_sam_fold1/log.csv 2>/dev/null
gbest=$(tail -n +2 checkpoints/exp_sam_fold1/log.csv 2>/dev/null | awk -F, 'NR==1||$5<m{m=$5} END{print m}')
echo "  GUINEA SAM BEST = $gbest"
echo "--- production-SAM (in-domain, deployment card; per-epoch for early-stop) ---"; cat checkpoints/exp_sam_prod/log.csv 2>/dev/null
echo "===== DONE ====="
