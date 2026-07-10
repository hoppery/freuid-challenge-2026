#!/usr/bin/env bash
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
PROXY=manifests/fantasyid_test.parquet; CAP=377916; PROD=375038
rdy(){ [ "$(tail -n +2 checkpoints/$1/log.csv 2>/dev/null | wc -l)" -ge "$2" ]; }
# Stage1: capture-SAM survives to 3ep (bf16 fix works) -> proxy eval
for i in $(seq 1 300); do
  rdy exp_e6sam_capgen 3 && break
  ps -p $CAP >/dev/null 2>&1 || { echo ">>> capture-SAM exited (crash?)"; break; }
  sleep 120
done
echo "===== capture-SAM proxy (fantasyid_test) vs E6 0.423 ====="
cat checkpoints/exp_e6sam_capgen/log.csv 2>/dev/null
for ck in epoch0.pt epoch1.pt epoch2.pt; do
  p="checkpoints/exp_e6sam_capgen/$ck"; [ -f "$p" ] || continue
  o=$(CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=2 python3 scripts/xeval_fantasyid.py --ckpt "$p" --manifest "$PROXY" 2>/dev/null | tail -1)
  echo "  $ck :: $o"
done
# Stage2: launch GUINEA on GPU0 once capture-SAM frees it
until ! ps -p $CAP >/dev/null 2>&1; do sleep 60; done
echo ">>> launching GUINEA SAM (GPU0)"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=_transfer/src OMP_NUM_THREADS=2 nohup python3 -m freuid.train --config _transfer/configs/exp_sam_fold1.yaml > .reports/sam_guinea.out 2>&1 &
GUI=$!
# Stage3: wait GUINEA 3ep + prod done
for i in $(seq 1 300); do
  g=0; rdy exp_sam_fold1 3 && g=1
  { [ "$g" = 1 ] && ! ps -p $PROD >/dev/null 2>&1; } && break
  ps -p $PROD >/dev/null 2>&1 || ps -p $GUI >/dev/null 2>&1 || break
  sleep 150
done
echo "===== SAM 5-fold FINAL + production ====="
echo "DINOv2: EGYPT.0035 MAUR.0052 BENIN.0165 GUINEA.0216 MOZ.0602 (mean.0214) | SAM: EGY.00211 MAUR.0024 MOZ.0217 BEN.0494"
echo "--- GUINEA SAM (vs 0.0216) ---"; cat checkpoints/exp_sam_fold1/log.csv 2>/dev/null
echo "  GUINEA SAM BEST = $(tail -n +2 checkpoints/exp_sam_fold1/log.csv 2>/dev/null | awk -F, 'NR==1||$5<m{m=$5} END{print m}')"
echo "--- production-SAM (in-domain per-epoch) ---"; cat checkpoints/exp_sam_prod/log.csv 2>/dev/null
echo "===== DONE ====="
