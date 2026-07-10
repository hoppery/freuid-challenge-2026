#!/usr/bin/env bash
# E3 capture-proxy completion monitor: wait for both runs to finish 3 epochs, then eval every
# saved epoch on fantasyid_test (leakage-free capture proxy, 1295 imgs). Baseline E2-early=0.5466.
set -u
cd /home/hoppery/ijcai_freuid_chanllenge
PROXY=manifests/fantasyid_test.parquet
RUNS="exp_e3_capture exp_e3_cdc_only"

ready() { [ "$(tail -n +2 checkpoints/$1/log.csv 2>/dev/null | wc -l)" -ge 3 ]; }

# wait up to ~5h for both runs to reach ep2 (or their procs to vanish)
for i in $(seq 1 200); do
  done_full=0; done_cdc=0
  ready exp_e3_capture && done_full=1
  ready exp_e3_cdc_only && done_cdc=1
  alive=$(ps -p 211743,211744 -o pid= 2>/dev/null | wc -l)
  if [ "$done_full" = 1 ] && [ "$done_cdc" = 1 ]; then echo ">>> both reached ep2"; break; fi
  if [ "$alive" = 0 ]; then echo ">>> both procs exited (full=$done_full cdc=$done_cdc)"; break; fi
  sleep 90
done

echo "================ E3 CAPTURE PROXY (fantasyid_test) vs E2-early 0.5466 ================"
for run in $RUNS; do
  echo "--- $run ---"
  cat checkpoints/$run/log.csv 2>/dev/null
  for ck in epoch0.pt epoch1.pt epoch2.pt best.pt; do
    p="checkpoints/$run/$ck"
    [ -f "$p" ] || continue
    out=$(CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=2 \
          python3 scripts/xeval_fantasyid.py --ckpt "$p" --manifest "$PROXY" 2>/dev/null | tail -1)
    echo "  $ck  ::  $out"
  done
done
echo "================ DONE ================"
