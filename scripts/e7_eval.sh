#!/usr/bin/env bash
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
PROXY=manifests/fantasyid_test.parquet; P0=292004; P1=292005
ready(){ [ "$(tail -n +2 checkpoints/$1/log.csv 2>/dev/null | wc -l)" -ge 3 ]; }
for i in $(seq 1 220); do
  r0=0; r1=0; ready exp_e7_bid8k && r0=1; ready exp_e7_bid16k && r1=1
  { [ "$r0" = 1 ] && [ "$r1" = 1 ]; } && { echo ">>> both E7 reached ep2"; break; }
  [ "$(ps -p $P0,$P1 -o pid= 2>/dev/null | wc -l)" = 0 ] && { echo ">>> both exited"; break; }
  sleep 90
done
echo "===== E7 BID-SCALING CAPTURE PROXY vs E6(BID3.2k) 0.423 / E5 0.4548 ====="
for RUN in exp_e7_bid8k exp_e7_bid16k; do
  echo "--- $RUN ---"; cat checkpoints/$RUN/log.csv 2>/dev/null
  for ck in epoch0.pt epoch1.pt epoch2.pt; do
    p="checkpoints/$RUN/$ck"; [ -f "$p" ] || continue
    out=$(CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=2 python3 scripts/xeval_fantasyid.py --ckpt "$p" --manifest "$PROXY" 2>/dev/null | tail -1)
    echo "  $ck  ::  $out"
  done
done
echo "===== DONE ====="
