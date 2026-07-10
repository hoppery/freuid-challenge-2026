#!/usr/bin/env bash
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
PROXY=manifests/fantasyid_test.parquet; P0=282286; P1=282287
ready(){ [ "$(tail -n +2 checkpoints/$1/log.csv 2>/dev/null | wc -l)" -ge 3 ]; }
for i in $(seq 1 220); do
  r0=0; r1=0; ready exp_e6_capgen && r0=1; ready exp_e6nobid_capgen && r1=1
  { [ "$r0" = 1 ] && [ "$r1" = 1 ]; } && { echo ">>> both E6 reached ep2"; break; }
  [ "$(ps -p $P0,$P1 -o pid= 2>/dev/null | wc -l)" = 0 ] && { echo ">>> both exited"; break; }
  sleep 90
done
echo "===== E6 (MORE captured sources) CAPTURE PROXY vs E5 0.4548 / E2-early 0.5466 ====="
for RUN in exp_e6_capgen exp_e6nobid_capgen; do
  echo "--- $RUN ($([ $RUN = exp_e6_capgen ] && echo '+MIDV2019+BID, 25 types' || echo '+MIDV2019 only, no BID')) ---"
  cat checkpoints/$RUN/log.csv 2>/dev/null
  for ck in epoch0.pt epoch1.pt epoch2.pt best.pt; do
    p="checkpoints/$RUN/$ck"; [ -f "$p" ] || continue
    out=$(CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=2 python3 scripts/xeval_fantasyid.py --ckpt "$p" --manifest "$PROXY" 2>/dev/null | tail -1)
    echo "  $ck  ::  $out"
  done
done
echo "===== DONE ====="
