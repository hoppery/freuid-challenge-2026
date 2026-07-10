#!/usr/bin/env bash
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
PROXY=manifests/fantasyid_test.parquet
P0=262126; P1=262415   # ep6 (x2 GPU0), capgen4 ep6 (x4 GPU1)
ready(){ [ "$(tail -n +2 checkpoints/$1/log.csv 2>/dev/null | wc -l)" -ge 6 ]; }
for i in $(seq 1 260); do
  r0=0; r1=0; ready exp_e5_capgen_ep6 && r0=1; ready exp_e5_capgen4_ep6 && r1=1
  a=$(ps -p $P0,$P1 -o pid= 2>/dev/null | wc -l)
  { [ "$r0" = 1 ] && [ "$r1" = 1 ]; } && { echo ">>> both ep6 reached ep5"; break; }
  [ "$a" = 0 ] && { echo ">>> both procs exited (r0=$r0 r1=$r1)"; break; }
  sleep 90
done
echo "===== E5 ep6 CAPTURE-OPTIMAL SWEEP vs E2-early 0.5466 / E5(3ep) ep2 0.4548 ====="
for RUN in exp_e5_capgen_ep6 exp_e5_capgen4_ep6; do
  echo "--- $RUN ---"; cat checkpoints/$RUN/log.csv 2>/dev/null
  for ck in epoch0.pt epoch1.pt epoch2.pt epoch3.pt epoch4.pt epoch5.pt; do
    p="checkpoints/$RUN/$ck"; [ -f "$p" ] || continue
    out=$(CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=2 python3 scripts/xeval_fantasyid.py --ckpt "$p" --manifest "$PROXY" 2>/dev/null | tail -1)
    echo "  $ck  ::  $out"
  done
done
echo "===== DONE ====="
