#!/usr/bin/env bash
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
RUN=exp_e5_capgen; PID=255211; PROXY=manifests/fantasyid_test.parquet
for i in $(seq 1 200); do
  m=$(tail -n +2 checkpoints/$RUN/log.csv 2>/dev/null | wc -l)
  [ "$m" -ge 3 ] && { echo ">>> $RUN reached ep2"; break; }
  ps -p $PID >/dev/null 2>&1 || { echo ">>> proc exited (m=$m)"; break; }
  sleep 90
done
echo "===== E5 (MAX-diverse captured-genuine: MIDV-500+Holo) CAPTURE PROXY vs E2-early 0.5466 ====="
echo "(E4-gf 0.6129 / E4g 0.5783 — both lost; E5 = diverse genuine, the E4g fix)"
cat checkpoints/$RUN/log.csv 2>/dev/null
for ck in epoch0.pt epoch1.pt epoch2.pt best.pt; do
  p="checkpoints/$RUN/$ck"; [ -f "$p" ] || continue
  out=$(CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src OMP_NUM_THREADS=2 python3 scripts/xeval_fantasyid.py --ckpt "$p" --manifest "$PROXY" 2>/dev/null | tail -1)
  echo "  $ck  ::  $out"
done
echo "===== DONE ====="
