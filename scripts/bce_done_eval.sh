#!/usr/bin/env bash
# Wait for both BCE ViT-L production runs (s42 GPU0, s43 GPU1) to finish, then generate the
# candidate public submission CSVs (s42-best, s43-best, s42+s43 logit-mean ensemble). best.pt is
# val-selected = faithful for the born-digital PUBLIC objective (val is same distribution). CSVs are
# left ready for the user to authorize a Kaggle submit (NO auto-submit — policy: submit only on instruction).
set -u
cd /home/hoppery/ijcai_freuid_chanllenge
S42=233735; S43=233736

for i in $(seq 1 240); do
  a=$(ps -p $S42,$S43 -o pid= 2>/dev/null | wc -l)
  [ "$a" = 0 ] && { echo ">>> both BCE runs exited"; break; }
  sleep 90
done

echo "================ BCE ViT-L PRODUCTION — in-domain val (born-digital proxy for public) ================"
for r in exp_prod896_large_bce exp_prod896_large_bce_s43; do
  echo "--- $r ---"; cat checkpoints/$r/log.csv 2>/dev/null
  echo "  ckpts: $(ls checkpoints/$r/*.pt 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
done

echo "================ generating candidate public submissions (GPU0) ================"
gen() { # $1=out  $2..=ckpt dirs
  local out="$1"; shift
  echo "--- $out  (ckpts: $*) ---"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=_transfer/src OMP_NUM_THREADS=2 \
    python3 _transfer/scripts/ensemble_infer.py --ckpts "$@" --out "$out" 2>&1 | grep -E "wrote|total=|predicted" | tail -3
}
gen submission_bce_s42.csv  checkpoints/exp_prod896_large_bce
gen submission_bce_s43.csv  checkpoints/exp_prod896_large_bce_s43
gen submission_bce_ens.csv  checkpoints/exp_prod896_large_bce checkpoints/exp_prod896_large_bce_s43
echo "================ DONE — CSVs ready, awaiting submit authorization ================"
