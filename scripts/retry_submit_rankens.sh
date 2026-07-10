#!/usr/bin/env bash
# Retry submitting the rank-ensemble until the daily limit resets (every 30min), then report public score.
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
TOK="${KAGGLE_API_TOKEN:?export your Kaggle API token before running}"
for i in $(seq 1 60); do
  out=$(KAGGLE_API_TOKEN="$TOK" kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai -f submission_rankens.csv -m "rank-ensemble 4 diverse archs (ViT-B DTC+FDA + ViT-L tail-ens + ViT-L BCE + ConvNeXt hpf) — cross-arch diversity for public generalization" 2>&1)
  if echo "$out" | grep -qi "success"; then
    echo ">>> SUBMITTED (attempt $i)"; break
  fi
  echo "attempt $i: limit not reset yet"; sleep 1800
done
sleep 60
echo "===== rank-ensemble public score ====="
KAGGLE_API_TOKEN="$TOK" kaggle competitions submissions the-freuid-challenge-2026-ijcai-ecai 2>/dev/null | grep -iE "rankens|fileName" | head -2
echo "===== DONE ====="
