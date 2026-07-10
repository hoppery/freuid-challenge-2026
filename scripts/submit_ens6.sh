#!/usr/bin/env bash
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
TOK="${KAGGLE_API_TOKEN:?export your Kaggle API token before running}"
for i in $(seq 1 30); do
  out=$(KAGGLE_API_TOKEN="$TOK" kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai -f submission_ens6.csv -m "6-arch rank-ensemble incl RINE + ViT-B FDA + ViT-L tail/bce + ViT-B rgb + ConvNeXt hpf — cross-arch diversity for public" 2>&1)
  echo "$out" | grep -qi "success" && { echo ">>> ens6 SUBMITTED (attempt $i)"; break; }
  echo "attempt $i: limit not reset"; sleep 300
done
sleep 60
echo "===== ens6 public score ====="
KAGGLE_API_TOKEN="$TOK" kaggle competitions submissions the-freuid-challenge-2026-ijcai-ecai 2>/dev/null | grep -E "ens6|publicScore" | head -2
echo "===== DONE ====="
