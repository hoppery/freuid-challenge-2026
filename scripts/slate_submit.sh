#!/usr/bin/env bash
# At daily reset, submit the prioritized slate of public candidates (best ensemble first), report scores.
# User authorized public submission. Up to 5/day.
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
TOK="${KAGGLE_API_TOKEN:?export your Kaggle API token before running}"

# (csv, message) priority order
declare -a CSVS=(
  "submission_ens6.csv|6-arch rank-ensemble incl RINE (DINOv2 intermediate-layer) + ViT-B/L + ConvNeXt — max cross-arch diversity"
  "submission_rine.csv|RINE: frozen DINOv2 ViT-L intermediate-layer aggregation (val 0.0467, anti-overfit) — diverse standalone"
  "submission_ens_rine_vitb.csv|rank-ens RINE + ViT-B DTC FDA (best single 0.0798) — diversity + strength"
  "submission_ens_rine3.csv|rank-ens RINE + ViT-B DTC FDA + ViT-B rgb FDA"
  "submission_rankens.csv|4-arch rank-ensemble (ViT-B FDA + ViT-L tail-ens + ViT-L BCE + ConvNeXt hpf)"
)

submit_one(){ # $1 csv $2 msg ; returns 0 on success
  KAGGLE_API_TOKEN="$TOK" kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai -f "$1" -m "$2" 2>&1 | grep -qi "success"
}

n=0
for entry in "${CSVS[@]}"; do
  csv="${entry%%|*}"; msg="${entry#*|}"
  [ -f "$csv" ] || { echo "skip missing $csv"; continue; }
  # retry until reset (limit clears) for this csv
  for try in $(seq 1 60); do
    if submit_one "$csv" "$msg"; then echo ">>> submitted $csv"; n=$((n+1)); break; fi
    echo "  $csv: limit/err (try $try)"; sleep 1200
  done
  sleep 30
done
echo "submitted $n candidates"
sleep 90
echo "===== public scores ====="
KAGGLE_API_TOKEN="$TOK" kaggle competitions submissions the-freuid-challenge-2026-ijcai-ecai 2>/dev/null | grep -E "ens6|rine|rankens|fileName" | head -8
echo "===== DONE ====="
