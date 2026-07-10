#!/usr/bin/env bash
# Download MIDV-2020 captured-genuine subsets (photo + scan_upright) for the E9 data-diversity lever.
# ~5GB total (skips clips/video/tif = 26-54GB). Resumable (curl -C -). Writes DONE sentinel.
set -u
cd /home/hoppery/ijcai_freuid_chanllenge
DEST=data/raw/midv2020/dataset
mkdir -p "$DEST"; cd "$DEST"
BASE="ftp://smartengines.com/midv-2020/dataset"
for f in photo.tar scan_upright.tar; do
  echo "[dl] $f ..."
  for try in 1 2 3 4 5 6; do
    curl -s --connect-timeout 30 --max-time 10800 -C - -o "$f" "$BASE/$f" \
      && { echo "  OK $f size=$(du -h "$f" | cut -f1)"; break; }
    echo "  retry $try for $f"; sleep 20
  done
  echo "[extract] $f ..."
  tar xf "$f" && echo "  extracted $f" || echo "  EXTRACT FAILED $f"
done
echo "MIDV2020_DL_DONE files: $(find . -maxdepth 2 -type d | head)"
