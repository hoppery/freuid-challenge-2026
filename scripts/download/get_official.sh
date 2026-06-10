#!/bin/bash
# Resilient downloader for the official FREUID archive. Kaggle's GCS throttles the
# bundle (~0.6-1.5 MiB/s) and the signed URL expires after a few hours, so we loop:
# fetch a FRESH signed URL, resume with aria2 (-c), repeat until the .aria2 control
# file is gone (download complete). Robust to URL expiry over a multi-hour pull.
cd /home/hoppery/ijcai_freuid_chanllenge || exit 1
unset PYTHONPATH
DEST=data/raw/freuid
OUT=the-freuid-challenge-2026-ijcai-ecai.zip
EXPECTED=17508390644

fetch_url() {
  python3 - <<'PY'
import os
os.environ.setdefault("KAGGLE_API_TOKEN", open(os.path.expanduser("~/.kaggle/access_token")).read().strip())
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest
api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as k:
    req = ApiDownloadDataFilesRequest(); req.competition_name = "the-freuid-challenge-2026-ijcai-ecai"
    open("/tmp/freuid_signed_url.txt", "w").write(k.competitions.competition_api_client.download_data_files(req).url)
PY
}

for attempt in $(seq 1 200); do
  cur=$(stat -c%s "$DEST/$OUT" 2>/dev/null || echo 0)
  if [ ! -f "$DEST/$OUT.aria2" ] && [ "$cur" -ge "$EXPECTED" ]; then
    echo "[get_official] COMPLETE ($cur bytes)"; break
  fi
  echo "[get_official] attempt $attempt: have $cur / $EXPECTED bytes; refreshing URL"
  fetch_url
  URL=$(cat /tmp/freuid_signed_url.txt)
  aria2c -c -x16 -s16 -k1M --file-allocation=none --max-tries=5 --retry-wait=15 \
    --summary-interval=60 --console-log-level=warn -d "$DEST" -o "$OUT" "$URL"
  rc=$?
  echo "[get_official] aria2 rc=$rc"
  if [ "$rc" -eq 0 ] && [ ! -f "$DEST/$OUT.aria2" ]; then
    echo "[get_official] COMPLETE via aria2"; break
  fi
  sleep 5
done
echo "[get_official] final size: $(stat -c%s "$DEST/$OUT" 2>/dev/null) bytes"
