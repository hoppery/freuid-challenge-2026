#!/usr/bin/env bash
# Download the 32 missing MIDV-500 doc-type zips from smartengines FTP (each ~600MB, ~19G total).
# Resumable (curl -C -). Logs progress; writes DONE sentinel at end.
set -u; cd /home/hoppery/ijcai_freuid_chanllenge
DEST=data/raw/midv500/dataset
mkdir -p "$DEST"
n=0; ok=0
while read -r z; do
  [ -z "$z" ] && continue
  n=$((n+1))
  url="ftp://smartengines.com/midv-500/dataset/$z"
  echo "[$n/32] $z ..."
  for try in 1 2 3; do
    curl -s --connect-timeout 20 --max-time 1800 -C - -o "$DEST/$z" "$url" && { ok=$((ok+1)); echo "  OK $z ($(du -h "$DEST/$z" | cut -f1))"; break; }
    echo "  retry $try for $z"; sleep 10
  done
done < /tmp/midv500_missing.txt
echo "DOWNLOAD DONE: $ok/$n zips. on-disk now: $(ls $DEST/*.zip | wc -l)/50"
