"""Update manifests/freuid_partial.parquet to include all files now present on disk.

Run this periodically while data is being copied:
  python3 scripts/refresh_partial_manifest.py

Prints a diff so you can see how many new files arrived since the last refresh.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
FULL = ROOT / "manifests" / "freuid.parquet"
PARTIAL = ROOT / "manifests" / "freuid_partial.parquet"


def main():
    df = pd.read_parquet(FULL)
    exists = df["path"].apply(lambda p: (ROOT / p).exists())
    partial = df[exists].reset_index(drop=True)

    prev_count = len(pd.read_parquet(PARTIAL)) if PARTIAL.exists() else 0
    partial.to_parquet(PARTIAL, index=False)

    pct = len(partial) / len(df) * 100
    delta = len(partial) - prev_count
    print(f"freuid_partial: {len(partial)}/{len(df)} ({pct:.1f}%)  [+{delta} since last refresh]")
    print("by doc_type:")
    print(partial.groupby("doc_type").size().to_string())
    print("by label:", partial["label"].value_counts().to_dict())

    if len(partial) == len(df):
        print("\nAll files present — switch configs to manifests/freuid.parquet")
        sys.exit(0)


if __name__ == "__main__":
    main()
