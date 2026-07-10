"""Rebuild the IDNet manifest from all extracted locations under data/raw/idnet_ext.

Diversity (many scripts/layouts/categories) is the lever, not volume per country, so
caps are kept low and the location count high. With 10 locations and the default caps
this yields ~50k images (FREUID train is ~69k, so FREUID stays the dominant domain and
IDNet acts as a diversity supplement rather than swamping the target style).

Usage:
  python3 scripts/build_idnet_manifest.py --out manifests/idnet_all.parquet \
      --pos-cap 2500 --fraud-cap 1250
"""
from __future__ import annotations
import argparse
from freuid.data.adapters.idnet import build_idnet_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manifests/idnet_all.parquet")
    ap.add_argument("--pos-cap", type=int, default=2500)
    ap.add_argument("--fraud-cap", type=int, default=1250)
    a = ap.parse_args()

    df = build_idnet_manifest(pos_cap=a.pos_cap, fraud_cap_each=a.fraud_cap)
    df.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} rows")
    print("  label:", df.label.value_counts().to_dict())
    print("  locations:", df.doc_type.nunique())
    for dt, n in df.doc_type.value_counts().sort_index().items():
        print(f"    {dt}: {n}")
    print("  attack_types:", df.attack_type.value_counts().to_dict())


if __name__ == "__main__":
    main()
