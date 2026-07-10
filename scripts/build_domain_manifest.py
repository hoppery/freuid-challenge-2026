"""Domain-router manifest: born-digital (label 0) vs physically-CAPTURED (label 1).
Repurposes the fraud 'label' column to mean DOMAIN (captured=1) so freuid.train's BCE learns a
born-digital-vs-captured classifier. Used to ROUTE private rows: captured -> capture card,
born-digital -> field-tamper card.

- captured (1): dedup is_digital=False from bid + midv500/2020/holo/2019 (diverse real photos/scans).
  fantasyid EXCLUDED so fantasyid_test stays a clean eval for the routed pipeline.
- born-digital (0): freuid (clean digital renders), sampled to balance.
"""
import pandas as pd

CAP_SRC = {"bid", "midv500", "midv2020", "midvholo", "midv2019"}
COLS = ["path", "label", "attack_type", "doc_type", "source", "split", "is_digital"]

def main():
    d = pd.read_parquet("manifests/freuid_fid_capgen_e9.parquet")[COLS].drop_duplicates("path")
    cap = d[(d["source"].isin(CAP_SRC)) & (~d["is_digital"])].copy()
    bd = d[(d["source"] == "freuid") & (d["is_digital"])].copy()
    # balance born-digital to captured count (deterministic sample; index order stable)
    n = len(cap)
    bd = bd.iloc[:: max(1, len(bd) // n)].head(n).copy()
    cap["label"] = 1
    bd["label"] = 0
    out = pd.concat([cap, bd], ignore_index=True)[COLS]
    out.to_parquet("manifests/domain_router.parquet")
    print(f"domain manifest: {len(out)} | captured {len(cap)} (src {sorted(cap['source'].unique())}) "
          f"| born-digital {len(bd)}")
    print("captured by source:", cap["source"].value_counts().to_dict())

if __name__ == "__main__":
    main()
