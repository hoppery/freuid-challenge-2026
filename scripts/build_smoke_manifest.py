"""Build a smoke manifest = the official FREUID manifest filtered to images that
actually exist on disk (the subset fetched by smoke_fetch.py)."""
from pathlib import Path
from freuid.data.adapters.freuid import build_freuid_manifest
from freuid.data.schema import write_manifest

df = build_freuid_manifest()
mask = df["path"].map(lambda p: Path(p).exists())
sub = df[mask].reset_index(drop=True)
print(f"full manifest rows={len(df)}  existing-on-disk={len(sub)}")
print("label counts:", dict(sub["label"].value_counts()))
print("doc_type counts:", dict(sub["doc_type"].value_counts()))
write_manifest(sub, "manifests/_smoke.parquet")
print("wrote manifests/_smoke.parquet")
