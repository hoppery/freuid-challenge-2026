"""Plumbing smoke: build a tiny manifest from the real public_test JPEGs already on
disk, with SYNTHETIC balanced labels + 2 synthetic doc_types. This verifies the full
pipeline (load->augment->train->metrics->checkpoint->infer->submission) runs without
errors before the 16.3GB archive (real train labels) finishes downloading."""
from pathlib import Path
import pandas as pd
from freuid.data.schema import write_manifest, Label, AttackType

test_dir = Path("data/raw/freuid/public_test/public_test")
imgs = sorted(test_dir.glob("*.jpeg"))
assert len(imgs) >= 8, f"need >=8 images, found {len(imgs)}"

rows = []
for i, p in enumerate(imgs):
    label = i % 2  # balanced
    rows.append({
        "path": str(p),
        "label": label,
        "attack_type": AttackType.NONE if label == Label.BONA_FIDE else AttackType.UNKNOWN,
        "doc_type": "smoke_A" if i < len(imgs) // 2 else "smoke_B",
        "source": "freuid",
        "split": "train",
    })
df = pd.DataFrame(rows)
write_manifest(df, "manifests/_smoke.parquet")
print(f"wrote manifests/_smoke.parquet rows={len(df)} "
      f"labels={dict(df['label'].value_counts())} doc_types={dict(df['doc_type'].value_counts())}")
