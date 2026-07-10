"""Adapter: IDNet-2025 (extracted scanned locations) -> unified manifest.

IDNet adds REAL-layout-diverse synthetic IDs & passports (European locations), which
FREUID lacks (FREUID is 4 African DLs + 1 ID). This layout diversity is the principled
fix for the unseen-doc-type / unseen-layout gap (the MAURITIUS-ID killer fold and the
private test's 2 unseen types). Public, CC-BY-4.0, cited in the report.

Extracted layout: data/raw/idnet_ext/<LOC>/scanned/{positive, fraud5_inpaint_and_rewrite,
fraud6_crop_and_replace}/*.jpg. positive=bona-fide(0); fraud5/fraud6=fraud(1).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from freuid.data.schema import COLUMNS, AttackType

ROOT = Path("data/raw/idnet_ext")
_FOLDERS = [
    ("positive", 0, AttackType.NONE),
    ("fraud5_inpaint_and_rewrite", 1, AttackType.GENAI_DIGITAL),
    ("fraud6_crop_and_replace", 1, AttackType.OTHER_DIGITAL),
]


def build_idnet_manifest(root=ROOT, pos_cap: int = 4000, fraud_cap_each: int = 2000,
                         seed: int = 42) -> pd.DataFrame:
    root = Path(root)
    rng = np.random.default_rng(seed)
    rows = []
    for loc_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        loc = loc_dir.name
        base = loc_dir / "scanned" if (loc_dir / "scanned").exists() else loc_dir
        for folder, label, at in _FOLDERS:
            d = base / folder
            if not d.exists():
                continue
            imgs = sorted(d.glob("*.jpg"))
            cap = pos_cap if label == 0 else fraud_cap_each
            if cap and len(imgs) > cap:
                sel = rng.choice(len(imgs), cap, replace=False)
                imgs = [imgs[i] for i in sel]
            for p in imgs:
                rows.append(dict(path=str(p), label=label, attack_type=at,
                                 doc_type=f"idnet_{loc}", source="idnet", split="train"))
    if not rows:
        raise FileNotFoundError(f"no IDNet images under {root} — extract tars first")
    return pd.DataFrame(rows, columns=COLUMNS)
