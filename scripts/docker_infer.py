"""FREUID reproducibility Docker entrypoint.

Reads a FLAT directory of images (default /data), scores each with the CAPTURE card
(3-model prob-avg: ViT-B DTC @896 + ViT-B DTC @1120 + RegNetY-160 DTC @1120), and writes
/submissions/submission.csv with columns id,label where:
  id    = input filename without extension
  label = P(document is fraudulent), higher = more confident fraud, finite float in [0,1]

Contract (organizer): docker run --network none -v <imgs>:/data:ro -v <out>:/submissions <image>
No network / no downloads at runtime (weights embedded, pretrained=False). Writes only /submissions.
Device: CUDA if available, else CPU. Robust to unreadable images (assigned 0.5, never crashes).
"""
from __future__ import annotations
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from freuid.config import Config
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier

DATA_DIR = os.environ.get("FREUID_DATA_DIR", "/data")
# Accept the organizer's env var names (FREUID_OUTPUT_DIR / FREUID_SUBMISSION_PATH) as well as ours,
# so the entrypoint honors the contract regardless of which the verification harness sets.
OUT_DIR = os.environ.get("FREUID_OUTPUT_DIR", os.environ.get("FREUID_OUT_DIR", "/submissions"))
SUB_PATH = os.environ.get("FREUID_SUBMISSION_PATH", os.path.join(OUT_DIR, "submission.csv"))
# The CAPTURE card: 3-model DTC prob-average (the single submitted model).
CKPTS = [
    "checkpoints/exp_e6_fda/epoch2.pt",        # ViT-B DINOv2 DTC @896
    "checkpoints/exp_e6_fda1120/epoch2.pt",    # ViT-B DINOv2 DTC @1120
    "checkpoints/exp_e6reg_fda1120/epoch1.pt", # RegNetY-160 DTC @1120 (arch-diversity)
]
EXTS = ("jpeg", "jpg", "png", "webp", "bmp", "tif", "tiff")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FILL = 0.5  # score for images that fail to decode


def index_data(d: str) -> pd.DataFrame:
    """Flat, non-recursive scan of /data. Case-insensitive on the extension (matches the organizer's
    reference discover_images: every file directly under /data whose suffix.lower() is a supported
    image extension; id = filename stem)."""
    exts = {"." + e for e in EXTS}
    paths = []
    if os.path.isdir(d):
        for f in os.listdir(d):
            p = os.path.join(d, f)
            if os.path.isfile(p) and os.path.splitext(f)[1].lower() in exts:
                paths.append(p)
    paths = sorted(set(paths))
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return pd.DataFrame({"path": paths, "image_id": ids})


class _FlatDataset(Dataset):
    """Reads images from a flat dir; returns (tensor, id, ok_flag). Never raises on a bad image."""
    def __init__(self, df, transform, size):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            return torch.zeros(3, self.size, self.size), str(r["image_id"]), False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.transform(image=img)["image"]
        return x, str(r["image_id"]), True


@torch.no_grad()
def run_one(ckpt: str, df: pd.DataFrame) -> dict:
    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    cfg = Config(**{k: v for k, v in ck.get("cfg", {}).items() if hasattr(Config, k)})
    model = build_classifier(
        cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size,
        freeze_backbone=cfg.freeze_backbone,
        use_prototype=getattr(cfg, "use_prototype", False),
        use_clip=getattr(cfg, "use_clip", False),
        use_chroma=getattr(cfg, "use_chroma", False),
        use_spectral=getattr(cfg, "use_spectral", False),
        use_gsd=getattr(cfg, "use_gsd", False), gsd_r=getattr(cfg, "gsd_r", 3),
        lora_rank=getattr(cfg, "lora_rank", 0), lora_alpha=getattr(cfg, "lora_alpha", 16),
        cma_chroma=getattr(cfg, "cma_chroma", False),
        cdc_theta=getattr(cfg, "cdc_theta", 0.0)).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()

    nw = min(8, (os.cpu_count() or 2))
    ds = _FlatDataset(df, build_transforms("eval", cfg.img_size), cfg.img_size)
    dl = DataLoader(ds, batch_size=max(1, cfg.batch_size), shuffle=False,
                    num_workers=nw, pin_memory=(DEVICE == "cuda"))
    out = {}
    for x, ids, ok in dl:
        with torch.autocast(DEVICE, enabled=(cfg.amp and DEVICE == "cuda")):
            p = torch.sigmoid(model(x.to(DEVICE))).float().squeeze(1).cpu().numpy()
        ok = ok.numpy() if hasattr(ok, "numpy") else np.asarray(ok)
        for j, i in enumerate(ids):
            out[i] = float(p[j]) if bool(ok[j]) else np.nan
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def _validate(sub: pd.DataFrame, expected_ids) -> None:
    """Mirror the organizer's contract check: columns [id,label], one finite row per image, no
    missing/extra ids. Raise (-> non-zero exit) on any violation."""
    if list(sub.columns) != ["id", "label"]:
        raise ValueError(f"submission.csv must have columns ['id','label']; got {list(sub.columns)}")
    got = set(sub["id"].astype(str))
    missing, extra = expected_ids - got, got - expected_ids
    if missing:
        raise ValueError(f"submission.csv missing {len(missing)} id(s), e.g. {sorted(missing)[:3]}")
    if extra:
        raise ValueError(f"submission.csv has {len(extra)} unexpected id(s), e.g. {sorted(extra)[:3]}")
    if int(sub["id"].duplicated().sum()) != 0:
        raise ValueError("submission.csv has duplicate ids")
    if not np.isfinite(sub["label"].to_numpy(dtype=float)).all():
        raise ValueError("submission.csv has non-finite labels")


def main() -> int:
    os.makedirs(os.path.dirname(SUB_PATH) or ".", exist_ok=True)
    df = index_data(DATA_DIR)
    print(f"[docker_infer] {len(df)} images in {DATA_DIR} | device={DEVICE} | {len(CKPTS)} models")
    if len(df) == 0:
        pd.DataFrame(columns=["id", "label"]).to_csv(SUB_PATH, index=False)
        print(f"[docker_infer] no images; wrote empty {SUB_PATH}")
        return 0

    per_model = [run_one(ck, df) for ck in CKPTS]
    rows = []
    for _id in df["image_id"].astype(str):
        vals = [m.get(_id, np.nan) for m in per_model]
        vals = [v for v in vals if v == v]  # drop NaN (unreadable in that model)
        rows.append(float(np.mean(vals)) if vals else FILL)
    sub = pd.DataFrame({"id": df["image_id"].astype(str), "label": rows})
    sub["label"] = sub["label"].fillna(FILL).clip(0.0, 1.0)
    _validate(sub, set(df["image_id"].astype(str)))
    sub.to_csv(SUB_PATH, index=False)
    print(f"[docker_infer] wrote {SUB_PATH} ({len(sub)} rows) "
          f"| label range {sub['label'].min():.4f}-{sub['label'].max():.4f} | dup {int(sub['id'].duplicated().sum())}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # exit non-zero on any failure, per the reproducibility contract
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
