"""Rank-mean fusion submission: run N checkpoints on the FREUID test index, rank-normalize each
model's scores, average the ranks, write a submission. The FREUID metric is RANK-based and our LODO
showed rank-mean >> mean-prob for heterogeneous members (e.g. base+CMA BENIN: 0.082 vs 0.127). Use
this for deployment fusion cards instead of freuid.infer (which mean-averages probabilities).

Usage:
  python3 scripts/infer_rankfuse.py --out submission_baseCMA.csv --fill-missing 0.5 \
      --ckpts checkpoints/exp_deploy896_all/best.pt checkpoints/exp_gen3_cma_all/epoch1.pt
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from freuid.config import Config
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.data.adapters.freuid import build_freuid_test_index

_ID, _PRED = "id", "label"


@torch.no_grad()
def _run_one(ckpt_path, test_df):
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    cfg = Config(**{k: v for k, v in ck.get("cfg", {}).items() if hasattr(Config, k)})
    m = build_classifier(cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size,
                         freeze_backbone=cfg.freeze_backbone,
                         use_prototype=getattr(cfg, "use_prototype", False),
                         use_clip=getattr(cfg, "use_clip", False),
                         use_chroma=getattr(cfg, "use_chroma", False),
                         use_spectral=getattr(cfg, "use_spectral", False),
                         use_gsd=getattr(cfg, "use_gsd", False), gsd_r=getattr(cfg, "gsd_r", 3),
                         lora_rank=getattr(cfg, "lora_rank", 0),
                         lora_alpha=getattr(cfg, "lora_alpha", 16),
                         cma_chroma=getattr(cfg, "cma_chroma", False)).cuda()
    m.load_state_dict(ck["model"]); m.eval()
    ds = ManifestDataset(test_df.assign(label=0, attack_type="none", doc_type="u",
                                        source="freuid", split="test"),
                         build_transforms("eval", cfg.img_size), with_label=False)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=12, pin_memory=True)
    s, ids = [], []
    for x, idb in dl:
        with torch.autocast("cuda", enabled=cfg.amp):
            s.append(torch.sigmoid(m(x.cuda())).float().squeeze(1).cpu().numpy())
        ids.extend(list(idb))
    del m; torch.cuda.empty_cache()
    return np.array(ids), np.concatenate(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", default="submission_rankfuse.csv")
    ap.add_argument("--fill-missing", type=float, default=0.5)
    a = ap.parse_args()
    test_df = build_freuid_test_index(existing_only=True)
    print(f"test images on disk: {len(test_df)} | fusing {len(a.ckpts)} ckpts (rank-mean)")
    ids0, ranks = None, []
    for c in a.ckpts:
        ids, sc = _run_one(c, test_df)
        if ids0 is None:
            ids0 = ids
        assert list(ids) == list(ids0), "ckpt id order mismatch"
        ranks.append(rankdata(sc) / len(sc))      # normalized rank in [0,1]
        print(f"  ran {c}")
    fused = np.mean(ranks, axis=0)
    sub = pd.DataFrame({_ID: ids0, _PRED: fused})
    full = build_freuid_test_index(existing_only=False)["image_id"]
    missing = full[~full.isin(sub[_ID])]
    if len(missing):
        sub = pd.concat([sub, pd.DataFrame({_ID: missing, _PRED: a.fill_missing})], ignore_index=True)
        sub = sub.set_index(_ID).loc[full].reset_index()
        print(f"filled {len(missing)} missing rows with {a.fill_missing}")
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
