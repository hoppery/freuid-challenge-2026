from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from freuid.config import Config
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.data.adapters.freuid import build_freuid_test_index
from freuid.metrics import apcer_at_bpcer

_ID_HEADER = "id"
_PRED_HEADER = "label"


@torch.no_grad()
def _run_one(ckpt_path: str, test_df: pd.DataFrame, existing_only: bool) -> np.ndarray:
    """Run inference for a single checkpoint; returns raw sigmoid scores."""
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    cfg_dict = ck.get("cfg", {})
    cfg = Config(**{k: v for k, v in cfg_dict.items() if hasattr(Config, k)})

    model = build_classifier(cfg.model_type, cfg.backbone, pretrained=False,
                             img_size=cfg.img_size, freeze_backbone=cfg.freeze_backbone,
                             use_prototype=getattr(cfg, "use_prototype", False),
                             use_clip=getattr(cfg, "use_clip", False),
                             use_chroma=getattr(cfg, "use_chroma", False),
                             use_spectral=getattr(cfg, "use_spectral", False),
                             use_gsd=getattr(cfg, "use_gsd", False),
                             gsd_r=getattr(cfg, "gsd_r", 3),
                             lora_rank=getattr(cfg, "lora_rank", 0),
                             lora_alpha=getattr(cfg, "lora_alpha", 16),
                             cma_chroma=getattr(cfg, "cma_chroma", False),
                             cdc_theta=getattr(cfg, "cdc_theta", 0.0)).cuda()
    model.load_state_dict(ck["model"])
    model.eval()

    ds = ManifestDataset(
        test_df.assign(label=0, attack_type="none", doc_type="u", source="freuid", split="test"),
        build_transforms("eval", cfg.img_size), with_label=False)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)

    scores, ids = [], []
    for x, idb in dl:
        with torch.autocast("cuda", enabled=cfg.amp):
            p = torch.sigmoid(model(x.cuda())).float().squeeze(1).cpu().numpy()
        scores.append(p)
        ids.extend(list(idb))

    return np.array(ids), np.concatenate(scores)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(
        description="Inference / ensemble for FREUID. Pass one --ckpt for single-model, "
                    "or multiple --ckpts for an ensemble (scores are averaged).")
    ap.add_argument("--ckpt", default=None, help="single checkpoint path (legacy)")
    ap.add_argument("--ckpts", nargs="+", default=None,
                    help="one or more checkpoint paths to ensemble")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--existing-only", action="store_true",
                    help="predict only test images present on disk (smoke tests)")
    ap.add_argument("--fill-missing", type=float, default=None,
                    help="fill submission rows whose image is not on disk with this "
                         "constant score. REQUIRED for valid Kaggle submissions while "
                         "only the 7,821 public-phase images (of 142,818 submission "
                         "ids) are published; the remaining 134,997 private-phase "
                         "images arrive later (see docs/experiments.md).")
    ap.add_argument("--val-manifest", default=None,
                    help="parquet manifest with labels to calibrate threshold on "
                         "(reports APCER@1%%BPCER; does not affect submission scores)")
    a = ap.parse_args()

    ckpt_paths = a.ckpts or ([a.ckpt] if a.ckpt else None)
    if not ckpt_paths:
        ap.error("provide --ckpt or --ckpts")

    test_df = build_freuid_test_index(existing_only=a.existing_only)
    print(f"test set: {len(test_df)} images | checkpoints: {len(ckpt_paths)}")

    all_scores = []
    ids = None
    for ckpt_path in ckpt_paths:
        print(f"  running {ckpt_path} ...")
        ck_ids, ck_scores = _run_one(ckpt_path, test_df, a.existing_only)
        if ids is None:
            ids = ck_ids
        all_scores.append(ck_scores)

    ensemble_scores = np.mean(all_scores, axis=0)

    sub = pd.DataFrame({_ID_HEADER: ids, _PRED_HEADER: ensemble_scores})
    if a.fill_missing is not None:
        full_ids = build_freuid_test_index(existing_only=False)["image_id"]
        missing = full_ids[~full_ids.isin(sub[_ID_HEADER])]
        if len(missing):
            sub = pd.concat([sub, pd.DataFrame(
                {_ID_HEADER: missing, _PRED_HEADER: a.fill_missing})], ignore_index=True)
            sub = sub.set_index(_ID_HEADER).loc[full_ids].reset_index()
            print(f"filled {len(missing)} missing-image rows with {a.fill_missing}")
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(sub)} rows)")
    print(sub.head())

    if a.val_manifest:
        val = pd.read_parquet(a.val_manifest)
        val_scores = []
        for ckpt_path in ckpt_paths:
            ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
            cfg_dict = ck.get("cfg", {})
            cfg = Config(**{k: v for k, v in cfg_dict.items() if hasattr(Config, k)})
            model = build_classifier(cfg.model_type, cfg.backbone, pretrained=False,
                                     img_size=cfg.img_size,
                                     freeze_backbone=cfg.freeze_backbone,
                                     use_chroma=getattr(cfg, "use_chroma", False),
                                     use_spectral=getattr(cfg, "use_spectral", False),
                                     cdc_theta=getattr(cfg, "cdc_theta", 0.0)).cuda()
            model.load_state_dict(ck["model"])
            model.eval()
            ds = ManifestDataset(val, build_transforms("eval", cfg.img_size))
            dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
            sc = []
            for x, _ in dl:
                with torch.autocast("cuda", enabled=cfg.amp):
                    sc.append(torch.sigmoid(model(x.cuda())).float().squeeze(1).cpu().numpy())
            val_scores.append(np.concatenate(sc))
        ens_val = np.mean(val_scores, axis=0)
        apcer = apcer_at_bpcer(val["label"].to_numpy(), ens_val, bpcer=0.01)
        print(f"val APCER@1%BPCER (ensemble, {len(ckpt_paths)} ckpts): {apcer:.4f}")


if __name__ == "__main__":
    main()
