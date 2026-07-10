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

# Submission header confirmed from the real sample_submission.csv: id,label
_ID_HEADER = "id"
_PRED_HEADER = "label"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/baseline/best.pt")
    ap.add_argument("--config", default="checkpoints/baseline/config.yaml")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--existing-only", action="store_true",
                    help="output only test images present on disk (smoke tests)")
    ap.add_argument("--fill", type=float, default=0.5,
                    help="score for test ids whose image is absent (private test not on disk)")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    amp_dtype = torch.bfloat16 if getattr(cfg, "amp_dtype", "bf16") == "bf16" else torch.float16
    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    model = build_classifier(cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size, n_doctypes=getattr(cfg,"n_doctypes",0)).cuda()
    model.load_state_dict(ck["model"])
    model.eval()

    # full id list (all 142,818 sample_submission ids); predict the ones on disk,
    # fill the rest with `--fill` so the submission has every required id.
    full = build_freuid_test_index(existing_only=False)
    have = build_freuid_test_index(existing_only=True)
    have_ids = set(have["image_id"])
    print(f"test ids total={len(full)} on-disk={len(have)} missing={len(full)-len(have)}")

    ds = ManifestDataset(
        have.assign(label=0, attack_type="none", doc_type="u", source="freuid", split="test"),
        build_transforms("eval", cfg.img_size), with_label=False)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    ids, preds = [], []
    for x, idb in dl:
        with torch.autocast("cuda", dtype=amp_dtype, enabled=cfg.amp):
            p = torch.sigmoid(model(x.cuda())).float().squeeze(1).cpu().numpy()
        preds.append(np.nan_to_num(p, nan=a.fill))
        ids.extend(list(idb))
    pred_map = dict(zip(ids, np.concatenate(preds) if preds else []))

    src = full if not a.existing_only else have
    out_ids = list(src["image_id"])
    out_scores = [float(pred_map.get(i, a.fill)) for i in out_ids]
    sub = pd.DataFrame({_ID_HEADER: out_ids, _PRED_HEADER: out_scores})
    sub.to_csv(a.out, index=False)
    n_pred = sum(1 for i in out_ids if i in pred_map)
    print(f"wrote {a.out} rows={len(sub)} predicted={n_pred} filled={len(sub)-n_pred} (fill={a.fill})")
    print(sub.head())


if __name__ == "__main__":
    main()
