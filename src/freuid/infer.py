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
                    help="predict only test images present on disk (smoke tests)")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    model = build_classifier(cfg.model_type, cfg.backbone, pretrained=False).cuda()
    model.load_state_dict(ck["model"])
    model.eval()

    test = build_freuid_test_index(existing_only=a.existing_only)
    ds = ManifestDataset(
        test.assign(label=0, attack_type="none", doc_type="u", source="freuid", split="test"),
        build_transforms("eval", cfg.img_size), with_label=False)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    ids, preds = [], []
    for x, idb in dl:
        with torch.autocast("cuda", enabled=cfg.amp):
            p = torch.sigmoid(model(x.cuda())).float().squeeze(1).cpu().numpy()
        preds.append(p)
        ids.extend(list(idb))
    sub = pd.DataFrame({_ID_HEADER: ids, _PRED_HEADER: np.concatenate(preds)})
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out} rows={len(sub)}")
    print(sub.head())


if __name__ == "__main__":
    main()
