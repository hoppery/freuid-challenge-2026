from __future__ import annotations
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from freuid.config import Config
from freuid.data.schema import read_manifest
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.metrics import compute_metrics, metrics_by_group
from freuid.train import split


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/baseline/best.pt")
    ap.add_argument("--config", default="checkpoints/baseline/config.yaml")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    model = build_classifier(cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size).cuda()
    model.load_state_dict(ck["model"])
    model.eval()

    df = read_manifest(cfg.manifest)
    _, val = split(df, cfg)
    vl = DataLoader(ManifestDataset(val, build_transforms("eval", cfg.img_size)),
                    batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    scores = []
    for x, _ in vl:
        with torch.autocast("cuda", enabled=cfg.amp):
            scores.append(torch.sigmoid(model(x.cuda())).float().squeeze(1).cpu().numpy())
    s = np.concatenate(scores)
    y = val["label"].to_numpy()
    m = compute_metrics(y, s)
    m["by_attack_type"] = metrics_by_group(y, s, val["attack_type"].to_numpy())
    m["by_doc_type"] = metrics_by_group(y, s, val["doc_type"].to_numpy())
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
