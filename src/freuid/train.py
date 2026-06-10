from __future__ import annotations
import argparse
import csv
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from freuid.config import Config
from freuid.data.schema import read_manifest
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.metrics import compute_metrics, metrics_by_group


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def split(df, cfg: Config):
    """Domain-holdout split by cfg.holdout_by, else random."""
    rng = np.random.default_rng(cfg.seed)
    if cfg.holdout_by and cfg.holdout_by in df and df[cfg.holdout_by].nunique() > 1:
        groups = np.asarray(df[cfg.holdout_by].astype(str).unique(), dtype=object)
        rng.shuffle(groups)
        n_val = max(1, int(len(groups) * cfg.val_fraction))
        val_groups = set(groups[:n_val])
        val = df[df[cfg.holdout_by].isin(val_groups)]
        tr = df[~df[cfg.holdout_by].isin(val_groups)]
        # guard: validation needs both classes; fall back to random if not
        if val["label"].nunique() < 2 or tr["label"].nunique() < 2:
            return _random_split(df, cfg, rng)
    else:
        return _random_split(df, cfg, rng)
    return tr.reset_index(drop=True), val.reset_index(drop=True)


def _random_split(df, cfg: Config, rng):
    idx = rng.permutation(len(df))
    n_val = max(1, int(len(df) * cfg.val_fraction))
    val = df.iloc[idx[:n_val]]
    tr = df.iloc[idx[n_val:]]
    return tr.reset_index(drop=True), val.reset_index(drop=True)


@torch.no_grad()
def evaluate(model, loader, device, val_df, amp: bool):
    model.eval()
    scores = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp):
            scores.append(torch.sigmoid(model(x)).float().squeeze(1).cpu().numpy())
    s = np.concatenate(scores)
    y = val_df["label"].to_numpy()
    m = compute_metrics(y, s)
    m["by_attack_type"] = metrics_by_group(y, s, val_df["attack_type"].to_numpy())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    cfg = Config.load(ap.parse_args().config)
    set_seed(cfg.seed)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    cfg.save(f"{cfg.out_dir}/config.yaml")
    device = "cuda"

    df = read_manifest(cfg.manifest)
    tr, val = split(df, cfg)
    print(f"train={len(tr)} val={len(val)} holdout_by={cfg.holdout_by!r}")

    tl = DataLoader(ManifestDataset(tr, build_transforms("train", cfg.img_size)),
                    batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    pin_memory=True, drop_last=True)
    vl = DataLoader(ManifestDataset(val, build_transforms("eval", cfg.img_size)),
                    batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                    pin_memory=True)

    model = build_classifier(cfg.model_type, cfg.backbone, pretrained=cfg.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(enabled=cfg.amp)
    lossfn = torch.nn.BCEWithLogitsLoss()

    with open(f"{cfg.out_dir}/log.csv", "w", newline="") as log:
        w = csv.writer(log)
        w.writerow(["epoch", "train_loss", "audet", "apcer@1%bpcer", "roc_auc"])
        log.flush()
        best = 1e9
        for ep in range(cfg.epochs):
            model.train()
            tot = 0.0
            for x, y in tl:
                x = x.to(device, non_blocking=True)
                y = y.to(device).unsqueeze(1)
                opt.zero_grad()
                with torch.autocast("cuda", enabled=cfg.amp):
                    loss = lossfn(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item() * len(x)
            sched.step()
            m = evaluate(model, vl, device, val, cfg.amp)
            tl_loss = tot / max(1, len(tr))
            print(f"ep{ep} loss={tl_loss:.4f} AuDET={m['audet']:.4f} "
                  f"APCER@1%={m['apcer_at_1pct_bpcer']:.4f} AUC={m['roc_auc']:.4f}")
            print("  by attack_type:", m["by_attack_type"])
            w.writerow([ep, tl_loss, m["audet"], m["apcer_at_1pct_bpcer"], m["roc_auc"]])
            log.flush()
            score = m["apcer_at_1pct_bpcer"]   # primary selection metric (lower better)
            if score < best:
                best = score
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                            "metrics": {k: v for k, v in m.items() if k != "by_attack_type"}},
                           f"{cfg.out_dir}/best.pt")
                print(f"  saved best (APCER@1%BPCER={best:.4f})")


if __name__ == "__main__":
    main()
