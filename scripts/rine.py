"""RINE-style detector (Koutlis & Papadopoulos, ECCV-W'24): a FROZEN DINOv2 backbone whose INTERMEDIATE
encoder-layer CLS tokens are aggregated by learned importance weights, then a tiny head classifies.
Intermediate layers carry low/mid-level FORENSIC cues that the final semantic layer discards — the lever
for the hard born-digital GenAI attacks our final-CLS models can't confidently flag. Only the layer-weights
+ head train (backbone frozen) → fast, OOD-robust (no catastrophic forgetting).

Modes: train (on FREUID all-types) | infer (write a submission CSV).
"""
from __future__ import annotations
import argparse, numpy as np, pandas as pd, torch, torch.nn as nn, timm
from torch.utils.data import DataLoader
from freuid.data.schema import read_manifest
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.data.adapters.freuid import build_freuid_test_index


class RINE(nn.Module):
    def __init__(self, backbone="vit_large_patch14_reg4_dinov2.lvd142m", img_size=896, n_layers=8):
        super().__init__()
        self.bb = timm.create_model(backbone, pretrained=True, num_classes=0, img_size=img_size)
        for p in self.bb.parameters():
            p.requires_grad = False
        self.bb.eval()
        d = self.bb.num_features
        self.n_layers = n_layers
        self.layer_w = nn.Parameter(torch.zeros(n_layers))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, x):
        with torch.no_grad():
            feats = self.bb.get_intermediate_layers(x, n=self.n_layers, return_prefix_tokens=True)
        # each feat = (patch_tokens, prefix_tokens); prefix = [CLS, reg1..4], CLS at index 0
        cls = torch.stack([f[1][:, 0] for f in feats], dim=1)  # (B, n_layers, d)
        w = torch.softmax(self.layer_w, dim=0).view(1, -1, 1)
        agg = (cls.float() * w).sum(1)                          # (B, d)
        return self.head(agg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "infer"], required=True)
    ap.add_argument("--ckpt", default="checkpoints/rine/best.pt")
    ap.add_argument("--img", type=int, default=896)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--out", default="submission_rine.csv")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--backbone", default="vit_large_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda"

    if a.mode == "train":
        import os; os.makedirs("checkpoints/rine", exist_ok=True)
        df = read_manifest("manifests/freuid.parquet")
        # random 12% val
        rng = np.random.default_rng(42); idx = rng.permutation(len(df)); nv = int(0.12 * len(df))
        val, tr = df.iloc[idx[:nv]], df.iloc[idx[nv:]]
        m = RINE(backbone=a.backbone, img_size=a.img, n_layers=a.layers).to(dev)
        tl = DataLoader(ManifestDataset(tr, build_transforms("heavy", a.img)), batch_size=24,
                        shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
        vl = DataLoader(ManifestDataset(val, build_transforms("eval", a.img)), batch_size=24,
                        shuffle=False, num_workers=12, pin_memory=True)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.05)
        lossfn = nn.BCEWithLogitsLoss()
        from freuid.metrics import compute_metrics
        best = 1e9
        for ep in range(a.epochs):
            m.train()
            for x, y in tl:
                x, y = x.to(dev), y.float().to(dev).unsqueeze(1)
                opt.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = lossfn(m(x), y)
                loss.backward(); opt.step()
            m.eval(); sc = []; ys = []
            with torch.no_grad():
                for x, y in vl:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        sc.append(torch.sigmoid(m(x.to(dev))).float().squeeze(1).cpu().numpy())
                    ys.append(y.numpy())
            mt = compute_metrics(np.concatenate(ys), np.concatenate(sc))
            print(f"ep{ep} FREUID={mt['freuid_score']:.4f} AUC={mt['roc_auc']:.4f} "
                  f"APCER@1%={mt['apcer_at_1pct_bpcer']:.4f} layer_w={torch.softmax(m.layer_w,0).detach().cpu().numpy().round(3)}")
            if mt["freuid_score"] < best:
                best = mt["freuid_score"]
                torch.save({"model": m.state_dict(), "img": a.img, "layers": a.layers, "backbone": a.backbone}, a.ckpt)
                print(f"  saved best {best:.4f}")
    else:
        ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
        m = RINE(backbone=ck.get("backbone","vit_large_patch14_reg4_dinov2.lvd142m"), img_size=ck["img"], n_layers=ck.get("layers",8)).to(dev); m.load_state_dict(ck["model"]); m.eval()
        test = build_freuid_test_index(existing_only=True)
        dl = DataLoader(ManifestDataset(test, build_transforms("eval", ck["img"]), with_label=False),
                        batch_size=24, shuffle=False, num_workers=12, pin_memory=True)
        sc = []
        with torch.no_grad():
            for batch in dl:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    sc.append(torch.sigmoid(m(x.to(dev))).float().squeeze(1).cpu().numpy())
        s = np.concatenate(sc)
        sub = pd.DataFrame({"id": test["image_id"].to_numpy(), "label": s})
        full = build_freuid_test_index(existing_only=False)["image_id"]
        miss = full[~full.isin(sub["id"])]
        sub = pd.concat([sub, pd.DataFrame({"id": miss, "label": 0.5})], ignore_index=True)
        sub = sub.set_index("id").loc[full].reset_index()
        sub.to_csv(a.out, index=False)
        print(f"wrote {a.out} ({len(sub)} rows, {len(s)} predicted)")


if __name__ == "__main__":
    main()
