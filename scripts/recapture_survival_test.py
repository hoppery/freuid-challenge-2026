"""Does ft03's field-tamper signal survive RECAPTURE (the private captured/physical axis)?

4 conditions on N genuine born-digital train docs, scored by ft03 (fraud prob):
  GEN       genuine raw                         -> should be LOW  (genuine)
  TAMP      field_tamper(genuine)               -> should be HIGH (ft03 catches it)  [sanity]
  GEN_RC    recapture(genuine)                  -> if HIGH => ft03 false-flags captured genuine (bad for private)
  TAMP_RC   recapture(field_tamper(genuine))    -> if LOW  => tamper signal DIED under recapture (won't transfer)

Decisive: if TAMP is high but TAMP_RC collapses to ~GEN, field-tamper is a born-digital artifact that
recapture erases -> ft03 will NOT generalize to captured/physical private.
"""
import sys, random
import numpy as np, pandas as pd, cv2, torch
import albumentations as A
sys.path.insert(0, "src")
from freuid.config import Config
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.data.field_tamper import field_tamper
from freuid.data.forensic_aug import recapture_transforms

N = 200
random.seed(0); np.random.seed(0)
df = pd.read_csv("data/raw/freuid/train_labels.csv")
gen = df[df.label == 0].sample(N, random_state=0).reset_index(drop=True)

ck = torch.load("checkpoints/exp_ft03/best.pt", map_location="cuda", weights_only=False)
cfg = Config(**{k: v for k, v in ck.get("cfg", {}).items() if hasattr(Config, k)})
model = build_classifier(cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size).cuda()
model.load_state_dict(ck["model"]); model.eval()
tf = build_transforms("eval", cfg.img_size)
recap = A.Compose(recapture_transforms(1.0))

@torch.no_grad()
def score(imgs):
    out = []
    for i in range(0, len(imgs), 16):
        batch = torch.stack([tf(image=im)["image"] for im in imgs[i:i+16]]).cuda()
        with torch.autocast("cuda", enabled=True):
            p = torch.sigmoid(model(batch)).float().squeeze(1).cpu().numpy()
        out.append(p)
    return np.concatenate(out)

GEN, TAMP, GEN_RC, TAMP_RC = [], [], [], []
for _, r in gen.iterrows():
    im = cv2.imread("data/raw/freuid/train/train/" + r.id + ".jpeg")
    if im is None: continue
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    t = field_tamper(im)
    GEN.append(im); TAMP.append(t)
    GEN_RC.append(recap(image=im)["image"]); TAMP_RC.append(recap(image=t)["image"])

s = {k: score(v) for k, v in [("GEN", GEN), ("TAMP", TAMP), ("GEN_RC", GEN_RC), ("TAMP_RC", TAMP_RC)]}
print(f"\nft03 fraud-prob over {len(GEN)} genuine docs (mean | median | %flagged>0.5):")
for k in ["GEN", "TAMP", "GEN_RC", "TAMP_RC"]:
    v = s[k]; print(f"  {k:9s}: {v.mean():.3f} | {np.median(v):.3f} | {(v>0.5).mean()*100:5.1f}%")
print("\n--- DECISIVE ---")
print(f"tamper detection (born-digital): TAMP {s['TAMP'].mean():.3f} vs GEN {s['GEN'].mean():.3f}  "
      f"(gap {s['TAMP'].mean()-s['GEN'].mean():+.3f})")
print(f"tamper detection AFTER recapture: TAMP_RC {s['TAMP_RC'].mean():.3f} vs GEN_RC {s['GEN_RC'].mean():.3f}  "
      f"(gap {s['TAMP_RC'].mean()-s['GEN_RC'].mean():+.3f})")
surv = (s['TAMP_RC'].mean()-s['GEN_RC'].mean()) / max(1e-6, s['TAMP'].mean()-s['GEN'].mean())
print(f"signal SURVIVAL after recapture: {surv*100:.0f}% of born-digital gap retained")
print(f"false-flag on captured genuine: GEN_RC flags {(s['GEN_RC']>0.5).mean()*100:.1f}% as fraud (vs GEN {(s['GEN']>0.5).mean()*100:.1f}%)")
