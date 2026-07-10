import sys, numpy as np, torch
from torch.utils.data import DataLoader
from freuid.config import Config
from freuid.data.schema import read_manifest
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
sys.path.insert(0, 'src')
from freuid.metrics import compute_metrics

val = read_manifest('manifests/fantasyid_test.parquet').sort_values('path').reset_index(drop=True)
y = val['label'].to_numpy()
for tag, d, ep in [('E6+FDA@896', 'exp_e6_fda', 2), ('E6+FDA@1120', 'exp_e6_fda1120', 2)]:
    try:
        ck = torch.load(f'checkpoints/{d}/epoch{ep}.pt', map_location='cuda', weights_only=False)
        cfg = Config(**{k: v for k, v in ck.get('cfg', {}).items() if hasattr(Config, k)})
        m = build_classifier(cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size).cuda()
        m.load_state_dict(ck['model']); m.eval()
        dl = DataLoader(ManifestDataset(val, build_transforms('eval', cfg.img_size)),
                        batch_size=cfg.batch_size, shuffle=False, num_workers=8)
        with torch.no_grad():
            s = np.concatenate([torch.sigmoid(m(x.cuda())).float().squeeze(1).cpu().numpy() for x, _ in dl])
        mm = compute_metrics(y, s)
        print(f"  {tag}: FREUID={mm['freuid_score']:.4f} AUC={1-mm['audet']:.4f} APCER@1%={mm['apcer_at_1pct_bpcer']:.4f}")
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  {tag}: ERR {str(e)[:70]}")
