import numpy as np
import torch
import pytest
from freuid.data.tracemix import TraceMix, sample_chain, sample_contrasting_chain
from freuid.models.dtc import DTCNet, _sim_stats
from freuid.models.classifier import build_classifier
import random


def _img(h=128, w=96):
    return (np.random.rand(h, w, 3) * 255).astype("uint8")


# ---------------- TraceMix ----------------

def test_tracemix_coherent_label_and_shape():
    tm = TraceMix(p_incoherent=0.0, seed=0)
    out, y = tm(_img())
    assert y == 1.0 and out.shape == (128, 96, 3) and out.dtype == np.uint8


def test_tracemix_incoherent_label_and_shape():
    tm = TraceMix(p_incoherent=1.0, seed=0)
    out, y = tm(_img())
    assert y == 0.0 and out.shape == (128, 96, 3) and out.dtype == np.uint8


def test_tracemix_incoherent_actually_modifies():
    img = _img()
    tm = TraceMix(p_incoherent=1.0, seed=0)
    out, _ = tm(img)
    assert not np.array_equal(out, img)


def test_contrasting_chain_differs_in_jpeg_and_moire():
    rng = random.Random(0)
    for _ in range(20):
        a = sample_chain(rng)
        b = sample_contrasting_chain(a, rng)
        a_q = a.jpeg_qualities[-1] if a.jpeg_qualities else 95
        assert abs(b.jpeg_qualities[-1] - a_q) >= 5
        assert b.moire is not None
        if a.moire is not None:
            assert abs(b.moire[1] - a.moire[1]) > 0.1


# ---------------- model ----------------

def test_sim_stats_shape_and_coherence_ordering():
    torch.manual_seed(0)
    # coherent: all patches near-identical -> high mean sim
    base = torch.nn.functional.normalize(torch.randn(1, 1, 16), dim=-1)
    coherent = torch.nn.functional.normalize(
        base.expand(1, 64, 16) + 0.01 * torch.randn(1, 64, 16), dim=-1)
    # incoherent: random patches -> low mean sim
    incoherent = torch.nn.functional.normalize(torch.randn(1, 64, 16), dim=-1)
    s_c, s_i = _sim_stats(coherent), _sim_stats(incoherent)
    assert s_c.shape == (1, 7) and s_i.shape == (1, 7)
    assert s_c[0, 0] > s_i[0, 0]   # mean pairwise similarity separates the two


def test_dtc_forward_default_is_fraud_logit_only():
    m = build_classifier("dtc", pretrained=False)
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 128, 128))
    assert y.shape == (2, 1)


def test_dtc_forward_with_consistency():
    m = DTCNet(pretrained=False)
    m.eval()
    with torch.no_grad():
        fraud, cons = m(torch.randn(2, 3, 128, 128), return_consistency=True)
    assert fraud.shape == (2, 1) and cons.shape == (2, 1)


def test_dtc_backward_flows_to_both_heads():
    m = DTCNet(pretrained=False)
    fraud, cons = m(torch.randn(2, 3, 96, 96), return_consistency=True)
    loss = fraud.sum() + cons.sum()
    loss.backward()
    assert m.cons_head[0].weight.grad is not None
    assert m.head[0].weight.grad is not None
    assert m.patch_proj.weight.grad is not None


# ---------------- dataset ----------------

def test_dtc_dataset_returns_triple(tmp_path):
    import cv2
    import pandas as pd
    from freuid.data.dataset import DTCTrainDataset
    p = str(tmp_path / "a.png")
    cv2.imwrite(p, _img())
    df = pd.DataFrame([{"path": p, "label": 1, "attack_type": "unknown",
                        "doc_type": "X", "source": "t", "split": "train"}])
    ds = DTCTrainDataset(df, size=64, p_incoherent=1.0)
    x, y, yc = ds[0]
    assert x.shape == (3, 64, 64)
    assert float(y) == 1.0 and float(yc) == 0.0


def test_dtc_on_partial_frozen_vit_host():
    m = build_classifier("dtc", backbone="vit_base_patch14_reg4_dinov2.lvd142m",
                         pretrained=False, img_size=126, freeze_backbone=True,
                         unfreeze_blocks=2)
    fraud, cons = m(torch.randn(2, 3, 126, 126), return_consistency=True)
    assert fraud.shape == (2, 1) and cons.shape == (2, 1)
    assert m.rgb.blocks[-1].mlp.fc1.weight.requires_grad is True
    assert m.rgb.blocks[0].mlp.fc1.weight.requires_grad is False
    # trace branch + heads always train
    assert all(p.requires_grad for p in m.trace.parameters())


def test_sbd_blend_shape_and_modification():
    from freuid.data.sbd import SelfBlendedDoc
    base, donor = _img(128, 96), _img(128, 96)
    out = SelfBlendedDoc(seed=0)(base, donor)
    assert out.shape == base.shape and out.dtype == np.uint8
    assert not np.array_equal(out, base)


def test_sbd_dataset_flips_label_and_consistency(tmp_path):
    import cv2
    import pandas as pd
    from freuid.data.dataset import DTCTrainDataset
    rows = []
    for k in range(2):
        p = str(tmp_path / f"{k}.png")
        cv2.imwrite(p, _img())
        rows.append({"path": p, "label": 0, "attack_type": "none",
                     "doc_type": "X", "source": "t", "split": "train"})
    ds = DTCTrainDataset(pd.DataFrame(rows), size=64, p_incoherent=0.0, sbd_p=1.0)
    x, y, yc = ds[0]
    assert float(y) == 1.0    # bona-fide became pseudo-attack
    assert float(yc) == 0.0   # and is trace-incoherent by construction


def test_patch_eval_dataset_returns_five_crops(tmp_path):
    import cv2
    import pandas as pd
    from freuid.data.dataset import PatchEvalDataset
    p = str(tmp_path / "a.png")
    cv2.imwrite(p, _img(300, 220))
    df = pd.DataFrame([{"path": p, "label": 1, "attack_type": "unknown",
                        "doc_type": "X", "source": "t", "split": "train"}])
    x, y = PatchEvalDataset(df, size=64)[0]
    assert x.shape == (5, 3, 64, 64) and float(y) == 1.0


def test_patch_mode_train_dataset(tmp_path):
    import cv2
    import pandas as pd
    from freuid.data.dataset import DTCTrainDataset
    p = str(tmp_path / "a.png")
    cv2.imwrite(p, _img(300, 220))
    df = pd.DataFrame([{"path": p, "label": 0, "attack_type": "none",
                        "doc_type": "X", "source": "t", "split": "train"}])
    x, y, yc = DTCTrainDataset(df, size=64, patch_mode=True)[0]
    assert x.shape == (3, 64, 64)


def test_grl_reverses_gradient():
    from freuid.models.dtc import grad_reverse
    x = torch.randn(4, 8, requires_grad=True)
    grad_reverse(x, 0.5).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x.grad, -0.5))


def test_dtc_adv_head_and_dataset(tmp_path):
    import cv2
    import pandas as pd
    from freuid.data.dataset import DTCTrainDataset
    from freuid.models.dtc import DTCNet
    rows = []
    for k, t in enumerate(["A", "B"]):
        p = str(tmp_path / f"{k}.png")
        cv2.imwrite(p, _img())
        rows.append({"path": p, "label": k, "attack_type": "unknown",
                     "doc_type": t, "source": "t", "split": "train"})
    ds = DTCTrainDataset(pd.DataFrame(rows), size=64, return_doc=True)
    x, y, yc, d = ds[1]
    assert int(d) == 1
    m = DTCNet(pretrained=False, n_doc_types=2)
    fraud, cons, doc = m(torch.randn(2, 3, 96, 96), return_consistency=True,
                         return_doc=True)
    assert fraud.shape == (2, 1) and cons.shape == (2, 1) and doc.shape == (2, 2)
    (fraud.sum() + cons.sum() + doc.sum()).backward()
    assert m.doc_head.weight.grad is not None


def test_prototype_head_forward_and_grad():
    m = build_classifier("dtc", pretrained=False, use_prototype=True)
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 96, 96))
    assert y.shape == (2, 1)
    m.train()
    fraud, cons = m(torch.randn(2, 3, 96, 96), return_consistency=True)
    (fraud.sum() + cons.sum()).backward()
    assert m.proto.grad is not None and m.proto_scale.grad is not None


def test_mixstyle_noop_eval_active_train():
    from freuid.models.dtc import MixStyle
    ms = MixStyle(p=1.0)
    x = torch.randn(4, 10, 16)
    ms.eval()
    assert torch.equal(ms(x), x)               # no-op at eval
    ms.train()
    out = ms(x)
    assert out.shape == x.shape and not torch.equal(out, x)   # active in train


def test_dtc_mixstyle_forward_vit():
    m = build_classifier("dtc", backbone="vit_base_patch14_reg4_dinov2.lvd142m",
                         pretrained=False, img_size=126, freeze_backbone=True,
                         unfreeze_blocks=2, mixstyle_p=0.5)
    m.train()
    fraud, cons = m(torch.randn(2, 3, 126, 126), return_consistency=True)
    assert fraud.shape == (2, 1) and cons.shape == (2, 1)
    fraud.sum().backward()


def test_dtc_localization_head():
    m = build_classifier("dtc", pretrained=False)
    m.train()
    fraud, cons, loc = m(torch.randn(2, 3, 96, 96), return_consistency=True, return_loc=True)
    assert fraud.shape == (2, 1) and cons.shape == (2, 1)
    assert loc.shape == (2, 64)            # 8x8 patch grid
    (fraud.sum() + cons.sum() + loc.sum()).backward()
    assert m.loc_head[0].weight.grad is not None


def test_tracemix_return_mask():
    from freuid.data.tracemix import TraceMix
    import numpy as np
    out, y, mask = TraceMix(p_incoherent=1.0, seed=0)(_img(), return_mask=True)
    assert y == 0.0 and mask.shape == (128, 96) and mask.max() > 0
    out2, y2, mask2 = TraceMix(p_incoherent=0.0, seed=0)(_img(), return_mask=True)
    assert y2 == 1.0 and mask2.max() == 0   # coherent -> empty mask


def test_dtc_loc_dataset(tmp_path):
    import cv2, pandas as pd
    from freuid.data.dataset import DTCTrainDataset
    p = str(tmp_path / "a.png"); cv2.imwrite(p, _img())
    df = pd.DataFrame([{"path": p, "label": 0, "attack_type": "none",
                        "doc_type": "X", "source": "t", "split": "train"}])
    ds = DTCTrainDataset(df, size=64, p_incoherent=1.0, loc_grid=8)
    x, y, yc, loc = ds[0]
    assert x.shape == (3, 64, 64) and loc.shape == (64,)
