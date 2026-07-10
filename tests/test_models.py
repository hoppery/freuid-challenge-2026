import torch
import pytest
from freuid.models.classifier import build_classifier, build_model


def test_rgb_model_forward():
    m = build_classifier("rgb", pretrained=False)
    y = m(torch.randn(2, 3, 128, 128))
    assert y.shape == (2, 1)


def test_freq_dual_forward():
    m = build_classifier("freq_dual", pretrained=False)
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 128, 128))
    assert y.shape == (2, 1)


def test_hpf_dual_forward():
    m = build_classifier("hpf_dual", pretrained=False)
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 128, 128))
    assert y.shape == (2, 1)


def test_highpass_suppresses_lowfreq():
    from freuid.models.freq_classifier import HighPassResidual
    hpf = HighPassResidual()
    x = torch.rand(1, 3, 64, 64)
    r = hpf(x)
    # residual should have much smaller magnitude than the (low-freq-dominated) input
    assert r.abs().mean() < x.abs().mean()


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="model_type"):
        build_classifier("nope", pretrained=False)


def test_base_build_model_default():
    m = build_model(pretrained=False)
    assert m(torch.randn(1, 3, 96, 96)).shape == (1, 1)


def test_partial_unfreeze_grad_flow():
    from freuid.models.classifier import FrozenBackboneClassifier
    m = FrozenBackboneClassifier("vit_base_patch14_reg4_dinov2.lvd142m",
                                 pretrained=False, img_size=126, unfreeze_blocks=2)
    y = m(torch.randn(2, 3, 126, 126))
    y.sum().backward()
    assert m.backbone.blocks[-1].mlp.fc1.weight.grad is not None     # unfrozen: grads
    assert m.backbone.blocks[0].mlp.fc1.weight.requires_grad is False  # frozen: none
    assert m.backbone.patch_embed.proj.weight.requires_grad is False
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in m.parameters())
    assert 0 < n_train < 0.3 * n_total   # only a small fraction trains
