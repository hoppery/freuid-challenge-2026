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


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="model_type"):
        build_classifier("nope", pretrained=False)


def test_base_build_model_default():
    m = build_model(pretrained=False)
    assert m(torch.randn(1, 3, 96, 96)).shape == (1, 1)
