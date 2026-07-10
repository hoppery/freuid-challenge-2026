"""Frequency-aware dual-stream classifier.

SOTA rationale (docs/sota-research.md §2): print-and-capture destroys digital noise
residuals but ADDS structured frequency artifacts (moiré/halftone = periodic FFT peaks).
DCT/FFT spectral features are the strongest single cue that survives (and exploits) the
analog hole, and are more domain-stable than RGB. This model fuses an RGB backbone with a
spectral branch that consumes the log-FFT magnitude computed on-GPU from the input (no
data-pipeline change needed).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import timm


class FreqEncoder(nn.Module):
    """Small CNN over a 1-channel log-FFT-magnitude map -> feature vector."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x):
        return self.proj(self.net(x))


class FreqDualStream(nn.Module):
    """RGB backbone (timm, pooled) + spectral branch, fused into a single logit."""

    def __init__(self, backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
                 pretrained: bool = True, drop_rate: float = 0.1, freq_dim: int = 256):
        super().__init__()
        self.rgb = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                     drop_rate=drop_rate)
        rgb_dim = self.rgb.num_features
        self.freq = FreqEncoder(freq_dim)
        self.head = nn.Sequential(
            nn.Linear(rgb_dim + freq_dim, 512), nn.GELU(), nn.Dropout(drop_rate),
            nn.Linear(512, 1),
        )

    @staticmethod
    def _log_fft(x):
        """x: (B,3,H,W) -> (B,1,H,W) per-sample-standardized log-FFT magnitude."""
        gray = x.mean(dim=1, keepdim=True)
        f = torch.fft.fft2(gray.float())
        f = torch.fft.fftshift(f, dim=(-2, -1))
        mag = torch.log1p(f.abs())
        mu = mag.mean(dim=(-2, -1), keepdim=True)
        sd = mag.std(dim=(-2, -1), keepdim=True) + 1e-6
        return ((mag - mu) / sd).to(x.dtype)

    def forward(self, x):
        fa = self.rgb(x)
        fb = self.freq(self._log_fft(x))
        return self.head(torch.cat([fa, fb], dim=1))


def build_freq_model(backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
                     pretrained: bool = True, drop_rate: float = 0.1):
    return FreqDualStream(backbone, pretrained=pretrained, drop_rate=drop_rate)


# ---------------------------------------------------------------------------
# V2: high-pass RESIDUAL dual-stream (fixes the exp_freq failure).
# Root cause of exp_freq failure: GLOBAL FFT magnitude encodes the document's
# layout/font/security-pattern spectrum, which is DOC-TYPE specific -> the freq
# branch learned a doc-type shortcut and failed on held-out doc-types (AUC ~0.55).
# Fix: feed a LOCAL high-pass residual (image minus its Gaussian blur) instead of the
# global spectrum. The residual suppresses low-frequency layout/structure and keeps
# the local forensic high-frequency signal (recapture moiré/halftone, compression,
# splice edges) that should transfer across doc-types.
# ---------------------------------------------------------------------------

def _gaussian_kernel(ksize: int = 5, sigma: float = 1.0):
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    k2d = torch.outer(g, g)
    return k2d


class HighPassResidual(nn.Module):
    """Fixed (non-learned) per-channel high-pass: x - gaussian_blur(x)."""

    def __init__(self, ksize: int = 5, sigma: float = 1.0):
        super().__init__()
        k = _gaussian_kernel(ksize, sigma)
        kernel = k.view(1, 1, ksize, ksize).repeat(3, 1, 1, 1)
        self.register_buffer("kernel", kernel)
        self.pad = ksize // 2

    def forward(self, x):
        blur = nn.functional.conv2d(x, self.kernel.to(x.dtype), padding=self.pad, groups=3)
        return x - blur


class HPFDualStream(nn.Module):
    """RGB backbone + high-pass-residual forensic branch, fused to a single logit."""

    def __init__(self, backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
                 pretrained: bool = True, drop_rate: float = 0.1, res_dim: int = 256):
        super().__init__()
        self.rgb = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                     drop_rate=drop_rate)
        rgb_dim = self.rgb.num_features
        self.hpf = HighPassResidual(ksize=5, sigma=1.0)
        self.res_enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, res_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(rgb_dim + res_dim, 512), nn.GELU(), nn.Dropout(drop_rate),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        fa = self.rgb(x)
        fb = self.res_enc(self.hpf(x))
        return self.head(torch.cat([fa, fb], dim=1))


def build_hpf_model(backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
                    pretrained: bool = True, drop_rate: float = 0.1):
    return HPFDualStream(backbone, pretrained=pretrained, drop_rate=drop_rate)
