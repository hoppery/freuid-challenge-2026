"""Forensic / recapture augmentations for the print-and-capture ("analog hole") axis.

SOTA finding (docs/sota-research.md): print->rephotograph DESTROYS digital noise and
ADDS recapture trace (moiré, halftone, blur, recompression). Training detectors with
print-scan / moiré / JPEG-recompression-chain augmentation measurably improves
cross-domain robustness. CRITICAL: apply these to BOTH classes (handled by the Dataset
being class-agnostic) so the model cannot shortcut "recapture present => fraud".

Implemented as albumentations 2.x ImageOnlyTransform subclasses operating on uint8 RGB
images (placed BEFORE Normalize in the Compose).
"""
from __future__ import annotations
import random
import cv2
import numpy as np
import albumentations as A


class JpegRecompress(A.ImageOnlyTransform):
    """Multi-stage JPEG recompression chain (mimics upload/processing pipelines)."""

    def __init__(self, quality_min: int = 60, quality_max: int = 95,
                 max_stages: int = 3, p: float = 0.5):
        super().__init__(p=p)
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.max_stages = max_stages

    def get_params(self):
        n = random.randint(1, self.max_stages)
        qs = [random.randint(self.quality_min, self.quality_max) for _ in range(n)]
        return {"qualities": qs}

    def apply(self, img, qualities=(85,), **params):
        out = img
        for q in qualities:
            ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
            if not ok:
                continue
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return out


class MoireArtifact(A.ImageOnlyTransform):
    """Additive sinusoidal interference approximating screen/print moiré."""

    def __init__(self, amplitude=(4.0, 18.0), freq=(0.05, 0.5), p: float = 0.4):
        super().__init__(p=p)
        self.amplitude = amplitude
        self.freq = freq

    def get_params(self):
        return {
            "amp": random.uniform(*self.amplitude),
            "f": random.uniform(*self.freq),
            "theta": random.uniform(0, np.pi),
            "phase": random.uniform(0, 2 * np.pi),
        }

    def apply(self, img, amp=8.0, f=0.2, theta=0.0, phase=0.0, **params):
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        pattern = amp * np.sin(2 * np.pi * f * (xx * np.cos(theta) + yy * np.sin(theta)) + phase)
        out = img.astype(np.float32) + pattern[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)


class DownscaleUpscale(A.ImageOnlyTransform):
    """Resolution jitter: downscale then upscale to mimic capture-device resolution loss."""

    def __init__(self, scale=(0.25, 0.7), p: float = 0.4):
        super().__init__(p=p)
        self.scale = scale

    def get_params(self):
        interps = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_CUBIC]
        return {
            "s": random.uniform(*self.scale),
            "down_i": random.choice(interps),
            "up_i": random.choice(interps),
        }

    def apply(self, img, s=0.5, down_i=cv2.INTER_AREA, up_i=cv2.INTER_LINEAR, **params):
        h, w = img.shape[:2]
        nh, nw = max(8, int(h * s)), max(8, int(w * s))
        small = cv2.resize(img, (nw, nh), interpolation=down_i)
        return cv2.resize(small, (w, h), interpolation=up_i)


def recapture_transforms(p_scale: float = 1.0):
    """Composite recapture-trace augmentation block (for use inside a larger Compose).

    WARNING: at full strength this DESTROYS FREUID's fine-grained digital-forgery cue
    (empirically: AUC collapses to 0.50, model can't learn). Use recapture_transforms_medium
    for this task; keep this only for tasks dominated by the print-recapture axis.
    """
    return [
        JpegRecompress(p=0.6 * p_scale),
        MoireArtifact(p=0.35 * p_scale),
        DownscaleUpscale(p=0.4 * p_scale),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3 * p_scale),
        A.GaussianBlur(blur_limit=(3, 5), p=0.25 * p_scale),
    ]


def recapture_transforms_medium():
    """Gentle recapture aug that PRESERVES fine detail (subtle digital-edit cues survive).
    Low probabilities + mild magnitudes: most images pass through near-clean; a minority
    get a single mild degradation for cross-domain robustness."""
    return [
        JpegRecompress(quality_min=75, quality_max=95, max_stages=1, p=0.3),
        MoireArtifact(amplitude=(2.0, 7.0), freq=(0.05, 0.3), p=0.15),
        DownscaleUpscale(scale=(0.6, 0.9), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.04), p=0.15),
    ]


def recapture_transforms_recap():
    """Capture-realistic tier targeting the host-confirmed private emphasis: 'robust to
    physical capture pipelines, print-and-capture effects, lighting and imaging variation...
    real-world conditions that may suppress fragile digital artifacts.' Between `medium` and
    `heavy`: more capture/lighting variation but cue-preserving (every op p<1 leaves a clean
    path; OneOf groups avoid stacking degradations). Label-agnostic (Dataset is class-agnostic),
    so it cannot teach 'recapture => fraud'. MUST be LODO-gated (heavy-strength destroys the cue)."""
    return [
        A.Perspective(scale=(0.02, 0.05), keep_size=True, p=0.3),          # capture geometry
        A.RandomGamma(gamma_limit=(85, 120), p=0.4),                       # display/print gamma
        A.RandomToneCurve(scale=0.10, p=0.3),                             # tone remap
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.4),  # lighting
        A.OneOf([                                                          # one capture artifact, not stacked
            MoireArtifact(amplitude=(3.0, 10.0), freq=(0.08, 0.35), p=1.0),   # screen moiré
            DownscaleUpscale(scale=(0.5, 0.85), p=1.0),                       # resolution loss
        ], p=0.35),
        A.OneOf([                                                          # sensor noise
            A.GaussNoise(std_range=(0.02, 0.06), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.1, 0.4), p=1.0),
        ], p=0.3),
        JpegRecompress(quality_min=65, quality_max=92, max_stages=2, p=0.5),  # capture recompression
    ]
