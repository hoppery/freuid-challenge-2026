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


class SpectralBandPerturb(A.ImageOnlyTransform):
    """FHAG — Frequency-band amplitude augmentation (E3 capture lever, 2026-06-18).

    Print/screen RECAPTURE reshapes the image's radial AMPLITUDE spectrum: low-pass blur
    attenuates high bands, sharpening/moiré injects mid-high energy, demosaicing shifts the
    band ratios. A born-digital-trained detector can shortcut on this band-energy profile
    (which differs systematically between the 99.97%-born-digital TRAIN set and the captured
    PRIVATE test). Randomly rescaling per-band amplitude (PHASE kept, so layout/forensic
    structure is preserved) at train time removes that shortcut → capture-invariant trace
    features. Applied to BOTH classes (so 'recapture present' can't become a fraud cue).
    Occasionally strongly attenuates the top band to mimic capture low-pass blur."""

    def __init__(self, n_bands: int = 4, gain=(0.5, 1.6), hf_drop_p: float = 0.3, p: float = 0.4):
        super().__init__(p=p)
        self.n_bands = n_bands
        self.gain = gain
        self.hf_drop_p = hf_drop_p

    def get_params(self):
        gains = [random.uniform(*self.gain) for _ in range(self.n_bands)]
        if random.random() < self.hf_drop_p:          # capture low-pass blur: kill the top band
            gains[-1] = random.uniform(0.1, 0.4)
        return {"gains": gains}

    def apply(self, img, gains=None, **params):
        if not gains:
            return img
        h, w = img.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        r = r / (r.max() + 1e-6)
        edges = np.linspace(0.0, 1.0, self.n_bands + 1)
        gain_field = np.ones((h, w), np.float32)
        for b in range(self.n_bands):
            hi = (r <= 1.0001) if b == self.n_bands - 1 else (r < edges[b + 1])
            gain_field[(r >= edges[b]) & hi] = gains[b]
        out = np.empty_like(img, dtype=np.float32)
        for c in range(img.shape[2]):
            f = np.fft.fftshift(np.fft.fft2(img[:, :, c].astype(np.float32)))
            rec = np.fft.ifft2(np.fft.ifftshift(f * gain_field)).real
            out[:, :, c] = rec
        return np.clip(out, 0, 255).astype(np.uint8)


def recapture_transforms(p_scale: float = 1.0):
    """Composite recapture-trace augmentation block (for use inside a larger Compose)."""
    return [
        JpegRecompress(p=0.6 * p_scale),
        MoireArtifact(p=0.35 * p_scale),
        DownscaleUpscale(p=0.4 * p_scale),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3 * p_scale),
        A.GaussianBlur(blur_limit=(3, 5), p=0.25 * p_scale),
    ]


def fhag_transforms(p_scale: float = 1.0):
    """FHAG frequency-band amplitude perturbation block (E3 capture lever)."""
    return [SpectralBandPerturb(n_bands=4, gain=(0.5, 1.6), hf_drop_p=0.3, p=0.4 * p_scale)]
