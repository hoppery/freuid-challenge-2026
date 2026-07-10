"""Self-blended document forgery (SBI/NSA-inspired) — generate a synthetic ATTACK from a GENUINE
document by blending a self-sourced region back with a photometric/geometric inconsistency. Teaches
the detector a DOC-AGNOSTIC forgery cue (local appearance/blend inconsistency) instead of doc-type or
generator-fingerprint shortcuts → generalizes to UNSEEN doc types (the prize axis). SelfMAD evidence:
self-blend alone (no real forgeries) generalizes to unseen morph/swap attacks.

Mimics FREUID attack types doc-agnostically: field-alteration / photo-substitution / splicing =
a local region whose appearance/illumination is subtly inconsistent with the rest + a blend seam.
"""
from __future__ import annotations
import numpy as np
import cv2


def _rand(a, b):
    return a + np.random.rand() * (b - a)


def _photometric(patch):
    """Perturb a patch so it's subtly inconsistent (the forgery signal): color/brightness/blur/sharp."""
    p = patch.astype(np.float32)
    # per-channel gain + bias (white-balance / illumination mismatch — the strongest splicing cue)
    p = p * np.array([_rand(0.85, 1.15) for _ in range(3)], np.float32) + _rand(-12, 12)
    # hue/sat shift via HSV
    if np.random.rand() < 0.6:
        hsv = cv2.cvtColor(np.clip(p, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + _rand(-8, 8)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * _rand(0.8, 1.2), 0, 255)
        p = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    # blur OR sharpen (resolution / re-render mismatch, e.g. inpainted field vs printed text)
    if np.random.rand() < 0.5:
        k = int(_rand(1, 3)) * 2 + 1
        p = cv2.GaussianBlur(p, (k, k), 0)
    elif np.random.rand() < 0.5:
        blur = cv2.GaussianBlur(p, (0, 0), 1.0)
        p = np.clip(p * 1.5 - blur * 0.5, 0, 255)
    # mild jpeg recompression of the patch (re-saved field)
    if np.random.rand() < 0.4:
        q = int(_rand(40, 90))
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(np.clip(p, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            p = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.clip(p, 0, 255)


def _feather_mask(h, w):
    """Smooth blend mask (feathered rectangle) → realistic seam (not a hard cut-paste tell)."""
    m = np.zeros((h, w), np.float32)
    by, bx = int(h * _rand(0.05, 0.25)), int(w * _rand(0.05, 0.25))
    m[by:h - by, bx:w - bx] = 1.0
    k = max(3, int(min(h, w) * _rand(0.05, 0.2)) | 1)
    m = cv2.GaussianBlur(m, (k, k), 0)
    return (m * _rand(0.7, 1.0))[..., None]


def self_blend(img):
    """img: HxWx3 uint8 RGB (genuine). Returns a forged RGB uint8 with a local self-blend inconsistency."""
    H, W = img.shape[:2]
    out = img.astype(np.float32).copy()
    n_regions = np.random.randint(1, 3)
    for _ in range(n_regions):
        # source region (a field-sized box: documents have text fields / a portrait)
        rw, rh = int(W * _rand(0.12, 0.45)), int(H * _rand(0.08, 0.35))
        sx, sy = np.random.randint(0, max(1, W - rw)), np.random.randint(0, max(1, H - rh))
        patch = out[sy:sy + rh, sx:sx + rw].copy()
        patch = _photometric(patch)
        # optional slight scale/shift of the patch (copy-move / mis-registration)
        if np.random.rand() < 0.5:
            s = _rand(0.92, 1.08)
            patch = cv2.resize(patch, (max(2, int(rw * s)), max(2, int(rh * s))))
            patch = cv2.resize(patch, (rw, rh))
        # target location: same (in-place alteration) or shifted (copy-move/splice)
        if np.random.rand() < 0.5:
            tx, ty = sx, sy
        else:
            tx = np.clip(sx + int(_rand(-0.15, 0.15) * W), 0, W - rw)
            ty = np.clip(sy + int(_rand(-0.15, 0.15) * H), 0, H - rh)
        m = _feather_mask(rh, rw)
        out[ty:ty + rh, tx:tx + rw] = m * patch + (1 - m) * out[ty:ty + rh, tx:tx + rw]
    return np.clip(out, 0, 255).astype(np.uint8)
