"""Targeted field-manipulation forgery synthesis (v2, 2026-06-30: character-aware copy-move).

Generates HARD, near-decision-boundary text-field tampering on a GENUINE born-digital ID -> synthetic
attack (label 1). v2 upgrades the copy-move from a random horizontal strip (v1) to a CHARACTER-LEVEL
copy-move: detect glyph-like connected components (morphology, no OCR engine) and substitute one character
with another of similar size from elsewhere on the doc. This mimics real field-value tampering (e.g. changing
a digit in a date/number) far more faithfully than a random strip — closer to the DeepID-winner "OCR-matched
copy-move" signal. The hardest public cases were exactly this class.

Discriminative signal preserved from v1 (the part that worked -> public 0.0103 rank9 on ViT-B, and the
ViT-L champion 0.00145): a LOCALIZED JPEG COMPRESSION MISMATCH on the tampered region (fake QF~70-88 vs host
~95) + alpha-FEATHERED blend (suppress copy-paste edge -> near-boundary hard positive). cv2-only, fast.

v4 (copy-move + strikethrough together) REGRESSED on public (0.00199 vs v3 0.00145), so v2 is copy-move-centric:
char-copymove is the dominant mode; strike/overwrite are rare fallbacks for robustness.
"""
from __future__ import annotations
import random
import cv2
import numpy as np


def _rand_field_box(h, w):
    """A small text-field-sized horizontal strip, biased to the right (the data zone of an ID)."""
    fh = int(h * random.uniform(0.03, 0.09))
    fw = int(w * random.uniform(0.12, 0.42))
    fh, fw = max(8, fh), max(20, fw)
    x0 = random.randint(int(w * 0.33), max(int(w * 0.33), w - fw - 1))
    y0 = random.randint(int(h * 0.08), max(int(h * 0.08), h - fh - 1))
    return x0, y0, fw, fh


def _feather_mask(fw, fh, border):
    m = np.ones((fh, fw), np.float32)
    b = max(1, border)
    ramp = np.linspace(0, 1, b)
    m[:b, :] *= ramp[:, None]; m[-b:, :] *= ramp[::-1, None]
    m[:, :b] *= ramp[None, :]; m[:, -b:] *= ramp[None, ::-1]
    return m


def _jpeg(region, qf):
    ok, enc = cv2.imencode(".jpg", region, [int(cv2.IMWRITE_JPEG_QUALITY), int(qf)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else region


def _glyph_boxes(img):
    """Detect glyph-like connected components (chars) in the right data-zone. cv2 morphology, no OCR."""
    h, w = img.shape[:2]
    zone = img[:, int(w * 0.30):]                       # right ~70% = data fields
    gray = cv2.cvtColor(zone, cv2.COLOR_RGB2GRAY)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    boxes = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        # glyph-like: char-height band, not too wide/tall, enough ink
        if h * 0.012 < hh < h * 0.06 and w * 0.004 < ww < w * 0.05 and area > 8:
            boxes.append((x + int(w * 0.30), y, ww, hh))   # back to full-image coords
    return boxes


def _char_copymove(out):
    """Substitute one glyph with a similar-sized glyph copied from elsewhere (realistic field tamper).
    Returns (box, ok): box = (x0,y0,fw,fh) of the edited region for downstream JPEG/feather, ok=success."""
    boxes = _glyph_boxes(out)
    if len(boxes) < 6:
        return None, False
    random.shuffle(boxes)
    tx, ty, tw, th = boxes[0]                            # target glyph to overwrite
    cand = [(x, y, ww, hh) for (x, y, ww, hh) in boxes[1:]
            if abs(hh - th) <= max(2, th * 0.30) and abs(ww - tw) <= max(2, tw * 0.6)]
    if not cand:
        return None, False
    sx, sy, sw, sh = random.choice(cand)                # source glyph of similar size
    src = out[sy:sy + sh, sx:sx + sw]
    if src.size == 0:
        return None, False
    src = cv2.resize(src, (tw, th), interpolation=cv2.INTER_AREA)
    out[ty:ty + th, tx:tx + tw] = src                   # char substitution
    pad = max(2, int(th * 0.4))
    H, W = out.shape[:2]
    x0 = max(0, tx - pad); y0 = max(0, ty - pad)
    fw = min(W - x0, tw + 2 * pad); fh = min(H - y0, th + 2 * pad)
    return (x0, y0, fw, fh), True


def _multiglyph_copymove(out):
    """v3: substitute a RUN of adjacent glyphs (a whole field VALUE — date/number/name) with a same-width
    run from another row. More realistic + more diverse than single-char (decorrelates from the v2 model)."""
    boxes = _glyph_boxes(out)
    if len(boxes) < 12:
        return None, False
    # group glyphs into rows by y-overlap, sort each row by x
    boxes.sort(key=lambda b: (b[1], b[0]))
    rows = []
    for b in boxes:
        placed = False
        for r in rows:
            ry = r[0][1]; rh = r[0][3]
            if abs(b[1] - ry) <= rh * 0.6:
                r.append(b); placed = True; break
        if not placed:
            rows.append([b])
    rows = [sorted(r, key=lambda b: b[0]) for r in rows if len(r) >= 4]
    if len(rows) < 2:
        return None, False
    random.shuffle(rows)
    trow = rows[0]
    k = random.randint(2, min(5, len(trow)))             # run length = whole field value
    i = random.randint(0, len(trow) - k)
    run = trow[i:i + k]
    tx = run[0][0]; ty = min(g[1] for g in run); tw = run[-1][0] + run[-1][2] - tx
    th = max(g[1] + g[3] for g in run) - ty
    # source run of similar total width from another row
    for srow in rows[1:]:
        if len(srow) < k:
            continue
        j = random.randint(0, len(srow) - k)
        srun = srow[j:j + k]
        sx = srun[0][0]; sy = min(g[1] for g in srun); sw = srun[-1][0] + srun[-1][2] - sx
        sh = max(g[1] + g[3] for g in srun) - sy
        if sw < 4 or sh < 4:
            continue
        src = out[sy:sy + sh, sx:sx + sw]
        if src.size == 0:
            continue
        out[ty:ty + th, tx:tx + tw] = cv2.resize(src, (tw, th), interpolation=cv2.INTER_AREA)
        pad = max(2, int(th * 0.4)); H, W = out.shape[:2]
        x0 = max(0, tx - pad); y0 = max(0, ty - pad)
        fw = min(W - x0, tw + 2 * pad); fh = min(H - y0, th + 2 * pad)
        return (x0, y0, fw, fh), True
    return None, False


def field_tamper(img):
    """img: uint8 RGB HxWx3 (genuine). Returns a field-tampered RGB image (a synthetic attack)."""
    h, w = img.shape[:2]
    out = img.copy()

    # PRIMARY: glyph-level copy-move (realistic field tamper). v3 multi-glyph (whole field value) ~half the
    # time, else single-glyph (v2). Fallback to v1 strip modes. Mixed synthesis = more diverse positives.
    box = None
    r = random.random()
    if r < 0.4:
        box, ok = _multiglyph_copymove(out)
        if not ok:
            box, ok = _char_copymove(out)
        if not ok:
            box = None
    elif r < 0.75:
        box, ok = _char_copymove(out)
        if not ok:
            box = None
    if box is None:
        x0, y0, fw, fh = _rand_field_box(h, w)
        field = out[y0:y0 + fh, x0:x0 + fw].copy()
        mode = random.choice(["copymove", "overwrite", "strike"])
        if mode == "copymove":
            dy = random.choice([-1, 1]) * random.randint(fh, max(fh, int(h * 0.12)))
            sy = int(np.clip(y0 + dy, 0, h - fh))
            src = out[sy:sy + fh, x0:x0 + fw]
            if src.shape == field.shape:
                field = src.copy()
        elif mode == "overwrite":
            dx = random.choice([-1, 1]) * random.randint(int(fw * 0.3), fw)
            sx = int(np.clip(x0 + dx, 0, w - fw))
            src = out[y0:y0 + fh, sx:sx + fw]
            if src.shape == field.shape:
                field = (0.5 * field + 0.5 * src).astype(np.uint8)
        else:  # strike
            yc = fh // 2 + random.randint(-fh // 6, fh // 6)
            thick = random.randint(1, max(1, fh // 8))
            col = int(np.clip(field.mean() - random.uniform(40, 110), 0, 255))
            cv2.line(field, (2, yc), (fw - 2, yc), (col, col, col), thick, cv2.LINE_AA)
        out[y0:y0 + fh, x0:x0 + fw] = field
        box = (x0, y0, fw, fh)

    # ★ localized low-QF JPEG recompression on the tampered region (the compression-mismatch signal)
    x0, y0, fw, fh = box
    region = out[y0:y0 + fh, x0:x0 + fw]
    region = _jpeg(region, random.randint(70, 88))
    # alpha-feathered blend back (suppress edge -> hard, near-boundary positive)
    border = max(2, int(min(fw, fh) * random.uniform(0.08, 0.2)))
    m = _feather_mask(fw, fh, border)[..., None]
    out[y0:y0 + fh, x0:x0 + fw] = (m * region + (1 - m) * out[y0:y0 + fh, x0:x0 + fw]).astype(np.uint8)
    return out
