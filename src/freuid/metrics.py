from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def det_curve(y_true, scores):
    """Return (bpcer, apcer) arrays as the threshold sweeps. score = P(attack)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    bpcer = fpr            # bona-fide (0) scored >= thr (false reject)
    apcer = 1.0 - tpr      # attack (1) scored < thr (false accept)
    return bpcer, apcer


def apcer_at_bpcer(y_true, scores, bpcer: float = 0.01) -> float:
    """APCER at the operating point where BPCER == `bpcer`."""
    b, a = det_curve(y_true, scores)
    order = np.argsort(b)
    b, a = b[order], a[order]
    return float(np.interp(bpcer, b, a))


def audet_linear(y_true, scores) -> float:
    """Area under DET (BPCER, APCER) in linear space = 1 - ROC_AUC. Lower is better."""
    return float(1.0 - roc_auc_score(y_true, scores))


def freuid_score(audet: float, apcer: float) -> float:
    """Official competition score (reverse-engineered; LOWER = better).

    Kaggle describes it as "the harmonic mean of converted goodness scores" of AuDET and
    APCER@1%BPCER, lower=better: score = 1 - HM(1-AuDET, 1-APCER). Consistent with the
    observed leaderboard anchors: a constant submission (AuDET .5, APCER ~1) scores ~1.0
    and a perfect one scores 0.0. NOTE: harmonic mean is dominated by the WORSE goodness,
    so a bad APCER@1%BPCER craters the score even with a decent AuDET.
    """
    g1, g2 = 1.0 - audet, 1.0 - apcer
    if g1 <= 0 or g2 <= 0:
        return 1.0
    return float(1.0 - 2.0 * g1 * g2 / (g1 + g2))


def apcer_surrogate_loss(logits, targets, bpcer: float = 0.01, beta: float = 10.0):
    """Differentiable APCER@BPCER surrogate (iteration #24).

    The FREUID Score is dominated by APCER@1%BPCER: at the threshold where 1% of
    bona-fides score higher, what fraction of attacks score lower. BCE optimizes average
    separation, NOT this operating point — a few hard attacks ranking below the bona-fide
    tail crater the score while AuDET stays good. This loss targets the operating point
    directly: take the soft (1-bpcer) quantile of bona-fide scores as the threshold, then
    penalize every attack scoring below it via a smooth hinge. Combined with BCE.

    logits: (B,1) raw; targets: (B,1) in {0,1}.
    """
    import torch
    s = torch.sigmoid(logits).squeeze(1)
    y = targets.squeeze(1)
    bona, atk = s[y == 0], s[y == 1]
    if bona.numel() < 2 or atk.numel() < 1:
        return logits.sum() * 0.0
    # soft high quantile of bona-fide scores = the BPCER operating threshold
    k = max(1, int(round(bpcer * bona.numel())))
    thr = torch.topk(bona, k).values.min()           # ~ (1-bpcer) quantile
    # smooth hinge: attacks below the threshold are penalized (want them ABOVE)
    return torch.nn.functional.softplus(beta * (thr - atk)).mean() / beta


def compute_metrics(y_true, scores) -> dict:
    audet = audet_linear(y_true, scores)
    apcer = apcer_at_bpcer(y_true, scores, 0.01)
    return {
        "audet": audet,
        "apcer_at_1pct_bpcer": apcer,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "freuid_score": freuid_score(audet, apcer),
    }


def metrics_by_group(y_true, scores, groups) -> dict:
    """Per-group (e.g. attack_type, doc_type) metrics for diagnostics."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    groups = np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        mask = groups == g
        yt = y_true[mask]
        if len(np.unique(yt)) < 2:   # need both classes to score
            continue
        out[str(g)] = compute_metrics(yt, scores[mask])
    return out
