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


def freuid_score(audet: float, apcer_1pct: float) -> float:
    """Official competition metric (Kaggle Evaluation page, quoted verbatim):
        g_audet = 1 - AuDET
        g_apcer = 1 - APCER@1%BPCER
        FREUID  = 1 - 2*g_audet*g_apcer / (g_audet + g_apcer)
    i.e. 1 minus the harmonic mean of the two goodness scores. Bounded [0,1],
    LOWER is better. The harmonic mean is dragged toward the worse component,
    so the strict APCER@1%BPCER operating point dominates when it is weak."""
    g1, g2 = 1.0 - audet, 1.0 - apcer_1pct
    if g1 + g2 <= 0:
        return 1.0
    return float(1.0 - 2.0 * g1 * g2 / (g1 + g2))


def compute_metrics(y_true, scores) -> dict:
    audet = audet_linear(y_true, scores)
    apcer1 = apcer_at_bpcer(y_true, scores, 0.01)
    return {
        "audet": audet,
        "apcer_at_1pct_bpcer": apcer1,
        "freuid_score": freuid_score(audet, apcer1),
        "roc_auc": float(roc_auc_score(y_true, scores)),
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
