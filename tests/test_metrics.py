import numpy as np
import pytest
from freuid.metrics import apcer_at_bpcer, audet_linear, det_curve, compute_metrics


def test_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])  # attacks score higher
    assert apcer_at_bpcer(y, s, bpcer=0.01) == pytest.approx(0.0, abs=1e-6)
    assert audet_linear(y, s) == pytest.approx(0.0, abs=1e-6)


def test_random_scores_audet_near_half():
    rng = np.random.default_rng(0)
    y = np.array([0] * 5000 + [1] * 5000)
    s = rng.random(10000)
    assert audet_linear(y, s) == pytest.approx(0.5, abs=0.03)


def test_apcer_monotone_in_bpcer():
    rng = np.random.default_rng(1)
    y = np.array([0] * 2000 + [1] * 2000)
    s = np.concatenate([rng.normal(0.4, 0.1, 2000), rng.normal(0.6, 0.1, 2000)])
    a_strict = apcer_at_bpcer(y, s, bpcer=0.01)   # low BPCER -> high APCER
    a_loose = apcer_at_bpcer(y, s, bpcer=0.10)
    assert 0.0 <= a_loose <= a_strict <= 1.0


def test_det_curve_shapes():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.2, 0.4, 0.6, 0.8])
    bpcer, apcer = det_curve(y, s)
    assert bpcer.shape == apcer.shape
    assert bpcer.min() >= 0 and bpcer.max() <= 1


def test_compute_metrics_keys():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, s)
    assert {"audet", "apcer_at_1pct_bpcer", "roc_auc"} <= set(m)
