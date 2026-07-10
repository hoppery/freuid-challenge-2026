import torch
import pytest
from freuid.losses import TailMarginBCE, build_loss


def _fill_bank(loss, n=512, mean=0.15, spread=0.05):
    """Seed the bona-fide bank with prob-space scores (default space='prob')."""
    bona = (torch.randn(n) * spread + mean).clamp(0.001, 0.999)
    loss._update_bank(bona)


def test_falls_back_to_bce_when_bank_empty():
    lf = TailMarginBCE(min_bank=256, warmup_steps=0)
    logits = torch.randn(8, 1)
    targets = torch.randint(0, 2, (8, 1)).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.flatten(), targets.flatten())
    out = lf(logits, targets)  # this batch's bona-fide alone < min_bank
    assert torch.isfinite(out)
    assert out.item() == pytest.approx(bce.item(), abs=1e-5)


def test_warmup_disables_tail_terms():
    lf = TailMarginBCE(min_bank=64, warmup_steps=1000)
    _fill_bank(lf, n=128)
    logits = torch.tensor([[-3.0]])
    targets = torch.tensor([[1.0]])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.flatten(), targets.flatten())
    out = lf(logits, targets)
    assert out.item() == pytest.approx(bce.item(), abs=1e-5)  # tail inactive during warmup


def test_attack_below_tau_gets_pushed_up():
    lf = TailMarginBCE(min_bank=256, lambda_tail=1.0, mu_bona=0.0, warmup_steps=0)
    _fill_bank(lf, n=512, mean=0.3)   # tau ~ 0.4 in prob space
    logits = torch.tensor([[-3.0]], requires_grad=True)  # sigmoid ~0.047 << tau
    targets = torch.tensor([[1.0]])
    loss = lf(logits, targets)
    loss.backward()
    # decreasing loss requires INCREASING the logit
    assert logits.grad[0, 0] < 0


def test_bona_above_tau_gets_pulled_down():
    lf = TailMarginBCE(min_bank=256, lambda_tail=0.0, mu_bona=1.0, warmup_steps=0)
    _fill_bank(lf, n=512, mean=0.1)
    logits = torch.tensor([[5.0]], requires_grad=True)  # sigmoid ~0.993 >> tau
    targets = torch.tensor([[0.0]])
    loss = lf(logits, targets)
    loss.backward()
    assert logits.grad[0, 0] > 0


def test_prob_space_is_bounded_no_divergence():
    """Iteration-1 failure regression test: extreme logits must NOT explode the loss."""
    lf = TailMarginBCE(min_bank=64, warmup_steps=0, lambda_tail=1.0, mu_bona=0.5)
    _fill_bank(lf, n=128)
    logits = torch.tensor([[-50.0], [50.0], [-30.0], [30.0]])
    targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    out = lf(logits, targets)
    assert torch.isfinite(out)
    # BCE on confidently-wrong extremes is large, but tail terms add at most
    # lambda*(1+margin) + mu*1 — bounded, unlike the logit-space arms race
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.flatten(), targets.flatten())
    assert out.item() <= bce.item() + 1.0 * (1 + 0.1) + 0.5 * 1.0 + 1e-5


def test_single_class_batches_are_finite():
    lf = TailMarginBCE(min_bank=64, warmup_steps=0)
    _fill_bank(lf, n=128)
    all_attack = lf(torch.randn(6, 1), torch.ones(6, 1))
    all_bona = lf(torch.randn(6, 1), torch.zeros(6, 1))
    assert torch.isfinite(all_attack) and torch.isfinite(all_bona)


def test_bank_is_sliding_window():
    lf = TailMarginBCE(bank_size=100, min_bank=10)
    lf._update_bank(torch.zeros(150))
    assert int(lf.bank_filled) == 100   # capped at bank_size


def test_logit_space_still_available_but_explicit():
    lf = TailMarginBCE(space="logit", warmup_steps=0, min_bank=64)
    assert lf.space == "logit"
    with pytest.raises(ValueError):
        TailMarginBCE(space="nope")


def test_build_loss_dispatch():
    assert isinstance(build_loss("bce"), torch.nn.BCEWithLogitsLoss)
    assert isinstance(build_loss("tail_margin"), TailMarginBCE)
    with pytest.raises(ValueError):
        build_loss("nope")
