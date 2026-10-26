"""Preference losses: DPO family reproduces the 3B engine's forms; KTO matches the paper's value function."""
from __future__ import annotations

import math

import pytest
import torch

from kotodama.posttrain import losses


def test_dpo_at_reference_is_log2() -> None:
    z = torch.zeros(3)
    loss, margin = losses.pairwise_loss(z, z, z, z, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2)) and torch.all(margin == 0)


def test_dpo_variants_match_closed_forms() -> None:
    lp_c, lp_r = torch.tensor([-10.0]), torch.tensor([-12.0])
    ref_c, ref_r = torch.tensor([-11.0]), torch.tensor([-11.0])
    h = 2.0
    beta = 0.5
    dpo, m = losses.pairwise_loss(lp_c, lp_r, ref_c, ref_r, beta)
    assert dpo.item() == pytest.approx(-math.log(1 / (1 + math.exp(-beta * h))))
    assert m.item() == pytest.approx(beta * h)
    ipo, _ = losses.pairwise_loss(lp_c, lp_r, ref_c, ref_r, beta, loss_type="ipo")
    assert ipo.item() == pytest.approx((h - 1 / (2 * beta)) ** 2)
    # chosen above its reference: DPOP penalty inactive => equals DPO
    dpop, _ = losses.pairwise_loss(lp_c, lp_r, ref_c, ref_r, beta, loss_type="dpop")
    assert dpop.item() == pytest.approx(dpo.item())
    with pytest.raises(ValueError):
        losses.pairwise_loss(lp_c, lp_r, ref_c, ref_r, beta, loss_type="orpo")


def test_kto_value_function() -> None:
    lp = torch.tensor([-5.0, -5.0])
    ref = torch.tensor([-6.0, -6.0])  # r = +1 for both
    des = torch.tensor([True, False])
    loss, reward = losses.kto_loss(lp, ref, des, kl=0.0, beta=0.5, lambda_d=1.0, lambda_u=2.0)
    s = 1 / (1 + math.exp(-0.5))
    assert loss.item() == pytest.approx(((1 - s) + 2 * (1 - (1 - s))) / 2)
    assert reward.tolist() == pytest.approx([0.5, 0.5])


def test_kto_gradients_push_the_right_way() -> None:
    lp = torch.zeros(2, requires_grad=True)
    loss, _ = losses.kto_loss(lp, torch.zeros(2), torch.tensor([True, False]), kl=0.0, beta=0.1)
    loss.backward()
    assert lp.grad[0] < 0 < lp.grad[1]  # raise desirable logp, lower undesirable


def test_kto_kl_is_clamped_and_detached() -> None:
    lp = torch.tensor([-3.0, -4.0], requires_grad=True)
    kl = losses.kto_kl_estimate(lp, torch.tensor([-1.0, -1.0]))
    assert kl.item() == 0.0 and not kl.requires_grad
    assert losses.kto_balance(30, 20, 1.0, 1.0) == pytest.approx(1.5)
