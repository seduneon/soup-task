"""Reward parser (adversarial cases), advantages, loss gradient, and a sign control on a real update."""
import pytest
import torch

from common import engine_from, tiny_config, tiny_hf
from decode import generate
from engine import IGNORE
from grpo import completion_targets, group_advantages, make_loss, rollout_logp_grid, token_coef
from lora import LoRAParams, lora_shapes
from reward import math_boxed


@pytest.mark.parametrize("text,answer,expected", [
    ("so the answer is \\boxed{42}", "42", 1.0),
    ("\\boxed{ 42. }", "42", 1.0),
    ("\\boxed{1,000}", "1000", 1.0),
    ("\\boxed{\\frac{1}{2}}", "0.5", 1.0),
    ("\\boxed{\\dfrac{3}{4}}", "3/4", 1.0),
    ("\\boxed{$-7$}", "-7", 1.0),
    ("\\boxed{\\text{(B)}}", "(B)", 1.0),
    ("\\boxed{x^{2}+1}", "x^{2}+1", 1.0),
    ("42", "42", 0.0),                                   # no box: no fallback to the last number
    ("\\boxed{41} or maybe \\boxed{42}", "42", 0.0),     # hedging between answers
    ("\\boxed{42} ... final: \\boxed{42}", "42", 1.0),   # repeating the same answer is fine
    ("\\boxed{42", "42", 0.0),                            # truncated box
    ("\\boxed{420}", "42", 0.0),
    ("\\boxed{4.2}", "42", 0.0),
    ("\\boxed{}", "42", 0.0),
    # formats GSM8K answers come in (from the review): must score 1
    ("\\boxed{\\$18}", "18", 1.0),
    ("\\boxed{\\$1,000}", "1000", 1.0),
    ("\\boxed{25\\%}", "25", 1.0),
    ("\\boxed{18 \\text{ dollars}}", "18", 1.0),
    ("\\boxed{10 \\text{ hours}}", "10", 1.0),
    ("\\boxed{1{,}000}", "1000", 1.0),
    ("\\boxed{18\\ \\text{cm}}", "18", 1.0),
    ("\\boxed{x = 5}", "5", 1.0),
    ("\\boxed{90^\\circ}", "90", 1.0),
    ("\\boxed{18 dollars}", "19", 0.0),               # unit stripping must not make wrong numbers right
    ("\\boxed{18 or 19}", "18", 0.0),                 # "or" is not a unit
    # MATH-style gold answers (data/math35)
    ("\\boxed{0.5}", "\\frac12", 1.0),
    ("\\boxed{\\frac{10}{3}}", "\\frac{10}3", 1.0),
    ("\\boxed{-\\frac{1}{14}}", "-\\frac{1}{14}", 1.0),
    ("\\boxed{-1/14}", "-\\frac{1}{14}", 1.0),
    ("\\boxed{\\frac{1}{14}}", "-\\frac{1}{14}", 0.0),  # sign control
    ("\\boxed{-0.0714}", "-\\frac{1}{14}", 0.0),        # exact comparison: no rounding tolerance
])
def test_math_boxed(text, answer, expected):
    assert math_boxed(text, answer)[0] == expected


def test_group_advantages():
    r = torch.tensor([1., 0., 0., 1., 1., 1., 1., 1.])
    adv, st = group_advantages(r, G=4)
    assert torch.allclose(adv[:4], torch.tensor([1., -1., -1., 1.]), atol=1e-5)
    assert torch.equal(adv[4:], torch.zeros(4))  # zero-variance group carries no signal
    assert st["zero_std_groups"] == 0.5


def test_completion_targets_mask_prompt_and_pads():
    ids = torch.tensor([[0, 5, 6, 7, 8, 0], [9, 5, 6, 7, 8, 3]])
    comp = torch.tensor([[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1]]).bool()
    t = completion_targets(ids, comp)
    assert t.tolist() == [[IGNORE, IGNORE, 7, 8, IGNORE, IGNORE], [IGNORE, IGNORE, 7, 8, 3, IGNORE]]


@pytest.mark.parametrize("beta", [0.0, 0.1])
def test_loss_gradient_formula(beta):
    """d loss / d logp = -c·A + beta·c·(1 - exp(ref - logp)), token by token."""
    g = torch.Generator().manual_seed(0)
    targets = torch.full((3, 5), IGNORE)
    targets[:, 2:] = 1
    coef, adv = token_coef(targets), torch.randn(3, generator=g)
    ref = -torch.rand(3, 5, generator=g)
    logp = (-torch.rand(15, generator=g)).requires_grad_()
    idx = (targets.view(-1) != IGNORE).nonzero().squeeze(1)
    make_loss(coef, adv, ref, beta)(logp[idx], idx).backward()
    c, a, d = coef.view(-1)[idx], adv[:, None].expand(3, 5).reshape(-1)[idx], ref.view(-1)[idx] - logp.detach()[idx]
    assert torch.allclose(logp.grad[idx], -c * a + beta * c * (1 - torch.exp(d)), atol=1e-7)
    assert abs(coef.sum().item() - 1.0) < 1e-6  # seq normalisation: weights sum to 1 over the batch


def test_update_sign_control():
    """One step: the completion with positive advantage gains log-prob, the negative one loses it.
    A flipped sign, a dropped advantage or a gradient that never reaches LoRA fails this."""
    cfg = tiny_config(L=3, V=256)
    hf = tiny_hf(cfg)
    lora = LoRAParams(lora_shapes(cfg), 3, r=4, alpha=8)
    lora.randomize_B(std=0.05)
    eng = engine_from(hf, cfg, lora=lora, resident=())
    ro = generate(eng, [[3, 4, 5, 6]] * 2, 8, {255}, 0, generator=torch.Generator().manual_seed(0))
    targets = completion_targets(ro.ids, ro.comp)
    before = eng.logprobs(ro.ids, ro.attn, targets)
    adv = torch.tensor([1.0, -1.0])
    lora.zero_grad(set_to_none=True)
    eng.train_grads(ro.ids, ro.attn, targets, make_loss(token_coef(targets), adv))
    opt = torch.optim.SGD(lora.parameters(), lr=1e-2)
    opt.step()
    after = eng.logprobs(ro.ids, ro.attn, targets)
    m = targets != IGNORE
    delta = [(after[b] - before[b])[m[b]].sum().item() for b in range(2)]
    assert delta[0] > 0 > delta[1], delta
    # the training-time log-probs are the decode-time ones (same policy, same inputs)
    grid = rollout_logp_grid(ro, ro.ids.shape[1])
    assert torch.allclose(before[m], grid[m], atol=1e-4)
