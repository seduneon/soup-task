"""Streamed decode with a per-layer KV cache.

  greedy_matches_hf            our prefill + KV decode == HF generate (greedy), with and without left padding
  invariant_to_streaming       resident count, buffers and prefill micro-batches do not change a single token
  rollout_logp_matches_rescore the log-probs recorded while decoding == a full no-grad pass over the rollout,
                               i.e. what training will score is what was sampled (on-policy at ratio 1)
  sampling_is_seeded           the same generator state gives the same samples (needed for resume)
  dynamic_arm_token_identical  paged KV + residency changing every pass == static KV, regular residency
"""
import pytest
import torch

from common import engine_from, tiny_config, tiny_hf
from decode import generate
from engine import IGNORE

L, V, EOS, PAD, MAX_NEW = 4, 256, 255, 0, 12


@pytest.fixture(scope="module")
def model():
    cfg = tiny_config(L=L, V=V)
    return cfg, tiny_hf(cfg)


def prompts(lengths, seed=3):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, V - 1, (n,), generator=g).tolist() for n in lengths]


def hf_greedy(hf, ps):
    P = max(map(len, ps))
    ids = torch.tensor([[PAD] * (P - len(p)) + p for p in ps])
    attn = torch.tensor([[0] * (P - len(p)) + [1] * len(p) for p in ps])
    out = hf.generate(ids, attention_mask=attn, do_sample=False, max_new_tokens=MAX_NEW, eos_token_id=EOS,
                      pad_token_id=PAD)
    return out[:, P:]


def completions(r):
    return [r.ids[b, r.P:][r.comp[b, r.P:]].tolist() for b in range(r.ids.shape[0])]


@pytest.mark.parametrize("lengths", [[7, 7, 7], [5, 9, 3, 8]])
def test_greedy_matches_hf(model, lengths):
    cfg, hf = model
    ps = prompts(lengths)
    r = generate(engine_from(hf, cfg, resident=()), ps, MAX_NEW, {EOS}, PAD, temperature=0)
    theirs = hf_greedy(hf, ps)
    for b, ours in enumerate(completions(r)):
        ref = theirs[b].tolist()
        ref = ref[:ref.index(EOS) + 1] if EOS in ref else ref
        assert ours == ref, (b, ours, ref)


@pytest.mark.parametrize("resident,n_buffers,prefill_mb", [((), 1, None), ((), 2, 1), ((1, 3), 2, 2), (range(L), 2, 3)])
def test_invariant_to_streaming(model, resident, n_buffers, prefill_mb):
    cfg, hf = model
    ps = prompts([5, 9, 3, 8])
    base = generate(engine_from(hf, cfg), ps, MAX_NEW, {EOS}, PAD, temperature=0)
    eng = engine_from(hf, cfg, resident=resident, n_buffers=n_buffers)
    r = generate(eng, ps, MAX_NEW, {EOS}, PAD, temperature=0, prefill_mb=prefill_mb)
    assert torch.equal(r.ids, base.ids)
    # same shapes -> bit-exact; prefill micro-batches change GEMM shapes -> fp32 rounding only (~5e-7)
    if prefill_mb is None:
        assert torch.equal(r.logp, base.logp)
    else:
        assert torch.allclose(r.logp, base.logp, atol=1e-5, rtol=0)
    assert eng.loads == (r.decode_passes + 1) * (L - len(list(resident)))  # one copy per streamed layer per pass


def test_rollout_logp_matches_rescore(model):
    cfg, hf = model
    ps = prompts([5, 9, 3, 8])
    eng = engine_from(hf, cfg, resident=())
    g = torch.Generator().manual_seed(0)
    r = generate(eng, ps, MAX_NEW, {EOS}, PAD, temperature=1.0, generator=g)
    targets = torch.full(r.ids.shape, IGNORE)
    targets[:, :-1] = torch.where(r.comp[:, 1:], r.ids[:, 1:], IGNORE)
    lp = eng.logprobs(r.ids, r.attn, targets, chunk=5)
    rescored = lp[:, r.P - 1:-1]  # position c-1 predicts column c
    mask = r.comp[:, r.P:]
    assert mask.sum() > 0
    assert torch.allclose(rescored[mask], r.logp[mask], atol=1e-4, rtol=0), (rescored[mask] - r.logp[mask]).abs().max()


def test_sampling_is_seeded(model):
    cfg, hf = model
    ps = prompts([5, 9, 3, 8])
    eng = engine_from(hf, cfg, resident=())
    a = generate(eng, ps, MAX_NEW, {EOS}, PAD, generator=torch.Generator().manual_seed(7))
    g = torch.Generator().manual_seed(7)
    b = generate(eng, ps, MAX_NEW, {EOS}, PAD, generator=g)
    c = generate(eng, ps, MAX_NEW, {EOS}, PAD, generator=g)  # state advanced: different samples
    assert torch.equal(a.ids, b.ids) and not torch.equal(a.ids, c.ids)


@pytest.mark.parametrize("temperature", [0, 1.0])
def test_dynamic_arm_token_identical(model, temperature):
    cfg, hf = model
    ps = prompts([5, 9, 3, 8])
    regular = generate(engine_from(hf, cfg, resident=(0,)), ps, MAX_NEW, {EOS}, PAD, temperature=temperature,
                       generator=torch.Generator().manual_seed(1))
    eng = engine_from(hf, cfg, resident=(0,))
    schedule = lambda cols, cache: {0} | set(range(1, 1 + (cols * 7) % L))  # changes on every pass
    dyn = generate(eng, ps, MAX_NEW, {EOS}, PAD, temperature=temperature, kv="paged", page=4, plan=schedule,
                   generator=torch.Generator().manual_seed(1))
    assert torch.equal(dyn.ids, regular.ids) and torch.equal(dyn.logp, regular.logp)
    assert len(set(dyn.resident_trace)) > 1, "the schedule never changed residency: test proves nothing"
    assert eng.promotions > 1
