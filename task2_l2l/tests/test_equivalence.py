"""L2L equivalence on a tiny random Qwen3 (CPU, seconds). The real-model run is `equivalence.py`.

What each test rules out:
  reference_matches_hf     my layer driver (mask, positions, rotary, norm, tied head) differs from HF
  logits_bitexact          streaming changes the forward at all (same ops, same shapes -> torch.equal)
  grads_match              layer-by-layer recompute + chunked head computes different gradients
  l2l_one_copy_per_pass    micro-batches re-read the streamed weights (L2L order broken)
  negative controls        the comparison can actually fail: swapped layers, a dropped adapter
"""
import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare import compare, lora_grads  # noqa: E402
from engine import IGNORE, StreamedQwen3  # noqa: E402
from lora import LoRAParams, lora_shapes  # noqa: E402
from store import LayerStore  # noqa: E402

L, B, S, V, COMPLETION = 4, 4, 12, 256, 6


def tiny_config():
    cfg = Qwen3Config(vocab_size=V, hidden_size=64, intermediate_size=160, num_hidden_layers=L,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=64,
                      tie_word_embeddings=True, use_sliding_window=False, sliding_window=None,
                      attention_dropout=0.0)
    cfg._attn_implementation = "sdpa"
    return cfg


@pytest.fixture(scope="module")
def world():
    cfg = tiny_config()
    torch.manual_seed(0)
    hf = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():  # non-trivial norm weights, so a skipped or misplaced norm is visible
        for n, p in hf.named_parameters():
            if "norm" in n:
                p.add_(0.1 * torch.randn_like(p))
    lora = LoRAParams(lora_shapes(cfg), L, r=4, alpha=8)
    lora.randomize_B(std=0.05)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, V, (B, S), generator=g)
    pad = torch.ones(B, S, dtype=torch.long)
    pad[0, :3] = 0
    pad[2, :5] = 0
    targets = torch.full((B, S), IGNORE)
    targets[:, S - COMPLETION - 1:-1] = ids[:, S - COMPLETION:]  # position t predicts token t+1
    weights = torch.randn(B * S, generator=g, dtype=torch.float64)
    return SimpleWorld(cfg, hf, lora, ids, pad, targets, weights)


class SimpleWorld:
    def __init__(self, cfg, hf, lora, ids, pad, targets, weights):
        self.cfg, self.hf, self.lora, self.ids, self.pad, self.targets, self.weights = cfg, hf, lora, ids, pad, targets, weights

    def store(self, dtype):
        return LayerStore.from_state_dict(self.hf.state_dict(), L, dtype)

    def engine(self, dtype=torch.float32, resident=None, n_buffers=2, lora=None, store=None):
        # LoRA stays fp32 for fp32/bf16 (as in training); the fp64 truth gets an fp64 copy of the same values
        lora = lora if lora is not None else (self.lora.clone_as(dtype) if dtype == torch.float64 else self.lora)
        resident = range(L) if resident is None else resident
        return StreamedQwen3(self.cfg, store or self.store(dtype), lora, "cpu", resident=resident, n_buffers=n_buffers)

    def loss_fn(self, logp, idx):
        return -(self.weights[idx].to(logp.dtype) * logp).sum()


def run_grads(w, eng, mask, mb=None, chunk=3, reference=False):
    eng.lora.zero_grad(set_to_none=True)
    fn = eng.autograd_grads if reference else eng.train_grads
    kw = {} if reference else {"chunk": chunk}
    loss = fn(w.ids, mask, w.targets, w.loss_fn, mb=mb, **kw)
    return loss.item(), lora_grads(eng.lora)


def valid(mask):
    return mask.bool() if mask is not None else torch.ones(B, S, dtype=torch.bool)


@pytest.mark.parametrize("padded", [False, True])
def test_reference_matches_hf(world, padded):
    """With B = 0 the adapter adds exactly 0, so the resident engine must reproduce HF logits."""
    w, mask = world, (world.pad if padded else None)
    lora0 = LoRAParams(lora_shapes(w.cfg), L, r=4, alpha=8)  # B = 0
    ours = w.engine(lora=lora0).logits(w.ids, mask)
    kw = {} if mask is None else dict(attention_mask=mask, position_ids=(mask.cumsum(-1) - 1).clamp(min=0))
    with torch.no_grad():
        theirs = w.hf(w.ids, **kw).logits
    v = valid(mask)
    assert torch.allclose(ours[v], theirs[v], rtol=0, atol=1e-5), (ours[v] - theirs[v]).abs().max()


@pytest.mark.parametrize("resident,n_buffers", [((), 1), ((), 2), ((0, 2), 2), ((1,), 3)])
@pytest.mark.parametrize("padded", [False, True])
def test_logits_bitexact(world, resident, n_buffers, padded):
    w, mask = world, (world.pad if padded else None)
    ref = w.engine().logits(w.ids, mask)
    eng = w.engine(resident=resident, n_buffers=n_buffers)
    out = eng.logits(w.ids, mask)
    assert torch.equal(out[valid(mask)], ref[valid(mask)])
    assert eng.loads == L - len(resident)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mb", [None, 2, 1])
@pytest.mark.parametrize("padded", [False, True])
def test_grads_match(world, dtype, mb, padded):
    """Streamed (0 resident, 2 buffers, chunk 3) vs autograd reference with the same micro-batching,
    both judged against an fp64 full-batch truth."""
    w, mask = world, (world.pad if padded else None)
    truth_loss, truth = run_grads(w, w.engine(torch.float64), mask, reference=True)
    ref_loss, ref = run_grads(w, w.engine(dtype), mask, mb=mb, reference=True)
    s_loss, stream = run_grads(w, w.engine(dtype, resident=()), mask, mb=mb)
    rep = compare(stream, ref, truth)
    assert rep["all_nonzero"], "some LoRA tensor has a zero gradient: the test would prove nothing"
    assert rep["pass_"], {k: rep[k] for k in ("agg_err_stream", "agg_err_ref", "worst_ratio")}
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    assert abs(s_loss - truth_loss) <= tol * abs(truth_loss) and abs(ref_loss - truth_loss) <= tol * abs(truth_loss)


@pytest.mark.parametrize("resident,n_buffers", [((0,), 2), ((3,), 1), ((0, 2), 2), ((1, 2), 1), ((0, 1, 2), 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mb", [None, 1])
@pytest.mark.parametrize("padded", [False, True])
def test_grads_match_mixed_residency(world, resident, n_buffers, dtype, mb, padded):
    """The training configuration: some layers resident, the rest streamed through the slots (the runs used 24
    resident + 12 streamed). Same rule as test_grads_match, which covers only the fully streamed engine."""
    w, mask = world, (world.pad if padded else None)
    _, truth = run_grads(w, w.engine(torch.float64), mask, reference=True)
    _, ref = run_grads(w, w.engine(dtype), mask, mb=mb, reference=True)
    eng = w.engine(dtype, resident=resident, n_buffers=n_buffers)
    _, stream = run_grads(w, eng, mask, mb=mb)
    rep = compare(stream, ref, truth)
    assert eng.loads > 0, "nothing was streamed: the test would prove nothing"
    assert rep["all_nonzero"], "some LoRA tensor has a zero gradient: the test would prove nothing"
    assert rep["pass_"], {k: rep[k] for k in ("agg_err_stream", "agg_err_ref", "worst_ratio")}


def test_l2l_one_copy_per_pass(world):
    """L2L order: 4 micro-batches still cost one copy per streamed layer per pass (forward + backward = 2)."""
    w = world
    eng = w.engine(resident=(3,))
    run_grads(w, eng, None, mb=1)
    assert eng.passes == 2 and eng.loads == 2 * (L - 1)


def test_control_swapped_layers_fails(world):
    """Negative control: streaming layers 1 and 2 in the wrong order must fail both checks."""
    w = world
    store = w.store(torch.float32)
    store.flat[1], store.flat[2] = store.flat[2], store.flat[1]
    bad = w.engine(resident=(), store=store)
    assert not torch.equal(bad.logits(w.ids), w.engine().logits(w.ids))
    _, truth = run_grads(w, w.engine(torch.float64), None, reference=True)
    _, ref = run_grads(w, w.engine(), None, reference=True)
    _, stream = run_grads(w, bad, None)
    assert not compare(stream, ref, truth)["pass_"]


def test_control_dropped_adapter_fails(world):
    """Negative control: a layer whose adapter is never bound (B·A missing in forward) must fail."""
    w = world
    eng = w.engine(resident=())
    orig = eng.visit

    def visit_without_adapter(order):
        for i, layer in orig(order):
            if i == 2:
                for m in layer.modules():
                    if hasattr(m, "bound_to"):
                        m.bind(None, None, None)
            yield i, layer
    eng.visit = visit_without_adapter
    _, truth = run_grads(w, w.engine(torch.float64), None, reference=True)
    _, ref = run_grads(w, w.engine(), None, reference=True)
    _, stream = run_grads(w, eng, None)
    rep = compare(stream, ref, truth)
    assert not rep["pass_"]


def test_offloaded_checkpoints_match_and_reuse_arena(world):
    """Boundary checkpoints through the host arena (the CUDA path, minus pinning) give bit-identical grads,
    also when the sequence length changes between steps on the same engine (regions re-cut, not reallocated)."""
    w = world
    engines = {off: StreamedQwen3(w.cfg, w.store(torch.float32), w.lora, "cpu", resident=(), offload=off)
               for off in (False, True)}
    arenas = []
    for S_cut in (S, S - 3, S):  # long, shorter (reuses the arena), long again
        ids, tg = w.ids[:, :S_cut], w.targets[:, :S_cut].clone()
        tg[:, -1] = IGNORE
        out = {}
        for off, eng in engines.items():
            w.lora.zero_grad(set_to_none=True)
            eng.train_grads(ids, None, tg, w.loss_fn, mb=2, chunk=3)
            out[off] = lora_grads(w.lora)
        assert all(torch.equal(out[False][k], out[True][k]) for k in out[False])
        arenas.append(engines[True]._arena.data_ptr())
    assert len(set(arenas)) == 1, "arena reallocated although no step was longer than the first"
