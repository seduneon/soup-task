"""Merge: merged model (adapter off) == base + adapter, HF loads it, and the bf16 rounding trap is visible."""
import pytest
import torch
from transformers import Qwen3ForCausalLM

from common import tiny_config, tiny_hf
from engine import StreamedQwen3
from lora import LoRAParams, lora_shapes
from merge import merge
from store import LayerStore

L = 3


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp("merge")
    cfg = tiny_config(L=L, V=256)
    tiny_hf(cfg).save_pretrained(root / "base", max_shard_size="200KB", safe_serialization=True)
    return root, cfg


def adapter(cfg, std, seed=0):
    torch.manual_seed(seed)
    lora = LoRAParams(lora_shapes(cfg), L, r=4, alpha=8)
    lora.randomize_B(std=std, seed=seed)
    return lora, {"lora": lora.state_dict(), "r": lora.r, "alpha": lora.alpha, "targets": lora.targets}


def engine(path, cfg, lora, dtype):
    return StreamedQwen3(cfg, LayerStore.from_safetensors(path, L, dtype), lora, "cpu", resident=())


def test_merged_equals_adapter_fp32(base):
    root, cfg = base
    lora, ad = adapter(cfg, std=0.05)
    rep = merge(root / "base", ad, root / "m32")
    assert rep["merged_tensors"] == L * 7
    ids = torch.randint(0, 256, (2, 10), generator=torch.Generator().manual_seed(0))
    with_adapter = engine(root / "base", cfg, lora, torch.float32).logits(ids)
    m = engine(root / "m32", cfg, lora, torch.float32)
    with m.adapter_off():
        merged = m.logits(ids)
    b = engine(root / "base", cfg, lora, torch.float32)
    with b.adapter_off():
        base_logits = b.logits(ids)
    assert torch.allclose(merged, with_adapter, atol=1e-5, rtol=0)
    assert (merged - base_logits).abs().max() > 1e-2, "merged == base: the adapter was not applied"
    hf = Qwen3ForCausalLM.from_pretrained(root / "m32", attn_implementation="sdpa").eval()
    with torch.no_grad():
        assert torch.allclose(hf(ids).logits, merged, atol=1e-5, rtol=0)


def test_bf16_rounding_swallows_small_deltas(base):
    """A tiny adapter merged into bf16 weights mostly disappears: the report must show it."""
    root, cfg = base
    hf = Qwen3ForCausalLM.from_pretrained(root / "base").to(torch.bfloat16)
    hf.save_pretrained(root / "base16", max_shard_size="200KB", safe_serialization=True)
    _, small = adapter(cfg, std=1e-5)
    _, large = adapter(cfg, std=0.05)
    rs, rl = merge(root / "base16", small, root / "m16s"), merge(root / "base16", large, root / "m16l")
    assert rs["changed_frac_mean"] < 0.2 and rs["delta_kept_min"] < 0.5
    assert rl["changed_frac_mean"] > 0.5


def test_in_memory_merge_matches_merge_py(base):
    """store.merge_lora (evaluate.py --merge-in-memory) == merge.py's output, for an fp32 and a bf16 base."""
    root, cfg = base
    lora, ad = adapter(cfg, std=0.05)
    hf = Qwen3ForCausalLM.from_pretrained(root / "base").to(torch.bfloat16)
    hf.save_pretrained(root / "base16m", max_shard_size="200KB", safe_serialization=True)
    for src, dt in (("base", torch.float32), ("base16m", torch.bfloat16)):
        merge(root / src, ad, root / f"{src}_merged")
        s = LayerStore.from_safetensors(root / src, L, dt)
        s.merge_lora(ad)
        ref = LayerStore.from_safetensors(root / f"{src}_merged", L, dt)
        for i in range(L):
            assert torch.equal(s.flat[i], ref.flat[i]), (dt, i)
