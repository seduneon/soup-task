"""Shared tiny-model builders for the tests (CPU, random weights, no download)."""
import shutil
import sys
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import StreamedQwen3  # noqa: E402
from lora import LoRAParams, lora_shapes  # noqa: E402
from store import LayerStore  # noqa: E402

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def tiny_config(L=4, V=256):
    cfg = Qwen3Config(vocab_size=V, hidden_size=64, intermediate_size=160, num_hidden_layers=L,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                      tie_word_embeddings=True, use_sliding_window=False, sliding_window=None,
                      attention_dropout=0.0)
    cfg._attn_implementation = "sdpa"
    return cfg


def tiny_hf(cfg, seed=0):
    torch.manual_seed(seed)
    hf = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():  # non-trivial norm weights, so a skipped or misplaced norm is visible
        for n, p in hf.named_parameters():
            if "norm" in n:
                p.add_(0.1 * torch.randn_like(p))
    return hf


def engine_from(hf, cfg, lora=None, resident=None, dtype=torch.float32, **kw):
    L = cfg.num_hidden_layers
    store = LayerStore.from_state_dict(hf.state_dict(), L, dtype)
    lora = lora if lora is not None else LoRAParams(lora_shapes(cfg), L, r=4, alpha=8)
    return StreamedQwen3(cfg, store, lora, "cpu", resident=range(L) if resident is None else resident, **kw)


def local_tokenizer_dir():
    """Qwen3 tokenizer from the local HF cache (any Qwen3 size has the same tokenizer); None if absent."""
    hub = Path.home() / ".cache/huggingface/hub"
    for snap in sorted(hub.glob("models--Qwen--Qwen3-*/snapshots/*")):
        if all((snap / f).exists() for f in TOKENIZER_FILES):
            return snap
    return None


def save_tiny_model_dir(out, L=4, seed=0, shard="2MB"):
    """A real on-disk model directory (sharded safetensors + Qwen3 tokenizer), so train/merge run the same
    code path as on Qwen3-4B."""
    tok_dir = local_tokenizer_dir()
    assert tok_dir is not None, "no Qwen3 tokenizer in the local HF cache"
    cfg = tiny_config(L=L, V=151936)
    hf = tiny_hf(cfg, seed)
    hf.save_pretrained(out, max_shard_size=shard, safe_serialization=True)
    for f in TOKENIZER_FILES:
        shutil.copy(tok_dir / f, Path(out) / f)
    return cfg
