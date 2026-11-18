"""Host-side weight store: every decoder layer as one flat buffer; the streamed ones page-locked.

One flat buffer per layer means one H2D copy per layer visit, and a device slot whose parameters are views
into a flat device buffer of the same layout. Weights are filled tensor by tensor with `copy_` into memory
allocated once, which avoids the load -> .to() -> pin_memory() triple copy.

Pinning: torch's `pin_memory=True` goes through the caching host allocator, which rounds every block up to
a power of two (a 192.5 MiB layer takes 256 MiB) and never returns it to the OS. So buffers are allocated
pageable and page-locked at their exact size with cudaHostRegister, and only for layers that are streamed
(`pin`). Layers that stay resident for the whole run can drop their host copy (`release`).
"""
import json
import math
import re
from pathlib import Path

import torch

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def register_pinned(t):
    """Page-lock exactly t's bytes. The tensor must stay alive (and unmoved) until unregister_pinned."""
    torch.cuda.check_error(torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0))
    return t


def unregister_pinned(t):
    torch.cuda.check_error(torch.cuda.cudart().cudaHostUnregister(t.data_ptr()))


class LayerStore:
    def __init__(self, layout, n_layers, dtype):
        self.layout = layout  # [(name, shape, offset, numel)], identical for every layer
        self.numel = sum(n for *_, n in layout)
        self.n_layers, self.dtype = n_layers, dtype
        self.flat = [torch.empty(self.numel, dtype=dtype) for _ in range(n_layers)]
        self.filled = [set() for _ in range(n_layers)]
        self.pinned = set()
        self.other = {}  # embed_tokens, final norm (moved to the device by the engine)

    @property
    def layer_bytes(self):
        return self.numel * torch.empty(0, dtype=self.dtype).element_size()

    def pin(self, layers):
        """Page-lock these layers' buffers at their exact size (CUDA only)."""
        for i in layers:
            if i not in self.pinned:
                assert self.flat[i] is not None, f"layer {i} was released"
                register_pinned(self.flat[i])
                self.pinned.add(i)
        assert all(self.flat[i].is_pinned() for i in layers), "store not pinned: H2D copies would be synchronous"

    def release(self, layers):
        """Drop host copies of layers that will never be streamed again (resident for the whole run)."""
        for i in layers:
            if i in self.pinned:
                unregister_pinned(self.flat[i])
                self.pinned.discard(i)
            self.flat[i] = None

    @staticmethod
    def make_layout(shapes):
        layout, off = [], 0
        for name in sorted(shapes):
            n = math.prod(shapes[name])
            layout.append((name, tuple(shapes[name]), off, n))
            off += n
        return layout

    def views(self, flat):
        return {name: flat[off:off + n].view(shape) for name, shape, off, n in self.layout}

    def put(self, i, name, tensor):
        dst = self.views(self.flat[i])[name]
        assert dst.shape == tensor.shape, (i, name, dst.shape, tensor.shape)
        dst.copy_(tensor)
        self.filled[i].add(name)

    def merge_lora(self, ad):
        """Fold an adapter into the host weights in place: W += (alpha/r)·B·A, computed in fp32 and written in the
        store dtype. With an fp32 store nothing is rounded, so this isolates merge.py's bf16 cast (Part 3, R3).
        Returns per-tensor stats. Call before the engine copies anything to the device."""
        from lora import MODULE_OF
        lora, scale, stats = ad["lora"], ad["alpha"] / ad["r"], []
        for i in range(self.n_layers):
            views = self.views(self.flat[i])
            for t in ad["targets"]:
                W = views[f"{MODULE_OF[t]}.{t}.weight"]
                delta = scale * (lora[f"p.{i}_{t}_B"].float() @ lora[f"p.{i}_{t}_A"].float())
                new = (W.float() + delta).to(W.dtype)
                stats.append(float((new.float() - W.float()).norm() / delta.norm().clamp(min=1e-30)))
                W.copy_(new)
        return {"merged_tensors": len(stats), "delta_kept_min": min(stats), "delta_kept_mean": sum(stats) / len(stats)}

    def check_complete(self):
        names = {name for name, *_ in self.layout}
        missing = [(i, sorted(names - f)) for i, f in enumerate(self.filled) if f != names]
        assert not missing, f"layers with unfilled tensors: {missing[:3]}"

    @classmethod
    def from_state_dict(cls, sd, n_layers, dtype):
        shapes = {m[2]: v.shape for k, v in sd.items() if (m := LAYER_RE.match(k)) and m[1] == "0"}
        store = cls(cls.make_layout(shapes), n_layers, dtype)
        for k, v in sd.items():
            m = LAYER_RE.match(k)
            if m and int(m[1]) < n_layers:
                store.put(int(m[1]), m[2], v)
            elif not m:
                store.other[k] = v.detach().to(dtype).clone()
        store.check_complete()
        return store

    @classmethod
    def from_safetensors(cls, repo_dir, n_layers=None, dtype=torch.bfloat16):
        """Read one tensor at a time from the shards (mmap) straight into the flat buffers."""
        from safetensors import safe_open
        repo_dir = Path(repo_dir)
        index = repo_dir / "model.safetensors.index.json"
        files = sorted(set(json.load(open(index))["weight_map"].values())) if index.exists() else ["model.safetensors"]
        shapes, total_layers = {}, 0
        for fn in files:
            with safe_open(repo_dir / fn, "pt") as f:
                for k in f.keys():
                    if m := LAYER_RE.match(k):
                        total_layers = max(total_layers, int(m[1]) + 1)
                        if m[1] == "0":
                            shapes[m[2]] = f.get_slice(k).get_shape()
        n_layers = n_layers or total_layers
        store = cls(cls.make_layout(shapes), n_layers, dtype)
        for fn in files:
            with safe_open(repo_dir / fn, "pt") as f:
                for k in f.keys():
                    m = LAYER_RE.match(k)
                    if m and int(m[1]) < n_layers:
                        store.put(int(m[1]), m[2], f.get_tensor(k))
                    elif not m:
                        store.other[k] = f.get_tensor(k).to(dtype)
        store.check_complete()
        return store
