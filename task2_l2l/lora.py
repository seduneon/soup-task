"""LoRA kept outside the layer modules.

A streamed layer slot is reused by many layers, so the adapter cannot live inside the module. Each layer's
A and B are parameters of `LoRAParams`, and `LoRALinear` is bound to the current layer's pair before every
visit. The resident reference binds the same tensors permanently, so both paths train the same parameters.
"""
import math

import torch
from torch import nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MODULE_OF = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
             "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}


class LoRALinear(nn.Module):
    """y = base(x) + scale · (x A^T) B^T. The delta is computed in the LoRA dtype (fp32) and cast back."""

    def __init__(self, base: nn.Linear, scale: float):
        super().__init__()
        self.base, self.scale = base, scale
        self.A = self.B = None
        self.bound_to = None  # (layer index, name): tripwire against a stale binding

    def bind(self, A, B, tag):
        # Plain attributes, not registered parameters: the slot module must not own any layer's adapter.
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "B", B)
        self.bound_to = tag

    def forward(self, x):
        y = self.base(x)
        if self.A is None:
            return y
        d = (x.to(self.A.dtype) @ self.A.t()) @ self.B.t()
        return y + (self.scale * d).to(y.dtype)


class LoRAParams(nn.Module):
    """A [r, in] and B [out, r] for every (layer, target). PEFT init: A Kaiming-uniform, B zero."""

    def __init__(self, shapes, n_layers, r=16, alpha=32, targets=TARGETS, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.r, self.alpha, self.scale, self.targets, self.n_layers = r, alpha, alpha / r, tuple(targets), n_layers
        self.p = nn.ParameterDict()
        for i in range(n_layers):
            for t in self.targets:
                out_f, in_f = shapes[t]
                A = torch.empty(r, in_f, dtype=dtype, device=device)
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                self.p[f"{i}_{t}_A"] = nn.Parameter(A)
                self.p[f"{i}_{t}_B"] = nn.Parameter(torch.zeros(out_f, r, dtype=dtype, device=device))

    def pair(self, i, t):
        return self.p[f"{i}_{t}_A"], self.p[f"{i}_{t}_B"]

    def layer_params(self, i):
        return [x for t in self.targets for x in self.pair(i, t)]

    def clone_as(self, dtype):
        """Same values in another dtype (for the high-precision reference)."""
        new = LoRAParams.__new__(LoRAParams)
        nn.Module.__init__(new)
        new.r, new.alpha, new.scale, new.targets, new.n_layers = self.r, self.alpha, self.scale, self.targets, self.n_layers
        new.p = nn.ParameterDict({k: nn.Parameter(v.detach().to(dtype).clone()) for k, v in self.p.items()})
        return new

    def randomize_B(self, std=0.02, seed=0):
        """With B = 0 every A gradient is exactly 0 and a gradient test proves nothing."""
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for k, v in self.p.items():
                if k.endswith("_B"):
                    v.copy_(torch.randn(v.shape, generator=g, dtype=torch.float64).to(v.dtype) * std)


def wrap_layer(layer, scale, targets=TARGETS):
    """Replace the target nn.Linear modules of one decoder layer by LoRALinear wrappers."""
    wrappers = {}
    for t in targets:
        parent = getattr(layer, MODULE_OF[t])
        w = LoRALinear(getattr(parent, t), scale)
        setattr(parent, t, w)
        wrappers[t] = w
    return wrappers


def bind_layer(wrappers, lora, i):
    for t, w in wrappers.items():
        A, B = lora.pair(i, t)
        w.bind(A, B, (i, t))


def lora_shapes(config):
    """(out, in) of every target linear, from the model config."""
    h, ff, hd = config.hidden_size, config.intermediate_size, config.head_dim
    q, kv = config.num_attention_heads * hd, config.num_key_value_heads * hd
    return {"q_proj": (q, h), "k_proj": (kv, h), "v_proj": (kv, h), "o_proj": (h, q),
            "gate_proj": (ff, h), "up_proj": (ff, h), "down_proj": (h, ff)}
