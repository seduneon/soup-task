"""Layer-streamed Qwen3 with LoRA: forward, KV-cached decode passes, chunked log-probs, layer-by-layer backward.

Weights live in a host `LayerStore` (pinned on CUDA). `resident` layers get their own device slot, loaded once.
All other layers rotate through `n_buffers` device slots: while layer k computes, layer k+1 is copied on a
separate CUDA stream; an event per slot orders "compute done with the old layer" before "copy the next one in".

L2L order: each layer visit runs every micro-batch, so a weight pass costs one copy per streamed layer no
matter how many micro-batches there are. Every user of the weights (prefill, decode, ref log-probs, train
forward, train backward) goes through `visit`, so they all share the same streaming and the same counters.

Training (`train_grads`):
  1. no-grad forward over all layers, saving each layer's input hidden state (the boundary checkpoint;
     offloaded to pinned host memory on CUDA);
  2. the head (final norm + tied lm_head + log-softmax) in chunks of `chunk` selected tokens, each chunk a
     separate graph, giving dL/dh at the last boundary;
  3. layers in reverse: reload the layer, recompute it with grad from its checkpoint, backprop dL/dh_out,
     which accumulates that layer's LoRA grads and yields dL/dh_in. No graph spans more than one layer.

The loss must be additive over tokens (sum of per-token terms with fixed weights, e.g. GRPO with
precomputed advantages / lengths, or a per-token KL), because it is evaluated chunk by chunk.
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding

from lora import bind_layer, wrap_layer
from store import register_pinned, unregister_pinned

IGNORE = -100


def _set_param(module, name, tensor):
    *path, leaf = name.split(".")
    mod = module.get_submodule(".".join(path)) if path else module
    setattr(mod, leaf, nn.Parameter(tensor, requires_grad=False))


class Slot:
    """One device-side decoder layer whose weights are views into one flat device buffer."""

    def __init__(self, config, store, device, lora):
        with torch.device("meta"):
            layer = Qwen3DecoderLayer(config, layer_idx=0)
        self.flat = torch.empty(store.numel, dtype=store.dtype, device=device)
        for name, t in store.views(self.flat).items():
            _set_param(layer, name, t)
        meta = [n for n, p in layer.named_parameters() if p.is_meta] + [n for n, b in layer.named_buffers() if b.is_meta]
        assert not meta, f"slot tensors not covered by the store: {meta}"
        self.layer = layer.eval()
        self.wrappers = wrap_layer(layer, lora.scale, lora.targets)
        self.layer_id = None


class StreamedQwen3:
    def __init__(self, config, store, lora, device, resident=(), n_buffers=2, offload=None, release_host=False):
        """release_host (CUDA, training runs): drop the host copies of the `resident` layers and of the embedding
        once they are on the device. They can then never be streamed again, so the dynamic arm must keep
        `resident` as its floor (it does)."""
        assert config._attn_implementation == "sdpa", "set config._attn_implementation = 'sdpa'"
        self.cfg, self.store, self.lora, self.device = config, store, lora, torch.device(device)
        self.L, self.dtype = store.n_layers, store.dtype
        self.cuda = self.device.type == "cuda"
        self.offload = self.cuda if offload is None else offload
        self.adapter_enabled = True
        self.embed = store.other["model.embed_tokens.weight"].to(self.device)
        tied = getattr(config, "tie_word_embeddings", True)
        self.head_w = self.embed if tied else store.other["lm_head.weight"].to(self.device)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(self.device, self.dtype)
        with torch.no_grad():
            self.norm.weight.copy_(store.other["model.norm.weight"])
        self.norm.weight.requires_grad_(False)
        self.rotary = Qwen3RotaryEmbedding(config, device=self.device)
        self.n_buffers = n_buffers
        self.copy_stream = torch.cuda.Stream(self.device) if self.cuda else None
        self.loads = 0       # streamed-layer copies issued (one per streamed layer per weight pass)
        self.promotions = 0  # copies made to (re)fill resident slots
        self.passes = 0      # weight passes (calls of visit)
        self.profile = False  # CUDA: time every layer copy and every compute-stream wait (overlap evidence)
        self._prof = {"copies": [], "waits": [], "bytes": 0}
        self._arena = None   # pinned host arena for boundary checkpoints (exact size, grown on demand)
        self.res, self.buf, self._free = {}, [], []
        self.set_resident(resident)
        if release_host and self.cuda:
            store.release(self.resident)
            store.other.clear()

    # ------------------------------------------------------------ residency
    def set_resident(self, resident):
        """Change which layers own a device slot. Evicted slots are freed; new ones are filled from the store.
        Called between weight passes only (the dynamic arm calls it before every decode pass)."""
        new = set(resident)
        assert all(0 <= i < self.L for i in new)
        for i in [i for i in self.res if i not in new]:
            del self.res[i]
        for i in sorted(new - set(self.res)):
            assert self.store.flat[i] is not None, f"layer {i} has no host copy (released): cannot load it"
            s = Slot(self.cfg, self.store, self.device, self.lora)
            s.flat.copy_(self.store.flat[i])
            s.layer_id = i
            self.res[i] = s
            self.promotions += 1
        self.resident = sorted(self.res)
        want = min(self.n_buffers, self.L - len(self.res))
        while len(self.buf) < want:
            self.buf.append(Slot(self.cfg, self.store, self.device, self.lora))
            self._free.append(None)
        # extra buffers are kept: the dynamic arm only grows residency above the regular count
        if self.cuda:  # every layer that can be streamed must be page-locked, or its copy is synchronous
            self.store.pin([i for i in range(self.L) if i not in self.res])

    @contextmanager
    def adapter_off(self):
        """Base model (reference log-probs). The LoRA tensors are untouched, only not bound."""
        prev, self.adapter_enabled = self.adapter_enabled, False
        try:
            yield
        finally:
            self.adapter_enabled = prev

    # ------------------------------------------------------------ weight streaming
    def visit(self, order):
        """Yield (layer index, module) in `order`, prefetching streamed layers into the rotating slots."""
        order = list(order)
        streamed = [i for i in order if i not in self.res]
        # free[s] persists across passes: the first copies of a pass must also wait for the previous pass's
        # last use of the slot (otherwise only an incidental host sync between passes keeps it correct)
        n, ready, free = len(self.buf), [None] * len(self.buf), self._free
        self.passes += 1

        def issue(k):
            s, slot = k % n, self.buf[k % n]
            src = self.store.flat[streamed[k]]
            if self.cuda:
                with torch.cuda.stream(self.copy_stream):
                    if free[s] is not None:
                        self.copy_stream.wait_event(free[s])
                    if self.profile:
                        c0 = torch.cuda.Event(enable_timing=True)
                        c0.record(self.copy_stream)
                    slot.flat.copy_(src, non_blocking=True)
                    ready[s] = torch.cuda.Event(enable_timing=self.profile)
                    ready[s].record(self.copy_stream)
                    if self.profile:
                        self._prof["copies"].append((c0, ready[s]))
                        self._prof["bytes"] += src.numel() * src.element_size()
            else:
                slot.flat.copy_(src)
            slot.layer_id = streamed[k]
            self.loads += 1

        for k in range(min(n, len(streamed))):
            issue(k)
        k = 0
        for i in order:
            if i in self.res:
                slot = self.res[i]
            else:
                s = k % n
                slot = self.buf[s]
                assert slot.layer_id == i, f"slot holds layer {slot.layer_id}, expected {i}"
                if self.cuda:
                    cur = torch.cuda.current_stream(self.device)
                    if self.profile:  # w1 - w0 = time the compute stream sat idle waiting for this copy
                        w0, w1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        w0.record(cur)
                    cur.wait_event(ready[s])
                    if self.profile:
                        w1.record(cur)
                        self._prof["waits"].append((w0, w1))
            if self.adapter_enabled:
                bind_layer(slot.wrappers, self.lora, i)
            else:
                for w in slot.wrappers.values():
                    w.bind(None, None, None)
            slot.layer.self_attn.layer_idx = i  # the KV cache is indexed by the real layer
            yield i, slot.layer
            if i not in self.res:
                if self.cuda:
                    free[s] = torch.cuda.Event()
                    free[s].record(torch.cuda.current_stream(self.device))
                if k + n < len(streamed):
                    issue(k + n)
                k += 1

    def overlap_report(self):
        """Copy/compute overlap since the last call (CUDA with profile=True; None otherwise). GPU-side times
        from CUDA events: copy_ms = sum of copy durations on the copy stream, stall_ms = sum of the compute
        stream's waits on a copy. stall_ms << copy_ms means the copies are hidden behind compute; stall_ms ~
        copy_ms means the phase is transfer-bound (H2)."""
        if not (self.cuda and self.profile):
            return None
        torch.cuda.synchronize(self.device)
        p, self._prof = self._prof, {"copies": [], "waits": [], "bytes": 0}
        copy_ms = sum(a.elapsed_time(b) for a, b in p["copies"])
        stall_ms = sum(a.elapsed_time(b) for a, b in p["waits"])
        return {"copies": len(p["copies"]), "copy_ms": copy_ms, "stall_ms": stall_ms,
                "h2d_gbps": p["bytes"] / max(copy_ms, 1e-9) / 1e6, "hidden_frac": 1 - stall_ms / max(copy_ms, 1e-9)}

    # ------------------------------------------------------------ inputs
    def context(self, ids, attn_mask=None, mb=None):
        """Micro-batch slices with their attention mask and rotary embeddings, for full sequences.

        Left padding: positions count only real tokens; pad queries attend to themselves only (their output is
        never used, but an all-False row would make SDPA return NaN). No padding: mask None, SDPA causal path.
        """
        B, S = ids.shape
        mb = mb or B
        if attn_mask is None:
            attn_mask = torch.ones(B, S, dtype=torch.long, device=ids.device)
        pos = (attn_mask.long().cumsum(-1) - 1).clamp(min=0).to(self.device)
        if bool(attn_mask.all()):
            mask = None
        else:
            causal = torch.ones(S, S, dtype=torch.bool, device=self.device).tril()
            mask = (causal[None, None] & attn_mask.bool().to(self.device)[:, None, None, :]) \
                | torch.eye(S, dtype=torch.bool, device=self.device)[None, None]
        out = []
        for a in range(0, B, mb):
            s = slice(a, min(a + mb, B))
            out.append((s, None if mask is None else mask[s], self.rope(pos[s])))
        return out

    def rope(self, pos):
        return self.rotary(torch.empty(0, dtype=self.dtype, device=self.device), pos)

    # ------------------------------------------------------------ core loop
    def embed_tokens(self, ids, ctx):
        return [F.embedding(ids[s].to(self.device), self.embed) for s, *_ in ctx]

    def run_layers(self, hs, ctx, cache=None, keep=False, col=0):
        """One weight pass over all layers in L2L order. hs: hidden states per micro-batch.
        cache: KV cache (written at column `col` for the micro-batch's rows). keep: return boundary checkpoints."""
        ckpt = []
        if keep and self.offload:
            per_layer = sum(h.numel() for h in hs)
            arena = self._arena_for(self.L * per_layer)
            offs = [0]
            for h in hs:
                offs.append(offs[-1] + h.numel())
        for i, layer in self.visit(range(self.L)):
            if keep:
                ckpt.append([self._save(arena[i * per_layer + offs[j]:i * per_layer + offs[j + 1]] if self.offload else None, h)
                             for j, h in enumerate(hs)])
            out = []
            for h, (s, m, pe) in zip(hs, ctx):
                if cache is not None:
                    cache.rows, cache.col = s, col
                out.append(layer(h, attention_mask=m, position_embeddings=pe, past_key_value=cache)[0])
            hs = out
        return hs, ckpt

    def head(self, h):
        """Final norm + tied lm_head, in the compute dtype."""
        return self.norm(h) @ self.head_w.t()

    # ------------------------------------------------------------ checkpoints
    def _arena_for(self, numel):
        """One host buffer for all boundary checkpoints of a step, page-locked at its exact size on CUDA.
        Grows (never shrinks) when a step has more tokens; regions are fixed offsets, so a different
        sequence length does not reallocate anything until it exceeds the largest seen."""
        if self._arena is None or self._arena.numel() < numel:
            if self._arena is not None and self.cuda:
                torch.cuda.synchronize(self.device)  # no copy may still be using the old arena
                unregister_pinned(self._arena)
            self._arena = None
            self._arena = torch.empty(numel, dtype=self.dtype)
            if self.cuda:
                register_pinned(self._arena)
        return self._arena

    def _save(self, region, h):
        if not self.offload:
            return h
        buf = region.view(h.shape)
        buf.copy_(h, non_blocking=self.cuda)
        return buf

    def _load(self, c):
        return c.to(self.device, non_blocking=self.cuda) if self.offload else c

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def logits(self, ids, attn_mask=None, mb=None):
        """Full logits [B, S, V] (tests and small evals only: B·S·V is large)."""
        ctx = self.context(ids, attn_mask, mb)
        hs, _ = self.run_layers(self.embed_tokens(ids, ctx), ctx)
        return torch.cat([self.head(h) for h in hs])

    @torch.no_grad()
    def logprobs(self, ids, attn_mask, targets, mb=None, chunk=256):
        """fp32 log-prob of every target token [B, S] (0 where IGNORE), without full logits."""
        ctx = self.context(ids, attn_mask, mb)
        targets = targets.to(self.device)
        hs, _ = self.run_layers(self.embed_tokens(ids, ctx), ctx)
        out = torch.zeros(targets.shape, dtype=torch.float32, device=self.device)
        for (s, *_), h in zip(ctx, hs):
            t, o = targets[s].reshape(-1), out[s].view(-1)
            sel = (t != IGNORE).nonzero().squeeze(1)
            hf = h.reshape(-1, h.shape[-1])
            for c in range(0, len(sel), chunk):
                rows = sel[c:c + chunk]
                o[rows] = self._target_logp(hf[rows], t[rows])
            out[s] = o.view(out[s].shape)
        return out

    def _target_logp(self, x, t):
        return torch.log_softmax(self.head(x).float(), -1).gather(-1, t[:, None]).squeeze(1)

    # ------------------------------------------------------------ training
    def train_grads(self, ids, attn_mask, targets, loss_fn, mb=None, chunk=256):
        """Accumulate LoRA grads of sum(loss_fn) over all selected tokens; return the loss.

        targets [B, S]: position t predicts targets[:, t] (IGNORE = not in the loss).
        loss_fn(logp, flat_idx): logp fp32 [n] log-prob of each selected target, flat_idx [n] its index in the
        flattened B·S grid. Must be a sum of per-token terms.
        """
        ctx = self.context(ids, attn_mask, mb)
        S = ids.shape[1]
        targets = targets.to(self.device)
        with torch.no_grad():
            hs, ckpt = self.run_layers(self.embed_tokens(ids, ctx), ctx, keep=True)
        total = torch.zeros((), dtype=torch.float32, device=self.device)
        grads = []
        for (s, *_), h in zip(ctx, hs):
            t = targets[s].reshape(-1)
            sel = (t != IGNORE).nonzero().squeeze(1)
            hf = h.reshape(-1, h.shape[-1])
            g = torch.zeros_like(hf)
            for c in range(0, len(sel), chunk):
                rows = sel[c:c + chunk]
                x = hf[rows].detach().requires_grad_()
                with torch.enable_grad():
                    loss = loss_fn(self._target_logp(x, t[rows]), s.start * S + rows)
                loss.backward()
                g.index_add_(0, rows, x.grad)
                total += loss.detach()
            grads.append(g.view_as(h))
        del hs
        for i, layer in self.visit(reversed(range(self.L))):
            for j, (_, m, pe) in enumerate(ctx):
                x = self._load(ckpt[i][j]).detach().requires_grad_()
                with torch.enable_grad():
                    y = layer(x, attention_mask=m, position_embeddings=pe)[0]
                torch.autograd.backward(y, grads[j])
                grads[j] = x.grad
            ckpt[i] = None
        return total

    # ------------------------------------------------------------ non-L2L reference
    def autograd_grads(self, ids, attn_mask, targets, loss_fn, mb=None):
        """Reference: ordinary autograd over the whole model, one graph per micro-batch (gradient accumulation).
        Only valid with every layer resident (a streamed slot is overwritten while the graph still needs it)."""
        assert len(self.res) == self.L, "reference path needs every layer resident"
        S = ids.shape[1]
        targets = targets.to(self.device)
        total = torch.zeros((), dtype=torch.float32, device=self.device)
        for s, m, pe in self.context(ids, attn_mask, mb):
            h = F.embedding(ids[s].to(self.device), self.embed)
            for i, layer in self.visit(range(self.L)):
                h = layer(h, attention_mask=m, position_embeddings=pe)[0]
            t = targets[s].reshape(-1)
            sel = (t != IGNORE).nonzero().squeeze(1)
            logp = torch.log_softmax(self.head(h).float(), -1).reshape(-1, self.head_w.shape[0])
            logp = logp[sel].gather(-1, t[sel, None]).squeeze(1)
            loss = loss_fn(logp, s.start * S + sel)
            loss.backward()
            total += loss.detach()
        return total
