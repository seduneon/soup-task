"""Rollout generation on the streamed engine: chunked prefill, then one weight pass per generated token.

Every decode pass streams the non-resident layers once for the whole batch (B sequences share each copy).
Layout of a rollout row: left-padded prompt in columns 0..P-1, generated tokens from column P; after a
sequence emits EOS its later columns are pad with attn = 0 (right padding). Positions count real tokens,
so training on the same row sees exactly the positions the decoder used.

Residency plans (dynamic arm): `plan(cols, cache) -> set of resident layers` is called before each decode
pass, with `cols` the number of columns the cache will hold after the pass.
"""
import time
from dataclasses import dataclass, field

import torch

from kvcache import PagedKV, StaticKV


@dataclass
class Rollout:
    ids: torch.Tensor          # [B, P+C] long (cpu)
    attn: torch.Tensor         # [B, P+C] long: 1 = real token
    comp: torch.Tensor         # [B, P+C] bool: generated token that belongs to the completion (incl. EOS)
    logp: torch.Tensor         # [B, C] fp32: log-prob of each generated token under the policy at T = 1
    finished: torch.Tensor     # [B] bool: emitted EOS
    P: int
    decode_passes: int
    kv_bytes: int
    resident_trace: list = field(default_factory=list)
    prefill_s: float = 0.0
    decode_s: float = 0.0

    @property
    def lengths(self):
        return self.comp.sum(-1)


def left_pad(prompts, pad_id):
    P = max(map(len, prompts))
    ids = torch.full((len(prompts), P), pad_id, dtype=torch.long)
    attn = torch.zeros(len(prompts), P, dtype=torch.long)
    for b, p in enumerate(prompts):
        ids[b, P - len(p):], attn[b, P - len(p):] = torch.tensor(p), 1
    return ids, attn


def sample(logits, temperature, top_p, generator):
    if temperature == 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits / temperature, -1)
    if top_p < 1.0:
        sp, si = probs.sort(-1, descending=True)
        drop = sp.cumsum(-1) - sp > top_p  # keep the smallest prefix whose mass reaches top_p
        sp = sp.masked_fill(drop, 0)
        probs = torch.zeros_like(probs).scatter_(-1, si, sp)
    return torch.multinomial(probs, 1, generator=generator).squeeze(1)


@torch.no_grad()
def generate(eng, prompts, max_new, eos_ids, pad_id, temperature=1.0, top_p=1.0, generator=None,
             prefill_mb=None, kv="static", plan=None, page=64):
    dev, cfg = eng.device, eng.cfg
    assert max_new >= 1, "max_new must be >= 1"
    B = len(prompts)
    p_ids, p_attn = left_pad(prompts, pad_id)
    P, S = p_ids.shape[1], p_ids.shape[1] + max_new
    ids = torch.full((B, S), pad_id, dtype=torch.long, device=dev)
    attn = torch.zeros(B, S, dtype=torch.long, device=dev)
    ids[:, :P], attn[:, :P] = p_ids.to(dev), p_attn.to(dev)
    comp = torch.zeros(B, S, dtype=torch.bool, device=dev)
    logp = torch.zeros(B, max_new, dtype=torch.float32, device=dev)
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    eos = torch.tensor(sorted(eos_ids), device=dev)
    Cache = {"static": StaticKV, "paged": PagedKV}[kv]
    extra = {"page": page} if kv == "paged" else {}
    cache = Cache(eng.L, B, S, cfg.num_key_value_heads, cfg.head_dim, eng.dtype, dev, **extra)
    trace = []

    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    sync()
    t0 = time.perf_counter()
    # prefill: all prompt columns, micro-batches of prefill_mb sequences per layer visit
    if plan is not None:
        eng.set_resident(plan(P, cache))
    trace.append(len(eng.res))
    ctx = eng.context(ids[:, :P], attn[:, :P], mb=prefill_mb)
    hs, _ = eng.run_layers(eng.embed_tokens(ids[:, :P], ctx), ctx, cache=cache, col=0)
    logits = torch.cat([eng.head(h[:, -1]) for h in hs]).float()
    sync()
    t1 = time.perf_counter()
    passes = 0
    for t in range(max_new):
        tok = sample(logits, temperature, top_p, generator)
        live = ~finished
        tok = torch.where(live, tok, torch.full_like(tok, pad_id))
        c = P + t
        ids[:, c], attn[:, c], comp[:, c] = tok, live.long(), live
        logp[:, t] = torch.where(live, torch.log_softmax(logits, -1).gather(-1, tok[:, None]).squeeze(1), 0.0)
        finished |= torch.isin(tok, eos) & live
        if bool(finished.all()) or t == max_new - 1:
            break
        # decode pass for column c: the new token attends to every real column and to itself
        pos = attn[:, :c].sum(-1, keepdim=True)  # real tokens before column c
        keys = attn[:, :c + 1].bool()
        keys[:, c] = True
        if plan is not None:
            eng.set_resident(plan(c + 1, cache))
        trace.append(len(eng.res))
        step_ctx = [(slice(0, B), keys[:, None, None, :], eng.rope(pos))]
        hs, _ = eng.run_layers(eng.embed_tokens(ids[:, c:c + 1], step_ctx), step_ctx, cache=cache, col=c)
        logits = eng.head(hs[0][:, -1]).float()
        passes += 1
    sync()
    end = P + t + 1
    return Rollout(prefill_s=t1 - t0, decode_s=time.perf_counter() - t1, ids=ids[:, :end].cpu(), attn=attn[:, :end].cpu(), comp=comp[:, :end].cpu(),
                   logp=logp[:, :t + 1].cpu(), finished=finished.cpu(), P=P, decode_passes=passes,
                   kv_bytes=cache.bytes, resident_trace=trace)


def memory_plan(eng, base_resident, limit_bytes, headroom_bytes):
    """Dynamic arm on CUDA: before each pass keep as many extra layers resident as fit, counting memory the
    way nvidia-smi does. Device-wide used = total - free includes the CUDA context, other processes and the
    allocator's cache; of that, our cached-but-free memory (reserved - allocated) is reusable. So:

        committed = (used - reserved) + allocated - extras      context + others + live tensors w/o extras
        extras    = floor((limit - committed - KV growth this pass - headroom) / layer)

    Never below base. `limit_bytes` is the nvidia-smi cap minus the fragmentation reserve."""
    base = set(base_resident)
    order = [i for i in range(eng.L) if i not in base]
    layer_bytes = eng.store.layer_bytes

    def plan(cols, cache):
        free, total = torch.cuda.mem_get_info(eng.device)
        reserved, alloc = torch.cuda.memory_reserved(eng.device), torch.cuda.memory_allocated(eng.device)
        extras_now = sum(1 for i in eng.res if i not in base) * layer_bytes
        committed = (total - free - reserved) + alloc - extras_now
        growth = max(cache.bytes_at(cols) - cache.bytes, 0) if hasattr(cache, "bytes_at") else 0
        n = int(max(limit_bytes - committed - growth - headroom_bytes, 0) // layer_bytes)
        return base | set(order[:min(n, len(order))])
    return plan
