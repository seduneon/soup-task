# 01 — Concepts you need for this task

## 1. Qwen3-4B architecture (verified from config.json)

| Field | Value |
|---|---|
| Decoder layers | 36 |
| hidden_size | 2560 |
| intermediate_size (MLP) | 9728 |
| Query heads / KV heads | 32 / 8 (GQA) |
| head_dim | 128 |
| vocab_size | 151,936 |
| tie_word_embeddings | true (embedding matrix = LM head, one tensor) |
| dtype | bfloat16 |
| eos_token_id | 151645 |

One decoder layer = RMSNorm → attention (q, k, v, o projections, plus q_norm / k_norm) → residual → RMSNorm → MLP (gate, up, down; SiLU) → residual. ≈ 100.9M params ≈ **192.5 MiB in bf16**. Full model ≈ 4.02B params ≈ 8.04 GB.

**Rule:** a linear layer `in → out` has `in × out` parameters; bf16 = 2 bytes each.

**GQA:** 8 KV heads are shared by 32 query heads, so k/v projections (2560→1024) are 4× smaller than q (2560→4096), and the KV cache is 4× smaller than with full multi-head attention.

## 2. Prefill vs decode

- **Prefill:** process the whole prompt in one forward pass. Weights are read once, amortised over many tokens → compute-bound when tokens are many.
- **Decode:** generate one token per forward pass. Every new token needs a full pass over all layers.
- **Why it matters here:** under streaming, every decode pass re-reads the whole streamed model over PCIe. 512 generated tokens ≈ 512 full model streams. Training amortises 2 streams over thousands of tokens. → Generation is the bottleneck of a streamed GRPO step.
- The paper's own caveat: below roughly 128 tokens per pass the step is fixed-cost-bound (transfers + per-layer overhead matter); above that it is compute-bound.

## 3. KV cache

- During decode, keep each past token's keys and values per layer so they aren't recomputed.
- Size per token = 2 (K and V) × layers × kv_heads × head_dim × 2 bytes = 2 × 36 × 8 × 128 × 2 = **147,456 B ≈ 144 KiB/token**.
- Under layer streaming, layer i's KV is only needed while layer i is on the GPU → it could be streamed too (design choice; we keep it on the GPU).
- The KV cache **grows during decode** (P → P + C tokens). Memory it has not claimed yet can hold extra resident layers that are evicted as it grows. That is the dynamic-streaming arm; it needs the KV cache allocated in pages rather than at full length up front.
- Speculative decoding needs KV **rollback** (crop to accepted length) after each verify.

## 4. Speculative decoding (general)

- A cheap drafter proposes k tokens; the target checks all k in **one** forward pass (a mini-prefill).
- **Greedy:** accept the longest prefix where draft == target argmax, plus one corrected/bonus token from the target. Output is token-identical to plain greedy decoding.
- **Sampling, textbook rule:** accept draft token x with prob min(1, p(x)/q(x)); on rejection, resample from normalize(max(0, p − q)). Output distribution equals the target's exactly.
- **Acceptance length τ** = tokens gained per target pass (accepted + 1). Speedup ≈ τ when target passes dominate.

## 5. DFlash specifically (bonus arm; read from the drafter repo's `dflash.py`)

- Drafter: `z-lab/Qwen3-4B-DFlash-b16`, 537.4M params = 5 Qwen3-shaped layers + one fusion projection `fc` (5·2560 → 2560). ~1025 MiB bf16. **No embedding or LM head of its own** — it reuses the target's `embed_tokens` and `lm_head`.
- Conditioning: hidden states from target layers **1, 9, 17, 25, 33** are concatenated, passed through `fc` and a norm → one context vector per past token.
- Drafting a block (block size 16): slot 0 = last confirmed token, slots 1–15 = mask token (id 151669). One forward pass fills all 15 in parallel ("block diffusion"). Attention is **non-causal**; queries come from the block, keys/values = concat(context, block).
- Verify: target runs on the 16-token block in one pass, `output_hidden_states=True`.
- **Acceptance in reference code = exact match**: target samples its own token at each position (`posterior`); accepted = leading run where draft == posterior, plus the target's token at the first mismatch. Every emitted token is the target's own sample → distribution is exactly the target's (lossless), but acceptance is lower than the ratio test.
- KV: target cache cropped to accepted length; drafter cache cropped too; context features kept only for accepted positions.
- **Reference code is batch-size 1** (`[0].item()`). GRPO needs a batch of G sequences with different acceptance per sequence → ragged KV lengths. This generalisation is your work.
- **Silent trap:** HF `hidden_states[0]` is the embedding output, not layer 0's output. An off-by-one in which layer you capture still "works" — acceptance just quietly drops. Compare your captured features against HF `output_hidden_states`.
- Drafter was trained against the **base** model's hidden states → as LoRA changes the target, acceptance should fall (task asks you to measure this).
- Reported headline: up to ~6× lossless acceleration on Qwen3-8B, ~2.5× faster than EAGLE-3 (z-lab project page).

## 6. GRPO

1. For each prompt, sample a **group** of G completions (e.g. 8).
2. Score each with the verifiable reward (plain Python: parse answer, compare).
3. **Advantage** = (r − group mean) / group std. Better-than-siblings answers pushed up, worse pushed down.
4. Loss = −advantage × (clipped) probability ratio (new/old), per token, plus optional KL penalty to the reference.

Key facts:
- If all G answers get the same reward → advantages are 0 → the prompt teaches nothing. Pick prompts with base pass rate ~30–70%.
- **Loss ≈ 0 by construction**: advantages are zero-mean per group and with one update per batch the ratio is 1. The loss *value* is uninformative; its *gradient* is not.
- **Reference model is free with LoRA**: reference log-probs = base with the adapter disabled. Costs one extra streamed forward pass in time, no extra weights.
- Mask prompt and padding tokens out of the loss. Use the identical chat template (`enable_thinking=False`) in rollout, training and eval.
- Log-probs need the LM head over 151,936 vocab → **chunk** the computation (see budget file: 14 bytes/element).

## 7. LoRA and merge

- LoRA: freeze W; train small B (out × r) and A (r × in). Effective weight W + (α/r)·B·A.
- PEFT initialises **B = 0** → a freshly created adapter is a no-op, and an adapter that failed to load is byte-identical to a fresh one. Any "did it load?" check must first write a non-zero B.
- **Merge** = fold the adapter into the base: W′ = W + (α/r)·B·A for each adapted tensor, saved as a standalone model. (It is *not* about mixing dtypes like int4/fp16 — no quantization here.)
- Do it tensor by tensor: load W (bf16), compute delta in fp32, add, cast to bf16, write, free. Merged model must match base+adapter up to numerical noise (bf16 rounding → justify the tolerance).

## 8. Pinned (page-locked) memory

- Normal ("pageable") RAM can be moved or swapped by the OS at any time.
- The GPU's copy engine (DMA) reads physical addresses directly, so CUDA won't DMA from pageable memory: it first copies (with the CPU) into a hidden pinned staging buffer, then DMAs. Extra copy, slower, and **`non_blocking=True` silently becomes synchronous** → copy/compute overlap disappears, no error.
- Pinned memory: the OS promises not to move those pages → direct, asynchronous DMA at full PCIe speed.
- Costs: taken away from the OS (over-pinning starves the system); allocation is slow (do it once); never modify/free a pinned buffer until the async copy reading it has finished (wait on its event).
- CUDA pins through the driver (`cudaHostAlloc`), not `mlock`, so `ulimit -l` is not the usual limit, and torch raises if pinning fails. The silent version is a copy from a tensor that was never pinned, such as a slice of a memory-mapped safetensors file: check `is_pinned()` and whether `copy_(non_blocking=True)` returns immediately (`bench.py pcie` reports both).

```python
cpu_w = torch.empty(shape, dtype=torch.bfloat16, pin_memory=True)
cpu_w.copy_(weights)                     # fill once; avoid load→.to()→pin_memory() triple copies
assert cpu_w.is_pinned()
with torch.cuda.stream(copy_stream):
    gpu_slot.copy_(cpu_w, non_blocking=True)
    ready_event.record()
```

First experiment on the GPU box: `task1_budget/bench.py pcie` times a 192.5 MiB layer copy pinned / pageable / memmap while GEMMs run → also gives the measured H2D bandwidth for part 1.

## 9. Layer streaming vs L2L (two different things)

**Layer streaming** (the paper; Soup's `stream_layers`):
- All decoder layers live in pinned CPU RAM; GPU holds a pool of 2 layer buffers + embedding + LoRA + activations.
- Forward, per layer i: wait for prefetch event of i → launch prefetch of i+1 on a dedicated copy stream → run layer i inside `torch.utils.checkpoint(use_reentrant=False)` → release buffer i−1.
- Backward: autograd re-runs each checkpointed layer → weights fetched again in reverse order. Two reads per step is the floor (dL/dx = Wᵀ·dL/dy needs W).
- Activation checkpointing at layer boundaries: save only each layer's input (tokens × 2560 × 2 B), recompute inside the layer during backward.

```
copy stream:    [copy L1][copy L2][copy L3] ...
compute stream:          [run  L1][run  L2] ...
```

**L2L (layer-to-layer, Soup's internal meaning; Pudipeddi et al. 2020)** — a processing *order* on top of streaming:
- Gradient accumulation with k micro-batches: each micro-batch streams the whole model → each layer loaded k times per optimizer step.
- L2L: load layer 0 once, run **all k micro-batches** through it, then layer 1, …; backward in reverse the same way → each layer loaded **once per optimizer step**.
- Cost: keep boundary activations for all k micro-batches (pinned CPU RAM, spill to disk if needed).
- Correctness target: losses and LoRA grads equal to plain gradient accumulation (bit-exact in fp32; in bf16 within 1.5× the GA-vs-GA noise floor — their rule A3-07).
- Fits GRPO: the group of G rollouts = k micro-batches. Possible extra: compute reference log-probs in the same layer visit (their DPO path does this).

The task text's own L2L definition (pinned CPU weights, current + prefetched layer, boundary checkpoints, overlap) describes the streaming part; Soup's L2L adds the layer-major order. Clarify with the team (question 1 in `00`).

**Minimal forward loop (learning version):**

```python
copy_stream = torch.cuda.Stream()
slots = [empty_layer_on_gpu(), empty_layer_on_gpu()]
load_async(layer_cpu[0], slots[0], copy_stream)
for i in range(36):
    torch.cuda.current_stream().wait_event(ready[i])
    if i + 1 < 36:
        load_async(layer_cpu[i+1], slots[(i+1) % 2], copy_stream)
    x = checkpoint(run_layer, slots[i % 2], x)   # via torch.func.functional_call
```

Primitives (paper §3.3): `requires_grad=False` on base weights (NOT `detach()` / `no_grad()` — those silently cut the graph); `torch.func.functional_call` to run an unmodified HF layer with pooled buffers; `saved_tensors_hooks` to move boundary activations to host.

## 10. Data streaming (part 2b) — not the same "streaming"

- Read prompts and rollout records lazily from disk (iterator over jsonl/parquet), not all into memory.
- Persist a cursor in the checkpoint: epoch, position, shuffle seed, RNG state.
- Test: kill mid-epoch, resume, show from logged sample IDs that it continues at the exact next sample — no duplicates, no gaps. Danger: a resumed run with a different shuffle looks perfectly normal.

## 11. Silent failures — the mindset

The run finishes, logs look healthy, a file is written, and the result is wrong. Examples: adapter saved as zeros; graph severed at base weights (upper layers still train, loss still falls); reward parser always returns 1; drafter fed the wrong layer; resume replays samples. Only comparisons against an **independent reference** catch these — not tests that assert the code did what it says.
