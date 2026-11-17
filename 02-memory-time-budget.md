# 02 — Part 1: memory and time budget, step by step

Core mental model: **a GRPO step is a sequence of phases, each with a different set of live tensors.** Budget each phase; the peak is the worst phase; that phase decides how many layers can stay resident.

**Source of truth:** `task1_budget/budget.py` (every term is one line with its formula) and its output `task1_budget/results/predicted.md`. The pre-registered hypotheses and the A/B protocol are in `task1_budget/README.md`. This file explains the reasoning; if a number here disagrees with the script, the script wins.

Default settings (all *assumed* until measured): 2 prompts × G = 8 rollouts (B = 16 sequences), prompt P = 256, completion C = 512, LoRA r = 16 on all 7 linears, **plain decode** (DFlash is a bonus arm), **regular streaming** (dynamic residency is the research arm).

## The split

| Step | Question | Needs |
|---|---|---|
| 1. Weight inventory | Bytes of every weight tensor (target, embedding, drafter) | checkpoint headers |
| 2. Trainable state | LoRA weights + grads + Adam | LoRA choice |
| 3. Per-token costs | KV cache, hidden states, logits per token | config.json |
| 4. Phase timeline | What is alive in each phase; peak per phase | 1+2+3 |
| 5. Resident layers | How many layers stay on the GPU under the cap | 4 |
| 6. Time model | Generation and training time from PCIe + roofline compute | 1+5 |
| 7. Merge peak (optional) | GPU and CPU peak during merge | 1+2 |
| 8. CPU RAM | Pinned store, loading transients, offload | 1 |
| 9. Measure & explain gaps | Predicted vs allocated / reserved / nvidia-smi | GPU, last |

Steps 1–3 are independent arithmetic; 4–8 combine them; only 9 needs the rented box.

## Step 1 — Weight inventory

| Tensor (per layer) | Shape | Params |
|---|---|---|
| q_proj | 2560 → 4096 | 10,485,760 |
| k_proj | 2560 → 1024 | 2,621,440 |
| v_proj | 2560 → 1024 | 2,621,440 |
| o_proj | 4096 → 2560 | 10,485,760 |
| gate_proj | 2560 → 9728 | 24,903,680 |
| up_proj | 2560 → 9728 | 24,903,680 |
| down_proj | 9728 → 2560 | 24,903,680 |
| 2 RMSNorm + q_norm + k_norm | 2·2560 + 2·128 | 5,376 |
| **Layer total** | | **100,930,816 ≈ 192.5 MiB** |

- 36 layers = **6.77 GiB** (the streamable part).
- Embedding / tied LM head = 151,936 × 2560 × 2 B = **742 MiB** (resident; the "vocabulary floor").
- Total 4.02B params ≈ 8.04 GB.
- Drafter (bonus arm only): 5 × 100.9M + fc 32.8M = 537.4M params = **1025 MiB** (no own embedding).

`budget.py` reads the real safetensors headers (no weight download) and asserts the hand formula, the tied LM head and the drafter's lack of an embedding.

## Step 2 — Trainable state (LoRA + Adam)

Bytes per trainable param (fp32): weight 4 + Adam m 4 + Adam v 4 always, + grad 4 **only during the train phase** (set grads to `None` after the step).

LoRA params per adapted linear = r × (in + out).

| Choice (r = 16) | Trainable params | Always resident | + grads in train |
|---|---|---|---|
| All 7 linears | 33.0M | **378 MiB** | +126 MiB |
| q, v only | 5.9M | 68 MiB | +23 MiB |

Which modules get LoRA is a real budget decision — state it.

## Step 3 — Per-token costs

| Term | Formula | Per token |
|---|---|---|
| KV cache | 2 × L × n_kv × hd × 2 B | 144 KiB |
| Boundary activation (one layer) | h × 2 B | 5 KiB (× 37 boundaries = 185 KiB) |
| Drafter context features (bonus) | 5 × h × 2 B | 25 KiB |
| Logits for log-prob / CE | V × **14 B** | 2.03 MiB |

**14 bytes/element, not 6** (Exact Layer Streaming §5.6): bf16 logits + fp32 upcast + fp32 log-softmax + fp32 gradient are all live at once. Unchunked logits for B·C = 8192 tokens would be 16 GiB, so log-probs **must** be chunked (256 tokens → 519 MiB with grad, 371 MiB without).

## Step 4 — Phase timeline

| Phase | What happens | Weight passes | Phase-specific live tensors |
|---|---|---|---|
| prefill | Prompts, `prefill_mb` = 2 sequences per chunk inside each layer | 1 | KV cache, one chunk's activations |
| decode | One token per sequence per pass (bonus: draft + verify a block) | C (C/τ) | KV cache, logits for B tokens |
| ref | Forward, adapter off, no grad | 1 | Boundary hidden, logits chunk |
| train | Forward keeping boundary checkpoints (offloaded to pinned host), then recompute + backward per layer, Adam | 2 | LoRA grads, one layer's saved tensors, logits chunk with grad |

Persistent across phases: CUDA context, embedding, prefetch buffers (2), resident layers, LoRA weights + Adam (and the drafter in the bonus arm). The KV cache is freed before ref/train.

Predicted totals with 20 resident layers (MiB): **prefill 7641 (binding)**, decode 7597, train 6839, ref 6411. Largest non-weight term: the KV cache, 1728 MiB at full length (B × 784 tokens × 144 KiB).

## Step 5 — Resident layers

resident = the largest r for which **every** phase fits under cap − reserve.

- "8 GB" = 8192 MiB on nvidia-smi, CUDA context included. Reserve 512 MiB for fragmentation and unmodelled workspaces → limit 7680 MiB.
- Regular streaming: **20 resident / 16 streamed**.
- Each resident layer saves one 192.5 MiB copy on **every** pass. Plain decode makes 512 passes per step, so one more resident layer is worth **4.4 s/step**. Training makes 2 passes and is FLOP-bound, so it gains nothing.
- Dynamic arm: the KV cache starts at P = 256 tokens and grows to 784. Layers kept resident in the memory it has not claimed yet give 20 → 26 resident during decode (mean 22.9). Predicted step 111.8 → 99.0 s (−11.5 %).

## Step 6 — Time model

Measure first (`task1_budget/bench.py`), don't assume:
1. **Pinned H2D bandwidth** while GEMMs run (the overlap condition), plus pageable and memmap sources to catch silently synchronous copies. PCIe 4.0 x16 ≈ 22 GB/s, 3.0 ≈ 12 GB/s; Vast hosts are often x8 or on risers, so read the link width under load.
2. **VRAM read bandwidth** at decode size: small-batch GEMMs are bound by reading weights from VRAM, not by FLOPs.
3. **Achievable bf16 GEMM TFLOPS** at the real training shape, **in the same session, SM clock recorded** (paper §5.4).
4. **Launch overhead per layer**, with tiny weights so no weight read is included.

Per layer visit: t ≈ max(copy, compute) + overhead for streamed layers, compute + overhead for resident ones, with compute = max(FLOPs / TFLOPS, weight bytes / VRAM bandwidth).

| Item (assumed 22 GB/s, 380 GB/s, 20 TFLOPS, 0.5 ms) | Estimate |
|---|---|
| Copy per streamed layer | 9.2 ms |
| Decode pass, 16 tokens | FLOPs 0.17 ms, VRAM read 0.53 ms per layer → **PCIe-bound** |
| Decode, 512 passes × 16 streamed layers | **90.9 s** |
| Prefill, 4096 tokens | 1.5 s (FLOP-bound) |
| Ref, 12288 tokens | 4.9 s (FLOP-bound) |
| Train forward + backward | 4.9 + 9.5 = 14.5 s (FLOP-bound, copies hidden) |
| **Step** | **111.8 s, 83 % generation, 55.9 s/prompt** |

Conclusions to test: generation dominates; decode is PCIe-bound and training FLOP-bound; residency only buys decode time; the PCIe link sets the step time (PCIe 3.0 x16 → 174.5 s). Bonus arm: DFlash pays only if the batch τ (slowest sequence) beats 1.77, or 2.06 to beat the dynamic arm.

## Step 7 — Merge peak (optional)

Tensor by tensor: W bf16 → fp32, + (α/r)·B·A in fp32 → bf16 → write.
- **GPU peak** ≈ context + the largest LoRA-targeted tensor in bf16 in, fp32, fp32 delta and bf16 out (9728 × 2560 × 12 B = 285 MiB) ≈ **785 MiB**. Never move the 742 MiB embedding (no LoRA on it).
- **CPU peak** ≈ host baseline + one input shard (mmap) + one output shard buffer ≈ **10.6 GiB**. Smaller output shards → lower peak.
- Watch the bf16 rounding: a LoRA delta smaller than about half a bf16 step of W (≈ 4e-5 for |W| ≈ 0.02) disappears in the merge. The merged-vs-adapter check in Part 3 must be able to see that.

## Step 8 — CPU RAM (host ≥ 16 GB)

| Term | Size |
|---|---|
| Pinned store, 16 streamed layers | 3.0 GiB (6.8 GiB if all 36) |
| Pinned boundary checkpoints, 37 × B·S·h·2 B | 2.2 GiB |
| Python + torch + CUDA host side | ~3 GiB (*assumed*) |
| Loading transient, one shard | 3.7 GiB |
| **Training total** | **11.9 GiB** |

Load → `.to()` → `pin_memory()` creates 3 copies (paper trap 3): allocate pinned once, fill with `copy_`. If pinning fails, torch raises an error; the silent failure is a copy from a source that was never pinned (for example a slice of a memory-mapped safetensors file), which makes `non_blocking=True` synchronous.

## Step 9 — Measure and explain the gaps

| Measurement | Includes |
|---|---|
| `torch.cuda.max_memory_allocated()` | Live tensors only — what the formulas predict |
| `torch.cuda.max_memory_reserved()` | + caching-allocator pool (rounding, fragmentation) |
| nvidia-smi `memory.used` | + CUDA context, cuBLAS workspaces, kernel images — **what the cap is judged on** |

- Call `torch.cuda.reset_peak_memory_stats()` at each phase boundary; log per phase.
- Sample for the whole run: `nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm --format=csv -lms 200 > nvsmi.csv` (a deliverable).
- Rent an 8 GB card (3060 Ti / 3070) so the hardware enforces the cap. On a larger card `torch.cuda.set_per_process_memory_fraction` does not count the context — you must verify with nvidia-smi.
- Acceptance rule (H10): each phase's allocated peak within ±10 % (or 300 MiB), nvidia-smi never above 8192, each phase time within ±30 %.

Typical memory gaps: logits 14 vs 6 B/elem; autograd holding extra tensors; attention kernel workspace; fragmentation from variable lengths and KV paging.

Typical time gaps: effective PCIe below idle measurement; `.item()` / `.cpu()` syncs breaking overlap; per-layer launch overhead; batch τ below mean τ (bonus arm); thermal/clock state; silently synchronous copies.
