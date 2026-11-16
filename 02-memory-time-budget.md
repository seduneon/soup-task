# 02 — Part 1: memory and time budget, step by step

Core mental model: **a GRPO step is a sequence of phases, each with a different set of live tensors.** Budget each phase; the peak is the worst phase; that phase decides how many layers can stay resident.

All numbers below are for Qwen3-4B, computed from config.json. Values marked *assumed* must be replaced by measurements.

## The split

| Step | Question | Needs |
|---|---|---|
| 1. Weight inventory | Bytes of every weight tensor (target, embedding, drafter) | config.json |
| 2. Trainable state | LoRA weights + grads + Adam | LoRA choice |
| 3. Per-token costs | KV cache, hidden states, logits per token | config.json |
| 4. Phase timeline | What is alive in each phase; peak per phase | 1+2+3 |
| 5. Resident layers | How many layers stay on the GPU under the cap | 4 |
| 6. Time model | Generation and training time from PCIe + compute | 1+5 |
| 7. Merge peak | GPU and CPU peak during merge | 1+2 |
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
- Drafter: 5 × 100.9M + fc 32.8M = 537.4M params = **1025 MiB** (no own embedding).

Verify against the real checkpoint headers (no full download):

```python
# step1_inventory.py
from huggingface_hub import get_safetensors_metadata
from collections import defaultdict

def inventory(repo):
    meta = get_safetensors_metadata(repo)
    groups, total = defaultdict(int), 0
    for f in meta.files_metadata.values():
        for name, t in f.tensors.items():
            n = 1
            for d in t.shape: n *= d
            key = "layer." + ".".join(name.split(".")[3:-1]) if ".layers." in name else name
            groups[key] += n; total += n
    for k, n in sorted(groups.items()):
        print(f"{k:45s} {n:>14,d}  {n*2/2**20:9.1f} MiB")
    print(f"{'TOTAL':45s} {total:>14,d}  {total*2/2**30:9.2f} GiB")

inventory("Qwen/Qwen3-4B")
inventory("z-lab/Qwen3-4B-DFlash-b16")
```

Check: "layer.*" rows / 36 match the table; no separate `lm_head.weight` (proves tying); drafter has no embedding.

## Step 2 — Trainable state (LoRA + Adam)

Bytes per trainable param (fp32): weight 4 + grad 4 + Adam m 4 + Adam v 4 = **16 B**.

LoRA params per adapted linear = r × (in + out).

| Choice (r = 16) | Trainable params | Memory |
|---|---|---|
| All 7 linears | 33.0M | **504 MiB** |
| q, v only | 5.9M | 90 MiB |

Which modules get LoRA is a real budget decision — state it.

## Step 3 — Per-token costs

| Term | Formula | Per token |
|---|---|---|
| KV cache | 2 × L × n_kv × hd × 2 B | 144 KiB |
| Boundary activation (one layer) | h × 2 B | 5 KiB (× 36 layers = 180 KiB) |
| Drafter context features | 5 × h × 2 B | 25 KiB |
| Logits for log-prob / CE | V × **14 B** | 2.03 MiB |

**14 bytes/element, not 6** (paper §5.6): bf16 logits + fp32 upcast + fp32 log-softmax + fp32 gradient are all live at once. The paper's first-principles estimator was 2.33× too low here.

## Step 4 — Phase timeline

Example config (*assumed*, choose your own): G = 8 rollouts, prompt P = 256, completion C = 512 → 6144 tokens/step.

| Phase | What happens | Weight passes | Phase-specific live tensors |
|---|---|---|---|
| A. Rollout generation | Prefill + decode (or draft+verify) | one per decode/verify step (hundreds) | KV cache, drafter context, small logits |
| B. Reference log-probs | Forward, adapter off, no grad | 1 | Logits chunk |
| C. Policy forward | Forward, LoRA on, grad, boundary checkpoints | 1 | Checkpoints accumulate |
| D. Backward + optimizer | Recompute each layer, backprop, Adam on LoRA | 1 (reverse) | Checkpoints, one layer's internals, LoRA grads |

Persistent across phases: CUDA context, embedding, drafter, prefetch buffers, resident layers, LoRA + optimizer state. KV cache can be freed before C/D.

| Term | Example (MiB) | Notes |
|---|---|---|
| CUDA context | ~400–600 | Measure: `torch.cuda.init()` then nvidia-smi; cuBLAS workspace appears on first matmul |
| Embedding | 742 | Resident |
| Prefetch buffers (2) | 385 | Double buffering |
| Each resident layer | 192.5 | The knob |
| Drafter | 1025 | Resident (if DFlash used) |
| LoRA + Adam (all linears) | 504 | |
| KV cache (phase A) | 864 | 6144 tok × 144 KiB; + 16 slots/seq for the verify block |
| Drafter context + phase-A logits | ~100–150 | G·16·V·4 ≈ 74 MiB per verify pass |
| Boundary checkpoints (C/D) | 1080 | 30 MiB/layer; can offload to pinned CPU via `saved_tensors_hooks` |
| Intra-layer recompute (D) | 690 (whole group) / 86 (one sequence) | Only one layer's worth at a time |
| Logits, unchunked | 8.3 GiB ✗ | Must chunk: 256 tokens → 519 MiB |
| Reference log-probs | ~24 KiB | Memory trivial; costs one extra pass in time |

Phase sums without resident layers (example):
- **A:** 500 + 742 + 385 + 1025 + 504 + 864 + ~150 ≈ **4170 MiB**
- **D, whole group at once:** 500 + 742 + 385 + 1025 + 504 + 1080 + 690 + 519 ≈ **5445 MiB**
- **D, one sequence per micro-batch, checkpoints offloaded:** ≈ **3760 MiB**

## Step 5 — Resident layers

resident = floor((cap − binding_phase_sum − safety_margin) / 192.5 MiB)

- Decide what "8 GB" means: 8192 MiB or 8·10⁹ B = 7629 MiB. State it.
- Safety margin for fragmentation ~5–10%.
- Example: 8192 − 5445 − ~500 → **~11 resident layers** (whole-group training). With micro-batching + offloaded checkpoints, phase A binds → **~18**.

Trade-off to justify: each resident layer saves 192.5 MiB of PCIe traffic on *every* pass, and generation makes hundreds of passes → residency mostly speeds up generation. Micro-batching saves VRAM but each micro-batch re-reads the streamed weights — unless you use L2L order (each layer loaded once for all k micro-batches; costs host RAM for boundary activations).

## Step 6 — Time model

Measure first (don't assume):
1. **Pinned H2D bandwidth**: copy a 200 MB pinned tensor, best of 30, idle and on the live copy stream. PCIe 4.0 x16 ~22 GB/s, 3.0 ~12 GB/s; Vast hosts are often x8 or on risers.
2. **Achievable bf16 GEMM TFLOPS** at your real shapes, **in the same session, SM clock recorded** (paper §5.4: a ceiling is only comparable to throughput measured in the same session).
3. **Per-layer fixed overhead**: a tiny-token pass (paper measured ~10 ms per layer visit of Python/launch cost on its laptop).

Per layer: t_layer ≈ max(copy, compute) + overhead.
- Copy per layer = 192.5 MiB / BW ≈ **9.2 ms at 22 GB/s** (16.8 ms at 12 GB/s).
- Compute per layer = 2 × 100.9M × tokens / TFLOPS (forward). Training uses C = 6 FLOPs/param/token (forward + checkpoint recompute + dL/dx; no dL/dW for frozen base); lm_head at C = 4.

Example (*assumed* 22 GB/s, 25 TFLOPS, 14 resident → 22 streamed):

| Item | Estimate |
|---|---|
| One streamed pass | 22 × 9.2 ms ≈ 0.20 s |
| Decode/verify pass compute (8 seq × 16 tokens) | ~1 ms/layer → transfer-bound |
| Plain decode, 512 tokens | 512 × 0.20 ≈ **~104 s** (+ overhead) |
| DFlash, τ ≈ 3.5 (*assumed*) | ~146 passes ≈ **~30 s** (+ drafter + overhead) |
| Training compute, 6144 tokens | 6 × 3.63B × 6144 ≈ 1.4·10¹⁴ FLOP ≈ **~5.7 s** → compute-bound |
| Training transfer | 2 × 0.20 ≈ 0.4 s (hidden) |
| Reference pass | ≈ 1.8 s |
| Prefill (2048 prompt tokens) | ≈ 0.6 s |

Conclusion to test: generation dominates the step (~90%); training is compute-bound (like the paper), decode is transfer-bound (the paper's < 128 tokens/pass caveat). Batch effective τ < mean τ (the step waits for the slowest sequence). Per-layer overhead × 36 × passes can rival transfer time.

## Step 7 — Merge peak

Tensor by tensor: W bf16 → fp32, + (α/r)·B·A in fp32 → bf16 → write.
- **GPU peak** ≈ context + largest LoRA-targeted tensor (MLP 47.5 MiB bf16, ×2 for fp32, + fp32 delta) ≈ **context + ~250 MiB**. Never move the 742 MiB embedding (no LoRA on it).
- **CPU peak**: `safetensors.save_file` needs a whole output shard in memory → peak ≈ largest output shard you write + one tensor + page cache of mmap'd inputs. Smaller output shards → lower peak.
- Report the measured peak for both.

## Step 8 — CPU RAM (host ≥ 16 GB)

| Term | Size |
|---|---|
| Pinned store, streamed layers | 22 × 192.5 ≈ 4.1 GiB (6.8 GiB if all 36) |
| Pinned activation offload (if used / L2L) | ~1 GiB per group at the example size; scales with k |
| Python + torch + CUDA host side | ~2–3 GiB |
| Loading transients | load → `.to()` → `pin_memory()` creates 3 copies (paper trap 3) — allocate pinned once, fill with `copy_` |
| Drafter load transient, dataset buffers | small |

Check `ulimit -l` (memlock) in the container; pinning may fail or fall back to pageable → synchronous copies.

## Step 9 — Measure and explain the gaps

| Measurement | Includes |
|---|---|
| `torch.cuda.max_memory_allocated()` | Live tensors only — what the formulas predict |
| `torch.cuda.max_memory_reserved()` | + caching-allocator pool (rounding, fragmentation) |
| nvidia-smi `memory.used` | + CUDA context, cuBLAS/cuDNN workspaces, kernel images — **what the cap is judged on** |

- Call `torch.cuda.reset_peak_memory_stats()` at each phase boundary; log per phase.
- Sample for the whole run: `nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm --format=csv -lms 200 > nvsmi.csv` (a deliverable).
- Enforcing the cap on a 12 GB card: `torch.cuda.set_per_process_memory_fraction` does not count the context — you must verify with nvidia-smi.

Typical memory gaps: logits 14 vs 6 B/elem; autograd holding extra tensors; attention kernel workspace; fragmentation from variable lengths; KV over-allocation.

Typical time gaps: effective PCIe below idle measurement; `.item()` / `.cpu()` syncs breaking overlap; per-layer Python/launch overhead; batch τ below mean τ; thermal/clock state; pageable fallback.

**Deliverable tip:** write steps 1–6 as a script where every term is one line with its formula, and commit it *before* renting — the predicted column must exist before the measured one.
