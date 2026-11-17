# Part 1 — Memory and time budget

Predict GPU memory, host RAM and step time for one GRPO step (and the optional merge) **before running
anything**. Then compare with measured allocated / reserved / nvidia-smi and wall time, and explain the gaps.
Every prediction below is stated as a hypothesis with a decision rule, committed before the first GPU run.

| File | Role |
|---|---|
| `config.yaml` | Only the knobs the report argues about, the A/B arms, and hardware numbers marked *assumed* |
| `budget.py` | Reads the checkpoint headers once (cached in `results/inventory.json`) and asserts that the hand formulas match them. Then builds the per-phase memory ledger (every term = name, MiB, formula), the resident-layer choice, the roofline time model, merge and host-RAM peaks, the arms, a batch sweep and a residency sweep |
| `bench.py` | On the rented GPU: context size, PCIe bandwidth (pinned / pageable / mmap, with a check that copies are really async), VRAM read bandwidth, GEMM TFLOPS and launch overhead per layer, with nvidia-smi sampled throughout. Writes `results/measured_hw.yaml` |
| `results/predicted.md` | The committed prediction, with assumed hardware |

```bash
python3 budget.py                                                      # prediction (network once, for the headers)
python3 bench.py                                                       # on the GPU, same env vars as training
python3 budget.py --hw results/measured_hw.yaml --tag predicted_measured_hw
```

## Scope (after the team's clarification)

- **Required:** L2L, streaming, GRPO under the 8 GB cap. This budget's default arm is plain decode with
  regular streaming.
- **DFlash is a bonus arm** for A/B and extra research. It is modelled as a scenario, not the default.
- **Merge is optional.** Its peak is still budgeted (one line, 785 MiB), because Part 3 compares the merged
  model if it is built.

## Which streaming: regular baseline, dynamic as the research arm

Soup has two modes: regular streaming (`stream_layers`) and dynamic placement (D2, which picks GPU residency,
a RAM hotset and activation retention once per run from measured profiles). The suggested basis for dynamic
is 10Cache (arXiv 2511.14124). 10Cache profiles one iteration, then places tensors by priority (GPU, then
CPU, then NVMe), prefetches in execution order and reuses a pinned buffer pool.

For Qwen3-4B most of that choice space is empty:

| Dynamic option | Applies here? | Why |
|---|---|---|
| NVMe tier / RAM hotset | No | All 36 layers (6.8 GiB) fit in pinned host RAM, so nothing is read from disk after load |
| Activation retention (skip recompute of the last layers) | No | One layer's saved tensors for B = 16, S = 768 are about 1.5 GiB, more than the free VRAM |
| GPU residency, fixed per run | Yes | This is what `budget.py` already chooses analytically |
| **GPU residency that changes inside a step** | **Yes, untested** | During decode the KV cache grows from P to S tokens. The memory it has not claimed yet can hold extra resident layers, evicted as it grows |

The last row is where GRPO differs from the SFT setting D2 and 10Cache were measured on. 10Cache lists
inference and the KV cache as not evaluated, and D2 fixes its plan for the whole run. Plain decode makes 512
weight passes per step and is PCIe-bound, so each extra resident layer is worth 4.4 s/step there.

- **Baseline arm (`streaming: regular`):** a fixed resident count, the KV cache allocated at full length.
- **Research arm (`streaming: dynamic`):** the KV cache is allocated in pages carved from the buffers of evicted
  layers. Layers are kept resident while the KV cache is short and dropped as it grows, never below the
  regular count. Keeping a layer costs nothing extra: it is already being copied for that pass, it just isn't
  freed. Evicting is free too.

Following D2's rule, the dynamic arm is adopted only if the measured gain is at least 2 % with a CI excluding 0.
Otherwise regular stays. There is no profiling run to pay for: the plan comes from the same ledger.

## Fixed choices (not knobs)

- **L2L order:** each layer is loaded once and all micro-batches pass through it.
- **2 prefetch buffers** on a separate copy stream.
- **SDPA attention**, so no S × S score matrix is materialized.
- **LoRA in fp32 with Adam:** weights 4 B + Adam m, v 8 B always; grads 4 B only during training.
- **One update per batch:** old-policy log-probs equal the current ones, so no extra pass is needed.
- **Fixed-size buffers** (prefetch buffers, logits chunks, and the KV cache in the regular arm), so reserved
  stays close to allocated and the ledger stays checkable.
- **The KV cache is freed** before the training passes.

## Method

A GRPO step is a sequence of phases, each with its own set of live tensors. The peak is the worst phase, and
that phase decides how many layers can stay resident. A step holds B = prompts × G sequences; every weight
pass serves all of them.

| Phase | Work | Weight passes |
|---|---|---|
| prefill | prompts in chunks of `prefill_mb` sequences per layer | 1 |
| decode | plain decode (DFlash arm: draft + verify) | C (C/τ) |
| ref | no-grad forward over prompt + answer, adapter off | 1 |
| train | forward that keeps boundary checkpoints, then layer-by-layer recompute + backward, Adam on LoRA | 2 |
| merge (optional) | tensor by tensor: W + (α/r)·B·A in fp32 → bf16 | — |

Time per pass = Σ over streamed layers of max(copy, compute), + compute for resident layers, + launch overhead
per layer visit. Copy = layer bytes / H2D bandwidth. Compute is a roofline: max(FLOPs / TFLOPS, weight bytes /
VRAM bandwidth), with FLOPs = FLOP/param/token × (params + attention) × tokens (2 forward, 4 backward =
recompute + dX; no dW for the frozen base). The lm_head (and the drafter) use the same roofline.

## Prediction (assumed hardware: 22 GB/s H2D, 380 GB/s VRAM, 20 TFLOPS, 500 MiB context, 0.5 ms/layer)

Settings: 2 prompts × G = 8 (B = 16), P = 256, C = 512, LoRA r = 16 on all 7 linears, plain decode, regular
streaming, checkpoints offloaded to pinned host memory. Full ledger: [results/predicted.md](results/predicted.md).

| | Prediction |
|---|---|
| **Resident / streamed layers** | **20 / 16** |
| Peak per phase | **prefill 7641 MiB** (binding), decode 7597, train 6839, ref 6411 |
| Largest non-weight terms | KV cache 1728 · train: logits chunk with grad 519 · ref: logits chunk 371 |
| Merge GPU peak (optional) | 785 MiB (context + one 9728 × 2560 tensor in bf16, fp32 and its delta) |
| Host RAM | training 11.9 GiB (pinned layers + checkpoints + load transient) · merge 10.6 GiB |
| Step time | **111.8 s** = **55.9 s/prompt**: prefill 1.5, decode 90.9 (512 passes), ref 4.9, train 14.5 |
| Generation share | **83 %** |
| One more resident layer | −4.4 s/step |

| Arm | Resident | Step s | s/prompt | vs baseline |
|---|---:|---:|---:|---:|
| **regular (baseline)** | 20 | **111.8** | 55.9 | — |
| **dynamic** | 20 → 26 in decode (mean 22.9) | **99.0** | 49.5 | **−11.5 %** |
| DFlash bonus, τ = 3 | 11 | 74.5 | 37.3 | −33 % |
| DFlash bonus, τ = 2 | 11 | 101.1 | 50.6 | −10 % |
| DFlash + dynamic, τ = 2 / 3 | 11 → 18 | 95.1 / 70.5 | | −15 % / −37 % |
| risk: PCIe 3.0 x16 (12 GB/s) | 20 | 174.5 | 87.2 | +56 % |

Break-even points for the DFlash arm: it beats regular when τ > 1.77, and beats dynamic when τ > 2.06.

Batch: 2 prompts per step. Decode is PCIe-bound, so per-prompt cost falls with B (1 → 4 prompts: 83.7 → 43.1
s/prompt), but fewer steps per hour remain (43 → 21). 2 prompts gives two questions per update and 32 steps/h.

## Hypotheses and decision rules (pre-registered)

IDs follow `../docs/04-plan-and-hypotheses.md`. Each rule is fixed before the first GPU number.
- **Validity:** an arm whose validity check fails (cap exceeded on nvidia-smi, greedy output not token-identical to
  the reference, copies not async) has no verdict. It is never reported as "slower".
- **A/B protocol:** A-B-B-A blocks on one instance, same seed, prompts and warm-up (2 steps discarded), at
  least 4 blocks × 3 steps per arm. Report the median and a bootstrap 95 % CI of the paired difference. Every
  number is labelled with card, H2D bandwidth, SM clock and n.

| ID | Hypothesis (prediction) | Measurement | Accept if | If rejected |
|---|---|---|---|---|
| H1 | Generation dominates the step (83 %) | Per-phase wall time, cuda-synced at phase boundaries, ≥ 5 steps | median share ≥ 70 % | the time model is wrong about decode; find which term |
| H2 | Decode is PCIe-bound: each streamed layer costs one copy (9.2 ms at 22 GB/s) per pass | Decode pass time at r = 20, 16, 12, 8; linear fit vs streamed layers | slope within ±30 % of the measured copy time per layer, R² > 0.9 | decode is bound by overhead or compute; residency is worth less than predicted |
| H3 | Residency buys decode time only: −4.4 s/step per layer in decode, about 0 in train | Same sweep, per phase | train change < 1 % per layer, and decode gain ≥ 5 × train gain | train is not FLOP-bound; check TFLOPS and overlap |
| H6 | Dynamic residency beats regular on step time (−11.5 %) | A/B regular vs dynamic, same B | gain ≥ 2 % with CI excluding 0, greedy tokens identical, nvidia-smi ≤ 8192 | keep regular; report whether paging fragmentation (reserved − allocated) ate the gain |
| H10 | The ledger matches reality | Per-phase `max_memory_allocated/reserved` (reset at each boundary), nvidia-smi at 200 ms; phase wall times | each phase's allocated peak within ±10 % (or 300 MiB), nvidia-smi never above 8192, each phase time within ±30 % | name the missing or wrong term; recompute the resident count before the long run |
| H11 *(bonus)* | DFlash speeds up the whole step only if batch τ > 1.77 (> 2.06 to beat dynamic) | τ per pass and per batch at the sampling temperature; A/B step time | step time gain ≥ 2 % with CI excluding 0, and on-policy checks pass | report τ and why: drafter cost, residency lost to the drafter, slowest sequence |

## Assumptions (replace by measurement)

| Assumed | Value | Measured by |
|---|---|---|
| CUDA context + workspaces | 500 MiB | `bench.py ctx` (device-wide nvidia-smi) |
| Pinned H2D bandwidth | 22 GB/s | `bench.py pcie`, median while GEMMs run; link gen/width read under load |
| VRAM weight-read bandwidth | 380 GB/s | `bench.py vram`, 16-token GEMMs over one layer's linears |
| bf16 TFLOPS | 20 | `bench.py gemm`, whole-layer rate at 12288 tokens, SM clock logged |
| Launch overhead per layer | 0.5 ms | `bench.py overhead`, tiny weights with LoRA |
| P, C | 256, 512 | dataset: p99 of templated prompts; base-model samples: about p95 of answers |
| τ (bonus) | 3.0 | part 2c, at the sampling temperature, minimum over the batch |
| Host baseline | 3 GiB | process RSS during the run |

## Not modelled (expected gap sources)

- **Allocator effects:** rounding and fragmentation from variable rollout lengths (reserved − allocated), and
  from KV paging in the dynamic arm. Only the 512 MiB reserve covers them.
- **Autograd extras:** tensors it keeps that we didn't count, such as fp32 upcasts inside RMSNorm.
- **SDPA backend workspace.** The math fallback materializes the S × S scores.
- **`.item()`/`.cpu()` syncs** inside the layer loop that break copy/compute overlap.
- **Early stopping:** sequences that finish before C. Decode passes are set by the slowest sequence, so this is
  ignored (conservative only if one sequence runs to C).
- **Small terms we dropped:** LoRA FLOPs (≈1 %), logprob buffers (< 1 MiB), checkpoint D2H/H2D (≈ 2.2 GiB
  each way per step, hidden under FLOP-bound training).
- **Copies that are silently synchronous:** a source that isn't really pinned (for example a slice of a
  memory-mapped safetensors file) makes `copy_(non_blocking=True)` block the host, and overlap disappears.
  `bench.py pcie` reports `host_ms` and `src_pinned` per source.

## Measured vs predicted

*(filled in from part 2: `max_memory_allocated/reserved` reset at each phase boundary, nvidia-smi sampled every
200 ms for the whole run, wall time per phase.)*

| Phase | Predicted MiB | allocated | reserved | nvidia-smi | Gap explained by |
|---|---:|---:|---:|---:|---|
| prefill | 7641 | | | | |
| decode | 7597 | | | | |
| ref | 6411 | | | | |
| train | 6839 | | | | |
| merge (optional) | 785 | | | | |

| Phase | Predicted s | Measured s | Gap explained by |
|---|---:|---:|---|
| prefill | 1.5 | | |
| decode | 90.9 | | |
| ref | 4.9 | | |
| train | 14.5 | | |
