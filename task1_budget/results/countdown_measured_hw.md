# Part 1 budget: countdown_measured_hw

Cap 8192 MiB on nvidia-smi, 512 MiB reserve. Hardware: `{'ctx_mib': 233, 'h2d_gbps': 25.54, 'vram_gbps': 334.2, 'tflops': 42.29, 'overhead_ms': 0.655, 'host_base_gib': 3}`.

Streaming `regular`, DFlash off. Symbols: prompts=4, G=8, B=prompts·G=32 sequences, P=96, C=128, S=P+C=224, blk=1, τ=3.0, L=36, h=2560, kv=1024, V=151936, layer=192.5 MiB, pmb=2, mb=1, chunk=256.

## Weights (from safetensors headers)

|  | MiB |
|---|---|
| one decoder layer | 192.5 |
| 36 layers | 6,930.4 |
| embedding = tied lm_head | 741.9 |
| target total | 7,672.2 |
| DFlash drafter | 1,025.1 |
| largest shard | 3,802.7 |

## GPU memory per phase: **25 resident / 11 streamed**, binding phase **prefill**

| term | prefill | decode | ref | train | formula |
|---|---|---|---|---|---|
| CUDA context + workspaces | 233 | 233 | 233 | 233 | assumed / bench.py ctx |
| embedding = tied lm_head | 742 | 742 | 742 | 742 | V·h·2B |
| resident layers (25) | 4,813 | 4,813 | 4,813 | 4,813 | r · layer |
| prefetch buffers | 385 | 385 | 385 | 385 | 2 · layer |
| LoRA weights, fp32 | 126 | 126 | 126 | 126 | 33.0M · 4B |
| Adam m,v, fp32 | 252 | 252 | 252 | 252 | 33.0M · 8B |
| target KV cache | 1,008 | 1,008 |  |  | B·(S+blk)·2·L·kv·2B, full length |
| prefill activations, one layer, chunk | 16 |  |  |  | pmb·P·44544·2B |
| last-token logits | 28 |  |  |  | B·V·(2+4)B |
| decode logits |  | 28 |  |  | B·V·(2+4)B |
| boundary hidden in+out |  |  | 70 |  | 2·B·S·h·2B |
| activations, one layer, micro-batch |  |  | 19 |  | mb·S·44544·2B |
| logits chunk, no grad |  |  | 371 |  | chunk·V·(2+4+4)B |
| LoRA grads, fp32 |  |  |  | 126 | lora · 4B, set to None outside train |
| boundary checkpoints on GPU |  |  |  | 70 | 2·B·S·h·2B, rest on host |
| dL/dh in+out |  |  |  | 70 | 2·B·S·h·2B |
| recompute one layer, saved for backward |  |  |  | 29 | mb·S·67184·2B |
| logits chunk with grad |  |  |  | 519 | chunk·V·14B: bf16 logits, fp32 upcast, log-softmax, grad (Exact Layer Streaming §5.6) |
| **total** | **7,603** | **7,586** | **7,011** | **7,365** |  |

### merge GPU: 0.51 GiB

| term | MiB | formula |
|---|---|---|
| CUDA context | 233 | assumed |
| one tensor: bf16 in, fp32, fp32 delta, bf16 out | 285 | 9728×2560·(2+4+4+2)B |

### host, training: 10.05 GiB

| term | MiB | formula |
|---|---|---|
| Python + torch + CUDA host | 3,072 | assumed |
| pinned streamed layers | 2,118 | (L−r) · layer |
| pinned checkpoints | 1,295 | (L+1)·B·S·h·2B |
| one shard while loading | 3,803 | largest safetensors shard |

### host, merge: 10.55 GiB

| term | MiB | formula |
|---|---|---|
| Python + torch | 3,072 | assumed |
| input shard (mmap) + output shard buffer | 7,605 | 2 · largest shard |
| LoRA fp32 | 126 | lora · 4B |

## Step time

| phase | tokens/pass | passes | resident | FLOP/param/tok | per layer ms | bound | s |
|---|---|---|---|---|---|---|---|
| prefill | 3072 | 1 | 25 | 2 | copy 7.9 / FLOPs 14.72 / VRAM 0.60 | FLOPs | 0.56 |
| decode | 32 | 128 | 25 | 2 | copy 7.9 / FLOPs 0.15 / VRAM 0.60 | PCIe | 16.38 |
| ref log-probs | 7168 | 1 | 25 | 2 | copy 7.9 / FLOPs 34.53 / VRAM 0.60 | FLOPs | 1.34 |
| train forward | 7168 | 1 | 25 | 2 | copy 7.9 / FLOPs 34.53 / VRAM 0.60 | FLOPs | 1.34 |
| train backward (recompute + dX) | 7168 | 1 | 25 | 4 | copy 7.9 / FLOPs 69.05 / VRAM 1.21 | FLOPs | 2.58 |

**Step 22.2 s, generation 76%, 5.6 s/prompt.** PCIe traffic 273 GiB/step; one more resident layer saves 0.93 s/step. DFlash arm break-even τ = 2.06 (assumed τ = 3.0).

## Batch sweep (prompts per step, best residency for each)

| prompts | B | resident | peak MiB (binding) | step s | s/prompt | s/seq | steps/h |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 27 | 7,645 (train) | 16.1 | 16.1 | 2.01 | 224 |
| 2 | 16 | 27 | 7,680 (train) | 17.5 | 8.7 | 1.09 | 206 |
| 3 | 24 | 26 | 7,536 (prefill) | 19.8 | 6.6 | 0.83 | 181 |
| 4 | 32 | 25 | 7,603 (prefill) | 22.2 | 5.6 | 0.69 | 162 |

## Scenarios

|  | resident | peak MiB (binding) | fits | step s | gen s | train s | s/prompt |
|---|---|---|---|---|---|---|---|
| default | 25 | 7,603 (prefill) | yes | 22.2 | 16.9 | 5.3 | 5.6 |
| dynamic | 25 | 7,603 (prefill) | yes | 20.8 | 15.6 | 5.3 | 5.2 |
| dflash_tau3 | 15 | 7,635 (decode) | yes | 17.1 | 11.8 | 5.3 | 4.3 |
| dflash_tau2 | 15 | 7,635 (decode) | yes | 22.6 | 17.3 | 5.3 | 5.6 |
| pcie3_x16 | 25 | 7,603 (prefill) | yes | 34.8 | 29.5 | 5.3 | 8.7 |

## Residency sweep

| resident | peak MiB (binding) | fits | step s | gen s | train s | s/prompt |
|---|---|---|---|---|---|---|
| 0 | 2,790 (prefill) | yes | 45.6 | 40.3 | 5.3 | 11.4 |
| 12 | 5,100 (prefill) | yes | 34.3 | 29.1 | 5.3 | 8.6 |
| 25 | 7,603 (prefill) | yes | 22.2 | 16.9 | 5.3 | 5.6 |
| 26 | 7,795 (prefill) | no | 21.3 | 16.0 | 5.3 | 5.3 |
| 36 | 9,335 (prefill) | no | 11.9 | 6.7 | 5.3 | 3.0 |
