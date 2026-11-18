# Part 1 budget: predicted_measured_hw

Cap 8192 MiB on nvidia-smi, 512 MiB reserve. Hardware: `{'ctx_mib': 233, 'h2d_gbps': 25.54, 'vram_gbps': 334.2, 'tflops': 42.29, 'overhead_ms': 0.655, 'host_base_gib': 3}`.

Streaming `regular`, DFlash off. Symbols: prompts=2, G=8, B=prompts·G=16 sequences, P=256, C=512, S=P+C=768, blk=1, τ=3.0, L=36, h=2560, kv=1024, V=151936, layer=192.5 MiB, pmb=2, mb=1, chunk=256.

## Weights (from safetensors headers)

|  | MiB |
|---|---|
| one decoder layer | 192.5 |
| 36 layers | 6,930.4 |
| embedding = tied lm_head | 741.9 |
| target total | 7,672.2 |
| DFlash drafter | 1,025.1 |
| largest shard | 3,802.7 |

## GPU memory per phase: **21 resident / 15 streamed**, binding phase **prefill**

| term | prefill | decode | ref | train | formula |
|---|---|---|---|---|---|
| CUDA context + workspaces | 233 | 233 | 233 | 233 | assumed / bench.py ctx |
| embedding = tied lm_head | 742 | 742 | 742 | 742 | V·h·2B |
| resident layers (21) | 4,043 | 4,043 | 4,043 | 4,043 | r · layer |
| prefetch buffers | 385 | 385 | 385 | 385 | 2 · layer |
| LoRA weights, fp32 | 126 | 126 | 126 | 126 | 33.0M · 4B |
| Adam m,v, fp32 | 252 | 252 | 252 | 252 | 33.0M · 8B |
| target KV cache | 1,728 | 1,728 |  |  | B·(S+blk)·2·L·kv·2B, full length |
| prefill activations, one layer, chunk | 44 |  |  |  | pmb·P·44544·2B |
| last-token logits | 14 |  |  |  | B·V·(2+4)B |
| decode logits |  | 14 |  |  | B·V·(2+4)B |
| boundary hidden in+out |  |  | 120 |  | 2·B·S·h·2B |
| activations, one layer, micro-batch |  |  | 65 |  | mb·S·44544·2B |
| logits chunk, no grad |  |  | 371 |  | chunk·V·(2+4+4)B |
| LoRA grads, fp32 |  |  |  | 126 | lora · 4B, set to None outside train |
| boundary checkpoints on GPU |  |  |  | 120 | 2·B·S·h·2B, rest on host |
| dL/dh in+out |  |  |  | 120 | 2·B·S·h·2B |
| recompute one layer, saved for backward |  |  |  | 98 | mb·S·67184·2B |
| logits chunk with grad |  |  |  | 519 | chunk·V·14B: bf16 logits, fp32 upcast, log-softmax, grad (Exact Layer Streaming §5.6) |
| **total** | **7,566** | **7,523** | **6,337** | **6,764** |  |

### merge GPU: 0.51 GiB

| term | MiB | formula |
|---|---|---|
| CUDA context | 233 | assumed |
| one tensor: bf16 in, fp32, fp32 delta, bf16 out | 285 | 9728×2560·(2+4+4+2)B |

### host, training: 11.70 GiB

| term | MiB | formula |
|---|---|---|
| Python + torch + CUDA host | 3,072 | assumed |
| pinned streamed layers | 2,888 | (L−r) · layer |
| pinned checkpoints | 2,220 | (L+1)·B·S·h·2B |
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
| prefill | 4096 | 1 | 21 | 2 | copy 7.9 / FLOPs 19.75 / VRAM 0.60 | FLOPs | 0.74 |
| decode | 16 | 512 | 21 | 2 | copy 7.9 / FLOPs 0.08 / VRAM 0.60 | PCIe | 80.46 |
| ref log-probs | 12288 | 1 | 21 | 2 | copy 7.9 / FLOPs 60.48 / VRAM 0.60 | FLOPs | 2.35 |
| train forward | 12288 | 1 | 21 | 2 | copy 7.9 / FLOPs 60.48 / VRAM 0.60 | FLOPs | 2.35 |
| train backward (recompute + dX) | 12288 | 1 | 21 | 4 | copy 7.9 / FLOPs 120.96 / VRAM 1.21 | FLOPs | 4.53 |

**Step 90.4 s, generation 90%, 45.2 s/prompt.** PCIe traffic 1455 GiB/step; one more resident layer saves 3.74 s/step. DFlash arm break-even τ = 1.54 (assumed τ = 3.0).

## Batch sweep (prompts per step, best residency for each)

| prompts | B | resident | peak MiB (binding) | step s | s/prompt | s/seq | steps/h |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 26 | 7,658 (prefill) | 66.8 | 66.8 | 8.35 | 54 |
| 2 | 16 | 21 | 7,566 (prefill) | 90.4 | 45.2 | 5.65 | 40 |
| 3 | 24 | 17 | 7,667 (prefill) | 110.3 | 36.8 | 4.60 | 33 |
| 4 | 32 | 12 | 7,575 (prefill) | 133.9 | 33.5 | 4.19 | 27 |

## Scenarios

|  | resident | peak MiB (binding) | fits | step s | gen s | train s | s/prompt |
|---|---|---|---|---|---|---|---|
| default | 21 | 7,566 (prefill) | yes | 90.4 | 81.2 | 9.2 | 45.2 |
| dynamic | 21 | 7,566 (prefill) | yes | 78.1 | 68.9 | 9.2 | 39.1 |
| dflash_tau3 | 13 | 7,652 (decode) | yes | 51.2 | 42.0 | 9.2 | 25.6 |
| dflash_tau2 | 13 | 7,652 (decode) | yes | 71.7 | 62.5 | 9.2 | 35.8 |
| pcie3_x16 | 21 | 7,566 (prefill) | yes | 158.9 | 149.7 | 9.2 | 79.5 |

## Residency sweep

| resident | peak MiB (binding) | fits | step s | gen s | train s | s/prompt |
|---|---|---|---|---|---|---|
| 0 | 3,523 (prefill) | yes | 168.9 | 159.7 | 9.2 | 84.5 |
| 10 | 5,448 (prefill) | yes | 131.5 | 122.3 | 9.2 | 65.8 |
| 21 | 7,566 (prefill) | yes | 90.4 | 81.2 | 9.2 | 45.2 |
| 22 | 7,759 (prefill) | no | 86.7 | 77.5 | 9.2 | 43.3 |
| 36 | 10,069 (prefill) | no | 34.4 | 25.1 | 9.2 | 17.2 |
