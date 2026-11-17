# Part 1 budget: predicted

Cap 8192 MiB on nvidia-smi, 512 MiB reserve. Hardware: `{'ctx_mib': 500, 'h2d_gbps': 22, 'vram_gbps': 380, 'tflops': 20, 'overhead_ms': 0.5, 'host_base_gib': 3}`.

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

## GPU memory per phase: **20 resident / 16 streamed**, binding phase **prefill**

| term | prefill | decode | ref | train | formula |
|---|---|---|---|---|---|
| CUDA context + workspaces | 500 | 500 | 500 | 500 | assumed / bench.py ctx |
| embedding = tied lm_head | 742 | 742 | 742 | 742 | V·h·2B |
| resident layers (20) | 3,850 | 3,850 | 3,850 | 3,850 | r · layer |
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
| **total** | **7,641** | **7,597** | **6,411** | **6,839** |  |

### merge GPU: 0.77 GiB

| term | MiB | formula |
|---|---|---|
| CUDA context | 500 | assumed |
| one tensor: bf16 in, fp32, fp32 delta, bf16 out | 285 | 9728×2560·(2+4+4+2)B |

### host, training: 11.89 GiB

| term | MiB | formula |
|---|---|---|
| Python + torch + CUDA host | 3,072 | assumed |
| pinned streamed layers | 3,080 | (L−r) · layer |
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
| prefill | 4096 | 1 | 20 | 2 | copy 9.2 / FLOPs 41.77 / VRAM 0.53 | FLOPs | 1.52 |
| decode | 16 | 512 | 20 | 2 | copy 9.2 / FLOPs 0.17 / VRAM 0.53 | PCIe | 90.87 |
| ref log-probs | 12288 | 1 | 20 | 2 | copy 9.2 / FLOPs 127.89 / VRAM 0.53 | FLOPs | 4.94 |
| train forward | 12288 | 1 | 20 | 2 | copy 9.2 / FLOPs 127.89 / VRAM 0.53 | FLOPs | 4.94 |
| train backward (recompute + dX) | 12288 | 1 | 20 | 4 | copy 9.2 / FLOPs 255.78 / VRAM 1.06 | FLOPs | 9.54 |

**Step 111.8 s, generation 83%, 55.9 s/prompt.** PCIe traffic 1552 GiB/step; one more resident layer saves 4.43 s/step. DFlash arm break-even τ = 1.77 (assumed τ = 3.0).

## Batch sweep (prompts per step, best residency for each)

| prompts | B | resident | peak MiB (binding) | step s | s/prompt | s/seq | steps/h |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 24 | 7,540 (prefill) | 83.7 | 83.7 | 10.46 | 43 |
| 2 | 16 | 20 | 7,641 (prefill) | 111.8 | 55.9 | 6.99 | 32 |
| 3 | 24 | 15 | 7,549 (prefill) | 144.4 | 48.1 | 6.02 | 25 |
| 4 | 32 | 11 | 7,650 (prefill) | 172.5 | 43.1 | 5.39 | 21 |

## Scenarios

|  | resident | peak MiB (binding) | fits | step s | gen s | train s | s/prompt |
|---|---|---|---|---|---|---|---|
| default | 20 | 7,641 (prefill) | yes | 111.8 | 92.4 | 19.4 | 55.9 |
| dynamic | 20 | 7,641 (prefill) | yes | 99.0 | 79.5 | 19.4 | 49.5 |
| dflash_tau3 | 11 | 7,534 (decode) | yes | 74.5 | 55.1 | 19.4 | 37.3 |
| dflash_tau2 | 11 | 7,534 (decode) | yes | 101.1 | 81.7 | 19.4 | 50.6 |
| pcie3_x16 | 20 | 7,641 (prefill) | yes | 174.5 | 155.0 | 19.4 | 87.2 |

## Residency sweep

| resident | peak MiB (binding) | fits | step s | gen s | train s | s/prompt |
|---|---|---|---|---|---|---|
| 0 | 3,790 (prefill) | yes | 200.3 | 180.9 | 19.4 | 100.2 |
| 10 | 5,715 (prefill) | yes | 156.1 | 136.7 | 19.4 | 78.0 |
| 20 | 7,641 (prefill) | yes | 111.8 | 92.4 | 19.4 | 55.9 |
| 21 | 7,833 (prefill) | no | 107.4 | 88.0 | 19.4 | 53.7 |
| 36 | 10,336 (prefill) | no | 41.0 | 21.6 | 19.4 | 20.5 |
