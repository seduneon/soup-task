# Part 2 — L2L engine, streaming, GRPO, merge

Own layer streaming + L2L for Qwen3 with LoRA, a GRPO loop on top, prompts and rollouts streamed from disk
with deterministic resume, and a tensor-by-tensor merge. `transformers` is used only for the Qwen3 layer
classes, config and tokenizer; streaming, prefetch, checkpointing, the backward, the KV cache, decoding,
LoRA, GRPO and the merge are ours. Everything below was built and tested on CPU (and smoke-run on Apple MPS
as a second device) with tiny random Qwen3 models; the 4B runs and every timing number need the rented GPU.

## Files

| File | Role |
|---|---|
| `store.py` | Host weight store: one flat buffer per decoder layer, filled tensor by tensor from the shards with `copy_` (no load → `.to()` → `pin_memory()` triple copy). Only streamed layers are page-locked, at exact size (`cudaHostRegister`); resident layers drop their host copy. Asserts every tensor filled and every streamed layer really pinned |
| `lora.py` | LoRA A/B per (layer, target), kept **outside** the layer modules and bound to the slot before each visit (one streamed slot serves many layers). Delta in fp32 |
| `engine.py` | `StreamedQwen3`: resident slots + rotating prefetch slots, copies on a separate CUDA stream ordered by events. One layer loop (`run_layers`) shared by prefill, decode, ref log-probs, train forward; layer-by-layer backward from boundary checkpoints (pinned host on CUDA); chunked log-probs; `adapter_off()`; `set_resident()` (dynamic arm); `autograd_grads` = the non-L2L reference; `profile` + `overlap_report()` = CUDA-event copy time vs compute-stream stall per phase |
| `kvcache.py` | `StaticKV` (regular arm: full length up front) and `PagedKV` (dynamic arm: pages on demand), the duck-typed cache HF attention calls |
| `decode.py` | Rollouts: chunked prefill, one weight pass per generated token for the whole batch, greedy / seeded sampling, per-token log-probs; `memory_plan` = dynamic residency from free CUDA memory |
| `reward.py` | `math_boxed`: last balanced `\boxed{}`, conflicting boxes → 0, no box → 0, exact fraction comparison |
| `grpo.py` | Group advantages, completion-only targets, additive per-token loss (+ optional k3 KL), diagnostics |
| `data_stream.py` | `PromptStream` (byte-offset index over JSONL shards, seeded per-epoch order, cursor + dataset fingerprint), `RolloutStore`, atomic saves |
| `train.py` + `train_config.yaml` | The GRPO loop; checkpoint/resume; per-phase time and memory; copy/compute overlap per phase; `reward_shuffle` negative control; nvidia-smi sampler; raw logs |
| `merge.py` | Tensor-by-tensor, shard-by-shard merge; GPU/host peaks; how much of each delta survives bf16 |
| `evaluate.py` | One arm of `verify.py`: pass rate (greedy or sampled) for base / base + adapter / merged, bootstrap CI, format and truncation rates |
| `prepare_data.py` | GSM8K → `data/gsm8k/{train,heldout}`, MATH levels 3–5 → `data/math35/{train,heldout}` (numeric gold answers only, 20 % dropped; 7 train questions that also appear in MATH test excluded). Both sources drop items whose templated prompt exceeds `--max-prompt-tokens` (256 = P of the memory plan) before sampling: GSM8K loses none, math35 5.1 % of train / 3.9 % of test candidates (its longest problems: a bias to report). Fails on leakage or unparseable gold answers. `train.py` (`max_prompt`) and `evaluate.py` (`--max-prompt`) abort on a longer prompt instead of breaking the cap |
| `verify.py` | Part 3: base / adapter / merged / shuffled-reward control on the same held-out items, paired bootstrap CIs, decision rule R1–R4 fixed in the docstring before any run, "what this does not detect" |
| `equivalence.py` | L2L equivalence on real weights (or a local model dir) → `results/equivalence_*.{json,md}` |
| `monitor.py` | Phase timer (CUDA-synced, peak counters reset per phase, device-wide used vs cap → `over_cap` / `strict_cap`), nvidia-smi samplers: device-wide CSV + per-process `.apps.csv` |
| `ballast.py` | Caps a >8 GB card physically (a second process holds total − cap); evidence is our PID's line in `.apps.csv` |

## How a GRPO step runs

| Phase | Weight passes | What happens |
|---|---|---|
| rollout: prefill | 1 | all prompt columns, `prefill_mb` sequences per layer visit, KV written per micro-batch |
| rollout: decode | one per token | all B sequences share each pass; new token attends to every real column |
| (rollout → disk → back) | — | the batch is written to `rollouts/step_N.pt` and the training passes read it back |
| ref (only if β > 0) | 1 | same engine, `adapter_off()`, no grad, chunked log-probs |
| train forward | 1 | no grad, boundary checkpoints saved (pinned host on CUDA) |
| train head | — | loss tokens only, in chunks of `chunk`, each its own small graph |
| train backward | 1 | layers in reverse: reload, recompute with grad from the checkpoint, backprop |

One update per rollout batch, so ratio = 1 and the loss is `Σ −c·A·logp (+ β·c·k3)` with fixed per-token
weights, which is what makes the chunked head exact. Every pass goes through `visit`, so copies and passes
are counted the same way everywhere (logged per step as `layer_copies`, `weight_passes`).

## Tests (CPU, `python3 -m pytest -q tests`, 87 passed in ~75 s)

| File | Proves | Control that could fail |
|---|---|---|
| `test_equivalence.py` (26) | streamed logits `torch.equal` to resident; streamed LoRA grads within 1.5 × the reference's own error vs fp64 (fp32, bf16, padded, micro-batches); one copy per streamed layer per pass; reference == HF logits; offloaded checkpoints through the host arena bit-identical, arena reused when the sequence length changes | swapped layers (error 1.13 vs 5e-7) and a dropped adapter both fail |
| `test_decode.py` (10) | greedy == HF `generate` (with left padding); tokens invariant to residency, buffers, prefill micro-batches; decode-time logp == training rescoring; seeded sampling; **dynamic arm (paged KV + residency changing every pass) token-identical to regular** | the dynamic test asserts residency actually changed |
| `test_grpo.py` (37) | reward parser on 32 cases (adversarial + GSM8K formats: `\$18`, `25\%`, `18 \text{ dollars}`, `1{,}000`, `x = 5`; MATH formats `\frac12`, `-\frac{1}{14}` with a sign control); advantages; completion-only targets; loss gradient = analytic formula; **one real update raises logp of the positive-advantage completion and lowers the negative one** | flipped sign / dropped advantage fail the sign test |
| `test_resume.py` (5) | real `train.py`, SIGKILLed at 3 points (after the optimizer step, after rollout, after the checkpoint), resumed: same sample ledger, no duplicates, no gaps across an epoch boundary, **identical rollouts and bit-identical final LoRA** vs an uninterrupted run, for `save_every` 1 and 2; a kill before the first checkpoint leaves no duplicate; dataset change refused; `reward_shuffle` trains on a seeded permutation of the true rewards and still resumes bit-identically | asserts LoRA actually trained; asserts some step was really shuffled |
| `test_merge.py` (2) | merged (adapter off) == base + adapter, HF loads it, merged ≠ base; bf16 swallows a tiny delta and the report shows it | — |
| `test_verify.py` (7) | each of R1–R4 fails on synthetic arms built to fail it (no effect, control gains too, merge lost the adapter, format-only gain); mismatched items/settings refused; real `verify.py` → `evaluate.py` × 4 → merge end to end: fp32 merged text identical to adapter, adapter ≠ base, arms reused on rerun | every rule has a failing case |

Equivalence rule: `err(x) = ‖x − truth‖/‖truth‖` against an fp64 truth (fp32 on GPU); pass if
`err(stream) ≤ 1.5 · err(reference)` per tensor and in aggregate. Streaming may not add more than half again
of the rounding error the reference already has. On CPU the result is bit-exact, so the margin is unused;
on the GPU it will be needed (cuBLAS picks kernels by shape). Decode tolerates only what changes GEMM shapes
(prefill micro-batches: tokens identical, logp within 5e-7).

## Findings so far (for the Part 1 ledger and Part 4)

- **Pinned host memory would have been ~2× the ledger** (found in review). torch's `pin_memory=True` rounds
  every block up to a power of two and caches it forever: 36 layers × 256 MiB = 9.2 GB pinned for a ledger
  term of 3.1 GB, plus 2.3–3.4 GB of per-shape checkpoint buffers → ~16 GB on a 16 GB host. Fixed: exact-size
  `cudaHostRegister`, only streamed layers pinned, resident host copies and the host embedding dropped, one
  checkpoint arena with fixed offsets.

- **`repeat_kv` transient.** HF's SDPA path expands K/V to all 32 query heads for the current layer at every
  call. At B = 16, full length, that is ≈ 200 MiB of transient during decode that the Part 1 ledger does not
  count (the 512 MiB reserve has to absorb it). `torch.nn.functional.scaled_dot_product_attention(enable_gqa=True)`
  would avoid it; kept as is until measured.
- **Paged KV concat transient** (dynamic arm): each layer call concatenates its pages (≈ 2 × 51 MiB at full
  length for 4B). `headroom_mib` in the config covers it.
- **Device bug caught before the GPU:** the training-time logp stash was a CPU tensor indexed with device
  indices; it only fails off-CPU. Found by review, confirmed fixed by running the full loop on MPS.
- **bf16 merge rounding:** a delta below half a bf16 step of W vanishes in the merge; `merge.py` reports
  `changed_frac` and `delta_kept` per tensor so Part 3 can tell "merged == adapter" from "merged == base".

## GPU runbook (in this order; each step gates the next)

Prefer an 8 GB card. On a 12 GB+ card start `python3 ballast.py --cap-mib 8192 &` first and pass the printed
`cap_offset_mib`; the cap evidence is then our PID's `used_memory` in the `.apps.csv` log.

```bash
pip install hf_xet                                   # fast HF downloads
python3 -m pytest -q tests                           # same 60 tests on the box (CPU), sanity
python3 ../task1_budget/bench.py                     # measured hardware -> results/measured_hw.yaml
python3 equivalence.py --model Qwen/Qwen3-0.6B --layers 28 --device cuda --dtype bfloat16   # first test of the CUDA stream/event/pinned paths
python3 equivalence.py --model Qwen/Qwen3-4B --layers 4 --device cuda --dtype bfloat16 --resident 0
python3 evaluate.py --model Qwen/Qwen3-4B --data data/gsm8k/heldout --n 100 --out runs/base/gsm8k_greedy
python3 evaluate.py --model Qwen/Qwen3-4B --data data/gsm8k/train --n 64 --k 8 --temperature 1 --out runs/base/gsm8k_k8
python3 evaluate.py --model Qwen/Qwen3-4B --data data/math35/train --n 64 --k 8 --temperature 1 --out runs/base/math35_k8   # dataset rule below
python3 train.py --set steps=3 out_dir=runs/smoke    # phase times + memory vs the Part 1 ledger (H1, H10)
python3 train.py --set steps=2 streaming=dynamic out_dir=runs/smoke_dyn   # first run of memory_plan
python3 train.py                                     # main run (regular streaming)
python3 train.py --set streaming=dynamic out_dir=runs/dynamic steps=12   # H6 A/B blocks
python3 train.py --set reward_shuffle=true out_dir=runs/control          # Part 3 negative control, same steps
python3 merge.py --base Qwen/Qwen3-4B --adapter runs/main/adapter.pt --out runs/main/merged
python3 verify.py --model Qwen/Qwen3-4B --adapter runs/main/adapter.pt --merged runs/main/merged \
    --control runs/control/adapter.pt --data data/<chosen>/heldout --n 500 --out runs/verify
```
(set `data_dir` in `train_config.yaml` to the chosen dataset before the main and control runs)

Expected on the GPU, not bugs: kill/resume gives identical rollouts, but the final LoRA may differ in the
last bits (the memory-efficient SDPA backward is not guaranteed deterministic); and with ~400 LoRA tensors in
bf16 a few per-tensor ratios may exceed 1.5 while the aggregate passes. Look at `worst_tensors` first.

Still to do on the GPU: read the overlap evidence (`overlap` in `train_log.jsonl`: `copy_ms`, `stall_ms`, `h2d_gbps`),
the residency sweep (H2/H3), regular vs dynamic A/B (H6), measured merge peak, DFlash (bonus).

## Open decision: dataset

GSM8K and MATH levels 3–5 are both prepared; Qwen3-4B non-thinking likely solves most of GSM8K. Rule (pre-registered): use it
only if the sampled base pass rate on train (`--k 8 --temperature 1`) is between 30 % and 70 % and fewer than
half the groups have zero reward variance; otherwise use `data/math35` (same rule applies to it).

## Independent review (second Opus 5.5 instance, fresh context) and what was done

| # | Finding | Status |
|---|---|---|
| 1 | Pinned host memory ~2× the ledger (power-of-two rounding, all 36 layers pinned, per-shape checkpoint buffers) → OOM risk on 16 GB | fixed (exact-size registration, streamed layers only, release, arena) + test |
| 2 | `data_dir` default pointed at a missing directory, found only after the full model load | fixed (path, stream built before the model) |
| 3 | Reward gave 0 to `\$18`, `25\%`, `18 \text{ dollars}`, `1{,}000`, `x = 5` | fixed + 11 test cases incl. 2 controls |
| 4 | Dynamic plan ignored the CUDA context / allocator cache; nothing enforced the cap | fixed (`mem_get_info` accounting, `over_cap`/`strict_cap`, per-process nvidia-smi log, `ballast.py`, eval/merge sampled) |
| 5 | `visit()` did not order a pass's first copies after the previous pass's compute (masked by incidental syncs) | fixed (free events persist across passes) |
| 6 | Ledger duplicate when killed before the first checkpoint | fixed + test |
| — | Dynamic arm left extra layers resident into ref/train (confounds H6) | fixed (reset after rollout) |
| — | Nits: `max_new = 0`, non-atomic ledger rewrite, tied head chosen by key presence, "mid-train" wording | fixed |
| — | Backward re-copies the last layer the forward just used (2 of 16 copies) | open (perf; measure first) |
| — | No CPU test reaches the CUDA stream / event / pinned code | open by nature: first GPU step is `equivalence.py --resident 0` |
