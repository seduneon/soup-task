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

**GSM8K result (2026-11-18, RTX 3070):** greedy held-out pass 0.94 (95 % CI 0.89–0.98), boxed 0.98, truncated
0.03, mean 251 tokens. Fails the 30–70 % rule by a wide margin, as the published prior predicted (Qwen3-4B-Base
GSM8K 87.8, 4-shot; Qwen3 report arXiv 2505.09388). Deviation from the rule as written: the `--k 8 --temperature 1`
GSM8K run was stopped early and not used. A sampled rate below 70 % is implausible with a greedy CI lower bound of 0.89.

**Fallback, added before the MATH 3–5 result was seen** (published prior: non-thinking MATH-500 84.8 at temp 0.7
and 32k tokens; ours is levels 3–5, temp 1, 512 tokens):
1. If MATH 3–5 overall passes the rule, use all of `data/math35`.
2. If its pass rate is above 70 %, break the same k=8 run down by level, using the `level` field (L3 439 / L4 453 /
   L5 608 train items). Use the widest level subset from the top that passes the rule: L4–5, then L5 alone.
   No new GPU run is needed to choose. Confirm the chosen subset with a fresh k=8 run on that subset only.
3. If its pass rate is below 30 %, apply the same procedure from the bottom: L3–4, then L3.
4. If the truncated rate is above 15 %, report it as a finding: those zeros are length, not wrong math.
   Raising `max_new` changes the Part 1 memory plan (KV cache length C), so it is a separate decision.

**MATH 3–5 result:** k=8 pass 0.59 (CI 0.49–0.69), zero-variance groups 0.53, **truncated 41 %**. Finished
answers were 95.7 % correct and truncated ones 5.3 %, so the reward measured "fits in 512 tokens", not maths.
Per level: L3 0.82 / L4 0.54 / L5 0.38, with L5 63 % truncated. The fallback above handles pass rate, not
length, so MATH was rejected.

**GSM8K answer-only** (`prompt_style: direct`, strict reward): max_new 32 → pass 0.078, 88 % truncated,
zero-var 0.86; max_new 64 → 0.111, 83 %, 0.91. Qwen3-4B non-thinking ignores "do not explain" and writes working.
Compliant answers were ~60 % correct, but too few groups had reward variance for GRPO.

**Chosen: Countdown-3** (`prepare_countdown.py`, `reward.countdown`, `tests/test_countdown.py`). A research pass
ranked 9 candidates; DAPO-Math-17k was rejected because it needs 1k+ token answers. Pilot, 64 prompts × k=8,
T=1, max_new 256: pass **0.34** (CI 0.24–0.44), truncated **0.6 %**, `<answer>` tag 100 %, length p50 17 /
p90 18 tokens, 0 allocator retries, zero-variance groups **0.66** (31 all-0, 11 all-1). Items that need * or /
pass 0.05 and +/- items pass 0.47. Dropping the * or / items only lowers zero-variance to 0.59.
**Deviation:** zero-variance is above the pre-registered 0.5. Adopted anyway, because 0.5 stood in for "enough
learning signal per step". Here that is restored directly: prompts per step go 2 → 4, which is cheap with
17-token answers, giving ~1.3 prompts with mixed rewards per step, ~5× answer-only GSM8K. The length confound
is gone: failures are wrong expressions, not truncation. Part 3 reports +/- items and * or / items separately.

**Smoke runs (3 steps, Countdown-3, B = 32):**
- **Budget at 25 resident:** predicted peak 7603 MiB. Measured 7835 MiB against the 7840 MiB CUDA ceiling (CUDA
  reports 7840 of the card's 8192), with 17 allocator retries/step: each decode token grew the KV cache by 2 MB,
  missed, flushed the cache and retried (4, 17, 3 retries in the three steps). That rollout took 17 s; the same
  step at 24 resident with 0 retries took as long, so the retries were a near-OOM warning, not a measured slowdown. The ledger
  under-counts by ~230 MiB, about the `repeat_kv` and logits transients noted above.
- **24 resident, A/B on the allocator:**
  - default: 2–6 retries per step, peak 7773 MiB;
  - `expandable_segments:True`: 0 retries, peak 7789 MiB.
  - Identical rewards and gradient norms, identical step times (29 / 28 / 12 s, mean 23 s vs 22.2 s predicted).
- `train.py` and `evaluate.py` now default to `expandable_segments`, and `attempts.jsonl` records it. Bench
  numbers were measured without it.
- Slow steps come from one rambling completion that hits max_new: the batch waits for its longest sequence.

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
