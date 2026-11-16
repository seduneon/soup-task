# 03 — Prior work: the paper, the Soup repo, internal docs, 10Cache

## A. Exact Layer Streaming paper (v3, 13 Aug 2026)

"Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB Laptop GPU" — Alpamys Makazhan (Soup author). Repo: github.com/MakazhanAlpamys/Soup.

### Claims
- Layer-streamed LoRA on a 4 GB RTX 3050 laptop: Llama-3.1-8B NF4 at 119.6 tok/s in 3.32 GB peak; Qwen2.5-3B bf16 base at 143.1 tok/s in 2.15 GB (OOMs when trained resident).
- Overhead vs resident 1.43× at 0.5B (the only valid resident baseline).
- Central contribution: a **correctness protocol gated on bit-exactness against a resident reference of the same numerics**.
- Reproduces on H100 at a median 113 tok/s → whatever bounds the step is common to both machines.

### Why v3 exists (the lesson)
v2 claimed "bound by host-to-device transfer" without measuring. v3 measured: step at 71.3% of the same-session GEMM ceiling; deleting every H2D byte → only 1.4% faster; compute stream blocked on a copy 0.20% of the step; largest streaming-specific cost = per-layer NF4 dequant 9.8%. **Never publish an inferred mechanism; ablate it.** Caveat they keep: below ~128 tokens/step transfers do matter.

### Method pieces to reuse (§3)
- Tiers: GPU = N layer buffers + embedding + final norm + adapters/grads/optimizer; CPU = all decoder layers, page-locked.
- Execution: wait prefetch(i) → launch prefetch(i+1) on dedicated stream → layer body in `checkpoint(use_reentrant=False)` → release i−1. Backward re-runs checkpoint and re-materialises weights in reverse.
- Primitives: `torch.utils.checkpoint`, `saved_tensors_hooks` (boundary activations to host), dedicated `torch.cuda.Stream` + pinned + `non_blocking`, `requires_grad=False` (not detach/no_grad), `torch.func.functional_call`.
- Buffer pool: allocated once; each slot has an owner field (tripwire for "autograd node holds a pooled tensor a later prefetch overwrites").

### The seven silent traps (§3.5) — starting list for part 4
1. PEFT `LoraModel.forward` calls `self.model.forward` directly → a pre-hook on the causal-LM wrapper never fires; attach to the module owning `.layers`.
2. PEFT inits adapters on the base layer's device (meta under streaming) → must re-materialise (A Kaiming, B zeros). B = 0 → "did it load?" checks pass vacuously.
3. load → `.to()` → `.pin_memory()` makes 3 transient copies; allocate pinned once and `copy_`.
4. `Trainer.__init__` calls `model.to(device)` → raises on meta params (the one that raises).
5. PEFT picks LoRA implementation from an `is_loaded_in_4bit` marker a meta skeleton lacks → silent fallback, 0.94 max logit divergence with byte-identical weights.
6. A wrapper changes saved parameter names → adapter loads as zeros with only a warning → merge/serve/chat return the untuned base while reporting success.
7. Inverse on load: 0 of 12 tensors land; resumed loss byte-identical to training from scratch.
(8th: wrapper must be attribute-transparent — transformers reads `decoder_layer.attention_type`.)

Lesson: traps 5–7 were found by the next release's correctness gate, not by tests. **Tests that matter compare against an independent reference.**

### Correctness protocol (§4)
SmolLM2-135M, streamed vs resident: logits bit-exact; layer-0 LoRA grad ≠ 0 and all layers get grad; 100-step loss curves identical; same seed twice identical; 2 vs 3 buffers identical; base weights asserted to stay on meta.
- **Reference must match numerics** (streamed NF4 vs resident NF4, never vs bf16 — quantisation error hides real bugs).
- **Vacuity defence:** randomise `lora_B` on the reference, copy adapters across, assert a non-zero count copied.
- Forward and backward are two separate claims — never collapse them.

### §4.3 — the bitsandbytes aliasing defect
At 32B/72B NF4: forward bit-exact, loss matched to every digit, **gradients wrong on 62/64 and 78/80 layers**. Cause: `bitsandbytes.MatMul4Bit` stashes the packed weight on `ctx` outside `save_for_backward`, so checkpointing can't recompute it; the reference aliases a pool slot a later prefetch refills. `cuda.synchronize()` doesn't fix it (not a race); cloning does (alias). Fix: dequantize + `F.linear`. A control that cannot fail is the tell for a measurement that cannot measure.

### Measurement findings (§5)
- Logits cost **14 B/element**, not 6.
- Windows/WDDM spills to host instead of OOM → "it didn't crash" ≠ "it fits". (Your version: allocator limit ≠ nvidia-smi cap.)
- Gradient accumulation is per-token I/O-neutral; real cost is opportunity cost vs batch size (2.52×). Guidance: raise batch until pre-flight refuses, then accumulate.
- Fraction-of-peak is meaningless unless throughput and ceiling are measured in the same session with SM clock recorded.
- Peak-VRAM predictor: peak = pool + extras + adapter×16 + 13.5 MB + batch×seq×(14·vocab + 2·n_layers·hidden + 4·(hidden+intermediate)); worst error 0.85%, never under-predicts.
- CUDA context + driver + display: stable 0.847 GB on the laptop (measured, not assumed).
- Quality (§5.9): paired over 5 disjoint subsets; within-arm spread is the noise floor; "no difference detected at ~1pp" ≠ "no difference". An n=3 pattern dissolved at n=5 — both readings kept. Scorer bug (json.loads over fenced output) made a 0.0 baseline — an instrument artifact.
- vs DeepSpeed ZeRO-3 offload: 2.88–2.93× at matched numerics, but on an 80 GB card resident training beats everything → streaming is for the card that can't hold the model.

## B. Soup public repo — what exists (checked Nov 2026)

- **GRPO + streaming is refused by design.** `src/soup_cli/config/schema.py`: `stream_layers` with a rollout task raises — generation re-reads every layer once per generated token, "a permanent exclusion rather than an unimplemented one."
- `utils/stream_inference.py`: streamed forward for prefill/scoring/eval only; per-token generation excluded; **no KV cache** (`use_cache=False`).
- `trainer/grpo.py`: wraps `trl.GRPOTrainer` → generation via `model.generate()`, can't use a custom streamed engine.
- No DFlash (only n-gram spec and a generic draft utility).
- Useful references: `utils/layer_stream.py` (estimators: logits bytes, activations, optimizer, retained layers), `utils/lowmem_merge.py` (tensor-wise merge), `trainer/rewards.py`, `commands/data_doctor.py`, `soup advise` (pre-flight), `commands/ship.py`.

**Implication:** write your own engine (PyTorch + transformers for layer classes/config/tokenizer). Own GRPO loop (not TRL). Install Soup alongside to run data doctor / pre-flight / `soup ship` for part 4. No fork needed. The task literally asks for what Soup excludes — made affordable by DFlash verifying a block per weight pass.

## C. Internal doc: dynamic streaming (D2, PR #5 in private soup-rnd)

- D2 picks, **once before training**, from **already-measured** variants: GPU-resident layers (`stream_resident_layers`), RAM hotset (`stream_ram_cache_mb`), activation retention for the last layers (`stream_recompute: auto`).
- Profile: `record` mode runs regular streaming + chosen variants (2 warm-up + 6 measured optimizer updates each), JSON keyed by model/weights index, data fingerprint, train config, GPU/driver, RAM limits, versions.
- `reuse` mode: estimate **T_run = T_setup − record-only probes + N × T_step**; pick minimum only if < 98% of regular streaming; re-probe memory with a fresh forward/backward at the real batch shape; any mismatch, stale (> 30 days) profile or resume → regular streaming.
- Example: regular 12 s setup + 9 s/step vs cache 30 s + 8.5 s/step → regular wins at 18 steps (174 vs 183 s), cache wins at 100 (880 vs 912 s).
- Scope: dense Transformers SFT, single CUDA device, fixed batch, disk source. **Incompatible with L2L**, per_expert, activation offload.
- Results worth remembering: 7B bf16 on 4090 (10 GiB soft cap) step −15.8 to −31.3% at batch 1; **physical RTX 3060 Ti 8 GiB, 8B bf16, 4 resident layers: only −2.39%**; 32B: caches sped up the step but lost on full short-run time; Qwen3-32B NF4 reuse picked regular streaming and won (−4.47%).
- For models that fit in RAM, keep weights in RAM. → For Qwen3-4B (6.8 GiB store, ≥16 GB host) the SSD tier and RAM hotset are moot; remaining knobs are GPU-resident layers and activation retention.
- Research gap for this task: D2 is measured on SFT. GRPO has a transfer-bound generation phase and a compute-bound training phase → **phase-aware placement** is a hypothesis worth testing.

## D. Internal doc: L2L context (soup-rnd main)

- L2L = load each layer once per optimizer step and run all k micro-batches through it (vs GA: k loads). Prior art: Pudipeddi et al. 2020 (arXiv 2002.05645), FlexGen 2023 (2303.06865) for inference. Don't claim "first".
- `training.l2l: true` requires `stream_layers: true`; k = `gradient_accumulation_steps` (ADR 0002). Incompatible configs are refused, never silently disabled.
- Equality with GA: SmolLM2-135M bit-exact at k=1,2,4 (CPU and RTX 3060), incl. dropout via RNG replay (A2). Mistral-7B NF4 bf16: tolerance rule **A3-07** — L2L-vs-GA error ≤ 1.5× GA-vs-GA error on another SDPA backend; fp32 ≤ 1e-4.
- Bit-exact conditions: one card, fixed SDPA backend (MATH), `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`.
- A bug found: backward called inside `autocast` in L2L but outside in GA → loss mismatch.
- Activation store: chunk batch × seq × hidden per layer per micro-batch; RAM budget = min(psutil available, cgroup remainder) → min(0.7×, −512 MiB); spill lowest layers to disk.
- Mistral-7B NF4 k=4: 244 → 64.3 layer loads per step; VRAM 1.15 → 1.13 GB; 191 MB pinned activations. Speed vs GA depends on host (ratio 0.79–2.6): fast H2D hosts make L2L slower, slow ones faster.
- Non-streamed resident training is faster than both when it fits → streaming is for when the model doesn't fit.
- Preference: DPO ready (reference pass in the same loaded layer → same loads as SFT); ORPO/SimPO/KTO not ready.
- Unsupported: model_type other than llama/mistral (and deepseek_v3 special path) — **qwen3 is not on the list**; fp16; left padding; >1 visible GPU.

### Their methodology rules (copy these)
- Decision rule committed **before** the first run; a rule written after the numbers doesn't count.
- An arm with any validity check `void` = "unknown", not "slow"/"worse"; unknown required check → "no verdict".
- No cross-host speed comparisons; label every result with card, CPU, H2D bandwidth, host load.
- A-B-B-A blocks on the same instance.
- 8 GB and 24 GB results labeled separately.
- Confirm Vast budget before renting; destroy instances and verify with `vastai show instances`.

## E. 10Cache (arXiv 2511.14124) — suggested basis for dynamic streaming

"10Cache: Heterogeneous Resource-Aware Tensor Caching and Migration for LLM Training" — Afroz, Khan, Albahar, Han, Butt (Nov 2025).
- Coordinates tensor placement across GPU, CPU and NVMe for training.
- Profiles tensor execution order → prefetch policies (move before needed).
- Pinned CPU buffers sized by the tensor-size distribution, reused to cut allocation overhead.
- Reports up to 2× training speedup, up to 86.6× GPU cache hit rate, 2.15× CPU and 1.33× GPU memory utilisation vs prior offloading.
- (Only the abstract was read so far — read the full PDF for the placement policy / cost model.)

## F. DFlash (z-lab)

- "DFlash: Block Diffusion for Flash Speculative Decoding" — Jian Chen, Yesheng Liang, Zhijian Liu. Project page: dflash z-lab. Drafter for Qwen3-4B: `z-lab/Qwen3-4B-DFlash-b16` (files: config.json, dflash.py, modeling_dflash.py, utils.py, model.safetensors 1.07 GB).
- Details of the algorithm and reference code: see `01-concepts.md` §5.
