# GRPO + LoRA on Qwen3-4B under an 8 GB cap

**Verdict: DON'T SHIP the trained artifact** (the LoRA adapter and its merged model). The streamed engine
works: every phase stays under 8 GB on nvidia-smi, L2L matches a non-streamed reference, resume is exact. The
model does not. **Why it failed:** (0) training had almost no signal: 72 % of GRPO groups had zero reward
variance (~1 informative group per step), and the training reward never moved (0.40 → 0.41); (1) a control trained
on *shuffled* rewards gained as much on held-out data (+11.0 vs +11.4 pts, counting its correct `÷` answers), so
the gain is not shown to come from the reward; (2) the reward graded the *last* `<answer>` tag: 18 % of outputs
hedge with two answers, and graded on the first answer half the gain disappears; (3) the one reward-specific change
is a shortcut: the adapter dropped * and /, and × ÷ items regressed (0.129 → 0.036); (4) the score depends on how the model is run: merged bf16 0.502, adapter 0.430 in bf16 and 0.374 in fp32.

**Setup.** Vast.ai RTX 3070 (8 GB physical, so the cap is the hardware; CUDA sees 7840 MiB), PCIe 4.0 x16 (25.5 GB/s
measured), 31 GB RAM. Qwen3-4B non-thinking, bf16; LoRA r = 16 on all 7 linears × 36 layers (33 M params, fp32).
GRPO: 4 prompts × G = 8, 100 steps, lr 2e-5, β = 0, sequence-normalised loss. **Task: Countdown-3** (reach a target
with 3 numbers and + − × ÷; exact-fraction reward; 1500 train / 500 held-out, generated, disjoint), chosen on GPU
pilots (GSM8K and MATH rejected, §3): ~20-token answers (each decoded token costs a 128 ms pass over streamed
weights), base pass 0.34 at T = 1, but 66 % zero-variance groups, over our pre-registered 0.5 limit; we trained
anyway, and this became failure (0). **Stack: our own** (`task2_l2l/`): Soup refuses `stream_layers` with
rollout tasks (`config/schema.py`); `transformers` supplies only the Qwen3 layer classes and tokenizer.

## 1. Memory, L2L, streaming (Parts 1–2)

The budget (`task1_budget/`) was committed before any GPU run (assumed hardware, GSM8K: 20 resident); re-run before
training with constants measured on the 3070, it predicted 25 resident, 7603 MiB. At 25 the smoke run hit 7835 of
7840 MiB with allocator retries, so we ran **24 resident** + `expandable_segments` (0 retries in training and
eval). Predictions below are at 24.

| quantity | predicted | measured (main run) | why the gap |
|---|---|---|---|
| allocated peak | 7177 MiB | 7232 MiB | +0.8 %: `repeat_kv`, logits transients |
| nvidia-smi peak | 7410 MiB | **7793** (≤ 7839 any run) | cache (reserved 7542); context 253 vs 233 |
| decode per pass | 135 ms | **128 ms** | 96 ms PCIe (12 × 7.9); overhead overestimated |
| prefill / train fwd+bwd | 0.56 / 5.3 s | 1.26 / 9.1 s | **model error**: small GEMMs far below peak |
| step; generation share | 23.1 s; 76 % | 20.1 s mean; **55 %** | decode stops early (75.6/128); median 26.7 s |
| host RSS GiB train/merge | 10.05 / 10.55 | 5.27 / 8.44 | tensor-by-tensor reads; merge: 1 shard |

- **L2L** (`engine.py`): pinned host buffers; resident + 2 prefetch slots, copies on a separate stream
  ordered by events; boundary checkpoints; layer-by-layer backward by recompute; one layer loop for every pass. Copy hidden: train **97 %**, decode **35 %**. **Bottleneck: PCIe in decode (predicted).**
- **Equivalence** (`equivalence.py`, GPU): streamed logits **bit-identical** to resident (= HF); LoRA grads within
  1.5 × the autograd reference's own bf16 error vs fp32 (worst 1.26) on Qwen3-0.6B and 4B (4 layers), i.e. down to
  bf16 noise (~1.6 %). Strict checks (fp32 vs fp64, mixed residency, a swapped-layer control) are CPU tests.
- **Resume** (GPU, `runs/resume_demo`): SIGKILL at 3 points (after an update, mid-rollout, after a checkpoint):
  sample ledger = uninterrupted run, 0 duplicates, 0 gaps, identical tokens and rewards in all 10 steps.
- **H6 dynamic residency** (one A-B-B-A block, 12 paired steps, identical rollouts; rule registered for GSM8K):
  **+0.1 % [−0.4, +0.8]** < 2 % → keep regular; it added only 1 layer, at 7839 MiB with retries. H1 (generation
  ≥ 70 %) failed (55 %). Not run: H2–H5, H7 (time); H11–H13 (no DFlash).
- **Merge** (`merge.py`): tensor by tensor, 15 s, 661 MiB nvidia-smi; merged eval 6729 MiB. **DFlash: not done**
  (optional); predicted to pay only above τ = 2.06 (drafter + verify buffers displace ~10 layers).

## 2. Did the model actually train? (Part 3)

A rising reward proves nothing (GRPO's loss is ~0 by construction). `verify.py` compares arms on 500 held-out
items, greedy, paired bootstrap 95 % CIs: base, adapter, merged, a **shuffled-reward control** (rewards permuted
across the batch) and a **random-direction adapter** (same per-tensor ‖ΔW‖, post hoc).

| arm | pass | × ÷ items (140) | `÷ ×` accepted: all / × ÷ | pass on 1st `<answer>` | >1 `<answer>` | * or / in × ÷ answers | truncated · tokens |
|---|---|---|---|---|---|---|---|
| base | 0.318 | 0.129 | 0.318 / 0.129 | 0.316 | 1.6 % | 37 % | 2.0 % · 19 |
| adapter | 0.430 | 0.036 | 0.432 / 0.043 | 0.370 | 18.0 % | 13 % | 10.8 % · 34 |
| control | 0.404 | 0.021 | 0.428 / 0.107 | 0.404 | 0 % | 75 % (of 20) | 55.8 % · 119 |
| merged | 0.502 | 0.043 | 0.502 / 0.043 | 0.488 | 7.0 % | 14 % | 4.2 % · 23 |
| random | 0.320 | 0.129 | 0.320 / 0.129 | 0.316 | 1.4 % | 37 % | 1.6 % · 19 |

Pre-registered rules: **R1** adapter − base > 0: +0.112 [+0.060, +0.162], pass. **R2** adapter − control > 0:
+0.026 [−0.018, +0.070], **fail** (+0.004 [−0.042, +0.048] with `÷ ×` accepted). **R3** merged − adapter ∋ 0:
+0.072 [+0.040, +0.104], 65 % identical text, **fail** (passes in fp32, item 4). **R4** not format-only: pass but
vacuous (all arms always tag, so R4 = R1). Items 1–5 are post hoc:

1. **A reward-free run gains as much** (control − base +8.6 pts [+3.6, +13.4]; +11.0 [+6.0, +15.8] counting its
   12 correct `÷`/`×` answers, which the reward scores 0). With ratio 1 and no KL, shuffled advantages have zero
   expected gradient, yet length drifted 19 → 119 tokens: Adam turns noise into fixed-size steps (‖ΔW‖ 0.85 vs
   1.00), and longer reasoning happens to score here. Its batch-wide shuffle also added noise (14 % vs 72 %
   zero-variance groups). One seed; mechanism unproven.
2. **Half the gain is hedging**: an answer, "Note: this is wrong…", a second answer; the reward reads the last.
   On the first answer (what a user reads): adapter − base +0.054 [0.000, +0.108].
3. **A learned shortcut: no × ÷** (adapter − base −0.093 [−0.157, −0.029] on × ÷ items). The adapter nearly stopped
   writing * and / (37 % → 13 % of × ÷ answers), e.g. `46 - 30 / 2` → `46 - (30 - 2)`; the control did not regress
   (−0.021 [−0.086, +0.043] with `÷` accepted; adapter − control −0.064 [−0.114, −0.014]). Likely cause: × ÷ groups
   are rarely solved, so carry no gradient (inferred).
4. **Precision changes the policy.** The update is ~0.08 % of a weight (Frobenius), below bf16's 0.2–0.39 % step:
   82 % of updated elements round back to base. A random-direction adapter of equal size
   (isotropic) changes 6 % of outputs (ours 84 %), and its merge scores the same.
   **fp32 rerun** (`fp32_check.py`): merged ≡ adapter (−0.004 [−0.010, 0.000], 97.6 % identical), so `merge.py` is
   right; but the *same* adapter scores 0.374 in fp32 vs 0.430 in bf16 (−5.6 pts [−8.4, −3.0]; no fp32 base arm).
5. **The sampled policy did not improve.** Each of the 400 training prompts was seen once, so the training reward
   is a T = 1 score on unseen prompts: last 25 − first 25 steps −0.004 [−0.114, +0.104]. Only the greedy answer moved.

**Not detected:** contamination; scorer bias shared by all arms; non-Countdown regressions; effects below ~±5 pts;
held-out T = 1 behaviour; seed variance; style-specific scorer bias, multi-answer gaming (both found post hoc).

## 3. Silent failures (Part 4): each one finishes without an error

| stage | what breaks silently | how checked · evidence | Soup check catches it? |
|---|---|---|---|
| data | too easy (GSM8K 0.94); reward = length (MATH 41 % truncated) | GPU pilots | no (doctor: format) |
| template | thinking mode on; instruction ignored (answer-only GSM8K 88 % truncated) | `enable_thinking=False` asserted; pilots | no |
| reward | **last tag graded: hedging pays (18 %)**; too strict (`\$18`, `÷ ×` → 0) | first-tag, `÷ ×` rescoring; 33 tests | no |
| GRPO | 72 % zero-var groups; 50 % steps clipped | per-step log; control | no |
| policy | +/− gain hides a "no × ÷" shortcut | per-subset CIs | no: one aggregate |
| L2L | pinned memory 2× (power-of-2 rounding); unordered prefetch | exact-size `cudaHostRegister`; events | no |
| on-policy | rollout ≠ train log-prob (mean 0.008, max 1.6) | logged per step | no |
| streaming | duplicates if killed before 1st checkpoint; torn ledger line | tests (fixed); GPU kill demo | no |
| merge | **bf16 rounding swallows the delta** | merged arm; fp32 rerun | no: `AdapterChange` never decides |
| inference | **bf16 vs fp32 flips 36 % of outputs** | fp32 arms | no |

Soup's checks (data doctor, pre-flight, `soup ship`) were read in source, not run. `soup ship`: SHIP if tuned > base
and no general regression; the first leg passes (0.430 > 0.318), we had no general benchmark for the second. None
compares to a negative control, checks merged ≡ adapter, or rescores the answer a user reads.

## 4. Verdict: DON'T SHIP, and what must change first (in priority order)

1. **Signal:** keep only prompts with base pass in (0, 1) at k = 8 (22 of 64 pilot items); sweep lr (2e-5 left the
   update below bf16's step); ≥ 3 seeds; a T = 1 eval; a within-group shuffled control.
2. **Reward:** exactly one `<answer>` tag, else 0 (removes the hedging); accept `÷ ×`; gate on × ÷ separately.
   Token-level loss normalisation is a hypothesis to A/B.
3. **Artifact:** ship the fp32 LoRA adapter unmerged; gate on merged ≡ adapter at users' precision.
4. **Soup:** make the negative control, merged ≡ adapter and a CI lower bound > 0 part of `soup ship`.

**Surprises and concerns.** The shuffled control gave the largest behaviour change and the same aggregate gain;
without it this is a clean +11-pt success. Precision alone moves the score 6–13 pts. Concerns: one seed per arm;
greedy only; 128 tokens cut the control (0.428 is a lower bound); no general eval.

**AI tools.** Claude Code (Opus) wrote most code, ran the GPU jobs and drafted this report. Fresh-context reviews
of the engine (6 fixes: pinned memory 2×, a missing stream dependency, resume duplicates…) and of the report, then
sub-agent reviews checked on raw files (found the `÷ ×` scoring error behind our earlier, wrong claim that the
control also regressed at × ÷). Human decisions (dataset, rules,
24 resident, merge and fp32 checks) are in git. Evidence: `logs/`, nvidia-smi CSVs at 200 ms (all runs but the 8 s
random-adapter merge), `verify.py`, `equivalence.py`, `tests/`, `task2_l2l/results/`.
