# Walkthrough: what was done, in order, and why

A study guide for the live discussion. It tells the project as a story: each phase says **what** we did, **why**,
**how it was checked** and **what we learned**. Numbers come from `task2_l2l/results/analysis.md` and the logs.
The report (`REPORT.md`) is the 3-page summary. `docs/APPENDIX.md` is the long version.

---

## 0. The problem in one picture

- Qwen3-4B in bf16 is ~8 GB of weights. The card has 8 GB (CUDA actually sees 7840 MiB), and we also need room
  for the CUDA context, activations, the KV cache, LoRA and the optimizer. **The model cannot sit on the GPU.**
- So the weights live in CPU RAM and are **streamed** to the GPU layer by layer, every time they are used. Every
  forward pass, every generated token and every backward pass copies the streamed layers over PCIe again.
- On top of that we run **GRPO**: for each prompt, sample G = 8 answers, score them with an automatic reward, and
  push up the answers that scored above their group's mean. The adapter is **LoRA** (small trainable matrices
  next to the frozen weights).
- The real question is not "can we train" but **"should this artifact be shipped to users?"** The task warns
  that a rising reward and a finished run prove nothing.

Grading weights: proving the model is real 25 %, silent failures 20 %, engineering (memory, L2L, streaming,
DFlash) 30 %, live discussion 25 %.

---

## Phase 1: predict before measuring (Part 1, 11-16 → 11-17)

**What.** `task1_budget/budget.py` is a ledger of every memory term (weights per layer, prefetch buffers,
activation checkpoints, logits with the 151k vocabulary, KV cache, LoRA + Adam, reference log-probs, drafter,
merge peak) and a time model (copy time = layer bytes / PCIe bandwidth; compute = FLOPs / GEMM rate).

**Why first.** The task says "before you run anything". A prediction made after seeing numbers can't be wrong,
so it proves nothing. Committing it first makes it testable.

**Decisions it produced.**
- How many layers stay resident: the most that fit (25 predicted), because each resident layer removes one PCIe
  copy from *every* decode pass.
- **Pre-registered hypotheses** (H1–H13, `docs/04-plan-and-hypotheses.md`), each with a decision rule fixed in
  advance. Examples: H1 "generation dominates the step", H2 "decode is PCIe-bound", H8 "the adapter really
  learned". Methodology rule: *every positive needs a control that could have failed.*
- Streaming type: regular streaming as the baseline. KV-aware *dynamic* residency (keep extra layers resident
  while the KV cache is still small) is the research arm, adopted only if it wins ≥ 2 % in an A/B.
- DFlash pays only if it accepts more than τ ≈ 1.77–2.06 tokens per pass, because its 1 GB drafter pushes
  ~10 resident layers off the GPU.

**Learned later (phase 6):** the prediction was right where the physics is simple (decode is pure transfer:
128 ms predicted = 128 ms measured) and wrong where it assumed peak GEMM speed (training 5.3 s predicted vs
9.1 s measured, because the micro-batches are small).

---

## Phase 2: build the engine and prove it is correct (Part 2a, 11-17)

**What.** Our own L2L engine (`engine.py`, `store.py`, `decode.py`, `lora.py`):
- each layer's weights sit in one flat CPU buffer, **pinned** (page-locked) so the copy can run asynchronously;
- a few layers are resident; the others rotate through 2 GPU slots: while layer k computes, layer k+1 is copied
  on a **separate CUDA stream**, ordered with events;
- training saves the input of each layer (the **checkpoint at the layer boundary**), then runs backward layer by
  layer, recomputing each layer from its checkpoint;
- one layer loop serves prefill, decoding (with a KV cache), scoring and training, so generation and eval are
  streamed too (the task requires that).

**Why our own code.** The task requires it ("do not just import a library"), and GRPO rollouts must come from
the same engine that trains, or they aren't on-policy.

**How correctness was checked** (this is "Prove equivalence" in the task):
| check | what it proves | result |
|---|---|---|
| streamed vs resident logits | streaming does not change the forward pass | **bit-identical** on GPU |
| LoRA gradients vs a non-streamed autograd reference, both against an fp32 "truth" | streaming does not change the backward pass | err(stream) ≤ 1.5 × err(reference); worst ratio 1.26 |
| the same on Qwen3-0.6B (all 28 layers streamed) and Qwen3-4B (4 layers) | holds on the real architecture | pass |
| greedy decode vs Hugging Face `generate` | our decoder is a correct decoder | token-identical (CPU tests) |
| **negative control:** swap two layers | the test can actually fail | error 1.13 vs 5e-7: fails as it should |

*Why that tolerance:* bf16 has rounding error no matter what, so "equal" can't mean bit-equal for gradients.
We allow streaming at most 1.5× the error the non-streamed bf16 reference itself has against fp32. Streaming
may add half again of the unavoidable error, no more.

**Independent review.** A second fresh-context AI instance reviewed the engine and found 6 issues, all fixed.
The two that matter most for the discussion:
- pinned memory took **2× the ledger**: PyTorch's pinned allocator rounds every block up to a power of two. Fixed
  by registering exact-size buffers with `cudaHostRegister`;
- a prefetch copy was not ordered after the previous pass's compute. **It still gave correct results**, because
  other syncs happened to hide it. It's a classic silent failure; fixed with persistent events.

---

## Phase 3: streaming data with deterministic resume (Part 2b, 11-17)

**What.** `data_stream.py`: prompts are read from JSONL shards on disk through a byte-offset index, with a
seeded order per epoch and a cursor saved in the checkpoint. Each step's rollouts are written to disk and read
back for training. Checkpoints are written atomically.

**Why.** The task asks for exactly this: kill mid-epoch, resume, same samples, no duplicates, no gaps.

**How checked.**
- CPU tests: SIGKILL at 3 points (after the optimizer update, mid-rollout, after the checkpoint), across an
  epoch boundary. The final LoRA is **bit-identical** to an uninterrupted run.
- GPU demo (`runs/resume_demo`): same 3 kill points, 4 attempts. Sample ledger identical to the main run,
  0 duplicates, 0 gaps, **identical rollout tokens and rewards for all 10 steps**.
- Bug found by a test: a kill *before the first checkpoint* duplicated samples. Fixed.

---

## Phase 4: choose a task the reward can actually measure (11-18 morning)

The reward has to be verifiable and able to move within a few dozen steps. Each candidate was tested on the
GPU **before** training, against a rule written in advance (base pass rate in a useful range; at most half of
the groups with zero reward variance).

| candidate | what the pilot showed | decision |
|---|---|---|
| GSM8K (math word problems) | greedy pass **0.94**: almost every group of 8 is all-correct, so there's no gradient | rejected: too easy |
| MATH levels 3–5 | **41 % truncated** at 512 tokens; finished answers were 96 % right | rejected: the reward measured "fits in 512 tokens", not maths |
| GSM8K answer-only | the model ignores "don't explain": 88 % truncated | rejected |
| DAPO-Math | needs 1000+ token answers; each token = one 128 ms pass | rejected on cost |
| **Countdown-3** (reach a target with 3 numbers and + − × ÷) | pass **0.34** at T = 1, 0.6 % truncated, ~17-token answers, exact reward | **chosen** |

**Deviation, stated openly:** 66 % of Countdown groups had zero variance, above the pre-registered 50 %. We
adopted it anyway and doubled the prompts per step (2 → 4) to get the learning signal back. This matters later.

**Learned.** Picking the task *is* a silent-failure risk: a reward can look fine and measure the wrong thing
(the MATH truncation case).

---

## Phase 5: smoke runs, first real memory numbers (11-18)

- At 25 resident layers: **7835 of 7840 MiB**, with 4, 17 and 3 allocator retries in the three steps. The
  allocator ran out, flushed its cache and retried, **without any error**. That's another silent failure.
  (An earlier draft said "rollout 5.7× slower". That compared two different batches; the same step at
  24 resident with 0 retries took as long. Retries are a near-OOM warning, not a measured slowdown here.)
- A/B at 24 resident: default allocator 2–6 retries/step vs `expandable_segments:True` **0 retries**, identical
  rewards and gradient norms. Adopted 24 resident + expandable_segments.
- **Learned:** "it fits" is not enough. You also have to check `torch.cuda.memory_stats` for retries, because
  nvidia-smi doesn't show them.

---

## Phase 6: the runs (11-18)

| run | purpose |
|---|---|
| **main** | GRPO, 4 prompts × 8, 100 steps, lr 2e-5 |
| **shuffled-reward control** | identical, except the rewards are permuted across the batch before advantages are computed. Same pipeline, same compute, no information about which answer was right |
| **random-direction adapter** | not trained: a LoRA with the **same size per tensor** as the trained one but a random direction (`random_adapter.py`). Tests "does any change this big move the score?" |
| **merge** | adapter folded into the bf16 weights, tensor by tensor; GPU peak 661 MiB on nvidia-smi |
| **verify** | every arm on 500 held-out items, greedy, 128 tokens |
| **resume demo** | phase 3 |

**Why each control exists:**
- *Shuffled control:* GRPO can change a model even when the reward is meaningless (noise, length effects, the
  optimizer itself). If the control gains as much, the gain is **not evidence of learning from the reward**.
  (lr = 0 would be useless here: LoRA B starts at 0, so an lr = 0 adapter is exactly the base model.)
- *Random direction:* separates "the trained direction matters" from "any perturbation of this size changes
  outputs or rounds differently".

**Memory and time against the prediction (H1, H2, H10):**
- nvidia-smi peak **7793 MiB** (≤ 7839 in every run), sampled every 200 ms during all runs: **the cap holds**.
- decode 128 ms per pass, predicted 128 ms. Of that, ~96 ms is the 12 layer copies over PCIe (only 35 % hidden
  behind compute); the rest is resident-layer compute and launch overhead. Decode is mostly PCIe-bound (H2), but
  the H6 run shows that removing one copy did not shorten the pass, so it is not *purely* copy-bound.
- generation is 55 % of the step, not the predicted 76 %. H1's rule (≥ 70 %) **fails**. Why: generation stops
  when the longest answer ends (mean 75.6 passes, not 128), and training is slower than predicted.
- overlap: training hides 97 % of the copy time behind compute; decode only 35 % (1 ms of compute per 7.9 ms of
  copy). **The bottleneck is PCIe during decode.**

---

## Phase 7: did it actually train? (Part 3)

Rules fixed in `verify.py` **before** any run:

| rule | meaning | result |
|---|---|---|
| R1 | adapter beats base (CI lower bound > 0) | +0.112 [+0.060, +0.162], **pass** |
| R2 | adapter beats the shuffled control | +0.026 [−0.018, +0.070], **fail** |
| R3 | merged equals adapter (CI contains 0) | +0.072 [+0.040, +0.104], **fail** |
| R4 | the gain is not just formatting | every arm uses the answer tag, **pass** |

All comparisons are **paired** (same item, two arms) with bootstrap 95 % CIs. At n = 500 the resolution is
about ±5 points.

What we found by digging (these analyses were **added after reading the outputs**, so they're exploratory, not
pre-registered; say so if asked):
1. **The control gained +8.6 of the +11.2 points.** It drifted to long step-by-step reasoning (19 → 119 tokens).
   *Mechanism (corrected after the review):* an earlier draft blamed the Dr. GRPO length bias. That's wrong
   here. With shuffled rewards the advantage is independent of the completion, so the **expected gradient is
   zero** under either normalisation (the length bias needs advantage and length to be correlated). Also, from
   step ~40 most control answers hit the 128-token cap, so all lengths are equal. Better explanation: a **random
   walk**. Adam scales each step to a roughly fixed size whatever the signal, so pure noise still moves the
   weights (‖ΔW‖ 0.85 for the control vs 1.00 for main), and the direction it drifted (longer reasoning) happens
   to score well on Countdown. One seed, so the mechanism is not proven. The control also permuted rewards
   across the *batch*, so it had only 14 % zero-variance groups vs 72 % in main: more noisy gradient. A
   within-group permutation would be the tighter control.
2. **Hedging.** 18 % of adapter outputs give an answer, then "Note: this is wrong…", then a second answer. The
   reward grades the **last** tag, so hedging pays. Graded on the first answer: the gain halves (+0.054).
3. **× ÷ regression.** It got better on +/− items (+0.19) and worse on × ÷ items (0.129 → 0.036). The aggregate
   hides it. The shuffled control regressed on × ÷ just as much (−0.107), so this too is not reward-driven.
4. **The merge changed the model.** The LoRA update is ~0.08 % of each weight. bf16 can only represent changes
   larger than ~0.39 % of a value, so **82 % of the updated elements round back to the base value**. The merged
   model scores 0.502 against the adapter's 0.430, with only 65 % identical text.
5. **The adapter is alive.** It changes 84 % of outputs; the random-direction adapter of equal size changes 6 %.
6. **Training reward was flat** (0.40 → 0.41) and 72 % of groups had zero variance: ~1 informative group per
   step, ~110 in the whole run. The gradient was clipped on 50 % of steps; lr 2e-5 was never swept. **This is
   the root cause**: the pilot already showed 0.66 zero-variance against our own pre-registered 0.5 limit, and
   we went ahead (doubling prompts per step was not enough).

**What this check cannot detect:** contamination in pre-training, a scorer bias shared by all arms, regressions
outside Countdown, effects below ~5 points, sampling behaviour (greedy only; training ran at T = 1, so a T = 1
eval is the matching metric), seed variance (one seed per arm). Also: the pre-registered rules **missed the
multi-answer gaming**; only the post-hoc first-tag rescoring found it.

**Follow-up (11-18 evening): fp32 merge check.** Question: is R3's failure the bf16 cast, or a bug in
`merge.py`? We evaluated the adapter on fp32 weights, and the adapter merged into fp32 weights in host memory
(bit-identical to `merge.py`, unit-tested), on the same 500 items (`fp32_check.py`, `runs/verify_fp32`).
- fp32 merged vs fp32 adapter: −0.004 [−0.010, 0.000], **97.6 % identical text**. So the merge code is correct,
  and in fp32 R3 passes.
- **Surprise:** the *same* adapter scores 0.374 in fp32 and 0.430 in bf16 (−5.6 pts [−8.4, −3.0]; only 64 %
  identical text; hedging 27 % vs 18 %). The policy's greedy outputs sit on near-ties between tokens, so tiny
  numeric differences flip them.
- **Conclusion:** it isn't "the merge breaks the model". This policy's measured score depends on numeric
  precision (bf16 merge: 0.502, bf16 adapter: 0.430, fp32: 0.374). That's one more reason the gain isn't
  trustworthy, and a rule for shipping: evaluate the artifact at the precision users will run.
- Limitation: no fp32 *base* arm, so we don't know how precision-sensitive the base model is.

---

## Phase 8: silent failures (Part 4)

A silent failure is one where the run **finishes without an error and is wrong**. These are the ones we found
or ruled out, along the chain:

| stage | silent failure | found by |
|---|---|---|
| data | task too easy / reward measures length | GPU pilots (phase 4) |
| template | instruction ignored; thinking mode on | pilots; `enable_thinking=False` asserted |
| reward | last-tag rule rewards hedging; parser too strict (`\$18` scored 0) | first-tag rescoring; 32 adversarial tests |
| GRPO | zero-variance groups give no gradient; random advantages drift length | per-step logs; shuffled control |
| policy | an easy-item gain hides a hard-item regression | per-subset CIs |
| L2L memory | pinned memory 2×; allocator retries near the ceiling (near-OOM warning) | review; retry counters |
| L2L sync | missing stream dependency, hidden by other syncs | review; GPU equivalence |
| on-policy | rollout log-prob ≠ training log-prob (max 1.6 nats) | logged every step (harmless at ratio 1) |
| streaming | duplicates if killed before the first checkpoint | resume test |
| merge | bf16 rounding swallows the LoRA delta | merge report; merged arm |
| eval / deps | cached arm reused across prompt styles; transformers 5.x API change | asserts; version pin |

**Soup's checks** (read in their source): data doctor checks template/EOS/roles; `soup ship` says SHIP if
tuned > base and no general-benchmark regression. **This adapter passes `soup ship`** (0.430 > 0.318). None of
the failures above is caught by Soup's built-in checks.

---

## Phase 9: verdict (Part 5)

**DON'T SHIP the trained artifact** (the adapter and its merged model). The *engineering* is fine; the *model*
is not:
1. its gain cannot be told apart from a reward-free control;
2. part of the gain is gaming the last-tag rule;
3. it regressed on × ÷;
4. the merged file users would get is not the policy we evaluated.

**What must change first, and why in this order:**
1. **Signal** (the root cause): keep only prompts the base model solves sometimes (pass rate strictly between
   0 and 1 at k = 8); ≥ 3 seeds per arm; shuffled control (permuted within groups) mandatory; add a T = 1 eval.
2. **Reward**: exactly one answer tag (0 otherwise) removes the hedging; gate on × ÷ separately. Token-level
   loss normalisation is a hypothesis to A/B, not a proven fix (the length-bias story was wrong, see item 1).
3. **Artifact**: ship the unmerged adapter (an fp32 merged model is ~16 GB, which doubles streaming for an 8 GB
   user); gate on merged ≡ adapter at the precision users will run.
4. **Soup**: put the negative control, merged ≡ adapter and a CI lower bound > 0 into `soup ship`'s decision.

**How to say it in one breath:** the infrastructure is shown correct by the equivalence and resume tests; the
model is shown not ship-worthy by the controls; `soup ship` would have shipped it.

---

## Phase 10: the H6 A/B (11-18 evening)

**What.** The pre-registered research arm: *dynamic* residency keeps extra layers on the GPU while the decode
KV cache is still small, and evicts them as it grows. The prediction was −11.5 % step time. Rule fixed in
advance: adopt only if the gain is ≥ 2 % with the CI excluding 0, rollouts identical and nvidia-smi under the cap.

**How.** A-B-B-A blocks (regular, dynamic, dynamic, regular) on one instance; 8 steps each on the same prompts
and seed; the first 2 steps of each block discarded as warm-up; steps paired by index (`h6_queue.sh`, `h6_ab.py`).
The A-B-B-A order cancels slow drift (thermal, page cache) between blocks.

**Result.** +0.1 % [−0.4 %, +0.8 %], rollouts identical. **Rejected, keep regular.** Why the prediction failed:
it was made for the GSM8K shape (20 regular layers, long decode, big KV cache). On Countdown we had already hand-
tuned regular to 24 layers and the KV cache is small, so dynamic could add only **one** layer. That removed one
copy per pass, but decode time did not move. Why is **unattributed**: paged-KV overhead, or the allocator
retries (9 per block) from running at 7839 of 7840 MiB; we did not separate them.
**Lesson:** a prediction is tied to the workload shape; when the task changed, H6 should have been re-predicted.

**Also note H1 failed:** generation is 55 % of the step, not ≥ 70 %. The cause is understood (decode stops at
the longest answer; training was slower than predicted). Say it plainly: a failed hypothesis with an
explanation is a good answer.

## What was planned but not done (know this before you're asked)

| item | status | why it matters |
|---|---|---|
| **DFlash** (Part 2c) | not built | the team made it optional; the budget predicts it pays only above τ ≈ 2 |
| **H6: dynamic vs regular residency A/B** | **done 11-18**: A-B-B-A, 12 paired steps: +0.1 % [−0.4, +0.8] → rejected, keep regular (see below) | closes the research arm |
| **H2/H3 residency sweep** | not run | H2 is confirmed indirectly (128 ms = prediction); H3 is untested |
| second seed per arm | not run | R2 rests on one control run |
| control at 256 tokens | not run | the control's 0.404 is cut off at 128 tokens, so it's a lower bound |

---

## Key ideas to be able to explain without notes

- **Why a rising reward proves nothing:** GRPO's loss averages to ~0 by construction (advantages are centred
  per group), and reward can be gamed (hedging here).
- **Why the shuffled control is the most important check:** it holds everything constant except the
  information in the reward. Any gain it also gets is not learning.
- **Why a reward-free run still changes the model:** shuffled advantages give zero *expected* gradient, but
  each step is still a noisy step of roughly fixed size under Adam, so the weights random-walk. Which way they
  drift is luck; here it was longer reasoning, which helps on Countdown. (Not the Dr. GRPO length bias: that
  needs advantage and length to be correlated.)
- **Why bf16 swallows the merge:** bf16 has 8 bits of mantissa, so it can't represent a 0.08 % change to a
  weight. The change rounds to zero, or to one full step.
- **Why decode is the bottleneck:** one generated token = one full pass of copies over PCIe with almost no
  compute to hide them behind. Training has big matrix multiplies that hide 97 % of the copies.
- **Why 24, not 25, resident layers:** 25 fits on paper but leaves no slack, and the allocator thrashes silently.

---

## Hard questions to expect (from the fresh-context review), with short answers

- **"The training reward at T = 1 was flat, yet greedy eval gained 11 points. What did GRPO optimise?"** The mode
  of the policy moved; the sampled distribution did not improve. A T = 1 eval (`verify.py --k 4 --temperature 1`)
  is the matching metric, and we didn't run it.
- **"Is the shuffled control fair?"** It permutes rewards across the whole batch, so it had only 14 % zero-variance
  groups vs 72 % in main: ~3× more groups with non-zero (noise) advantages. Permuting *within* groups would keep
  main's pattern and is the tighter control. Its gain is still informative: noise alone moved the model that far.
- **"Is hedging really gaming? The second answer is a self-correction."** It's two attempts in one output (an
  implicit pass@2), and the reward pays for whichever comes last. The user reads the first. The fix is to score
  conflicting answers as 0, as our GSM8K/MATH parser already did and the Countdown one didn't.
- **"Why lr 2e-5?"** Not swept; we own that. Under Adam, lr sets how far the weights move regardless of signal, so
  the bottleneck was signal (72 % zero-variance groups), not lr. Plan: a 3-point sweep on informative prompts only.
- **"The control is 91 % correct when it answers. Isn't long reasoning the real win?"** Probably. A missing
  baseline is the base model with a CoT prompt and 512 tokens; it might beat every arm.
- **"Why did merging *gain* 7 points?"** Unexplained. Outputs sit on near-ties, and the flips happened to land
  positive. That's a reason to distrust the number, not to ship the merged model.
- **"One seed, n = 500?"** The CIs cover item sampling only. R2's ±4.4-pt interval can't resolve a 2.6-pt
  difference, and seed variance is unknown.
- **"Is the cached decode checked against teacher-forced logits?"** Yes: `tests/test_decode.py` (decode-time
  log-prob = training rescoring), and the on-policy gap is logged every step (mean 0.008 nats).
- **"Do you trust your VRAM numbers?"** nvidia-smi sampled every 200 ms for every run is device-wide ground truth;
  allocated/reserved come from PyTorch. The gap between them (allocator cache + context) is explained in Part 1.
- **"What would you change in `soup ship`?"** A required shuffled-control arm; merged ≡ adapter at deployment
  precision; a single-answer rescore; per-subset CIs; and SHIP only if the CI lower bound is > 0, not the point estimate.
- **"Three days of H100s?"** Fix the signal first (filtered prompts, single-tag reward), then ≥ 3 seeds × {main,
  within-group shuffled control}, a T = 1 eval and a CoT baseline. Only then DFlash and residency work, because
  on an H100 the model fits and streaming isn't the bottleneck.
