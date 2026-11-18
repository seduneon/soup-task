# Appendix: full-detail report

The long version of `REPORT.md` (same numbers, all from `task2_l2l/results/analysis.md`). `REPORT.md` is the ≤ 3-page submission; this file keeps every table and explanation that did not fit.

**Corrections after a fresh-context review (2026-11-19); where the text below disagrees, these win:**
- *Shuffled control mechanism:* not the Dr. GRPO length bias. Shuffled advantages are independent of the completion, so the expected gradient is zero under any normalisation, and from step ~40 most control answers hit the 128-token cap anyway. Best explanation: a noise-driven random walk (Adam gives roughly fixed-size steps; ‖ΔW‖ 0.85 vs 1.00 for main) that happened to drift towards longer reasoning. One seed; unproven.
- *Allocator retries at 25 resident:* 4, 17 and 3 in the three smoke steps, not "17 per step"; "rollout 5.7× slower" compared different batches (the same step at 24 resident with 0 retries took as long). The H6 dynamic blocks had 9 retries each.
- *Part 1 gaps* should be read at 24 resident: nvidia-smi predicted 7410 vs 7793 measured (+383); decode predicted ~135 ms vs 128 ms; step 23.1 s vs 20.1 s, where the totals agree only by cancellation (decode stops early, training is 1.7× slower than modelled).
- *× ÷ regression* also appears in the shuffled control (−0.107), so it is not reward-driven.
- *Root cause:* 72 % zero-variance groups (~1 informative group per step), 50 % of steps gradient-clipped, lr not swept.
- *`soup ship`:* only its tuned > base leg was evaluated (no general benchmark); Soup was read in source, not run.
- *Fix 3:* ship the unmerged adapter; an fp32 merged model (~16 GB) doubles streaming for the target user.


**Verdict: DON'T SHIP.** The streamed L2L engine works and stays under the cap. The artifact is a different
matter: its held-out gain cannot be told apart from a shuffled-reward control, and about half of the gain
comes from hedged second answers that the reward allowed. The merged bf16 model is not the adapter that
was trained. Details in §3 and §5.

Setup: one Vast.ai RTX 3070 (8 GB physical, so the cap is the hardware; CUDA sees 7840 MiB), PCIe 4.0 x16
(measured 25.5 GB/s H2D), 31 GB host RAM. Qwen3-4B non-thinking, bf16, LoRA r = 16 on all 7 linears of 36
layers (33 M params, fp32). GRPO, 4 prompts × G = 8 per step, 100 steps, lr 2e-5, β = 0, seq-normalised loss,
one update per batch (ratio = 1). Task: **Countdown-3** (reach a target with 3 numbers and + − × ÷;
verifiable exact-fraction reward on the last `<answer>`; 1500 train / 500 held-out prompts, disjoint).
Base pass at T = 1: 0.34. GSM8K (0.94 greedy, too easy), MATH 3–5 (41 % truncated at 512 tokens, so the
reward measured length) and answer-only GSM8K (model ignores the instruction) were rejected on GPU
pilots before training; see `task2_l2l/README.md`. Engine, GRPO, streaming, merge and checks are our own
code (`task2_l2l/`); `transformers` supplies only the Qwen3 layer classes, config and tokenizer.

## 1. Memory and time: predicted vs measured

Prediction (`task1_budget/`, committed before the runs, re-run with measured hardware and the Countdown
shape): 25 resident / 11 streamed layers, peak 7603 MiB (prefill), 22.2 s/step, 76 % generation.
Smoke runs at 25 resident hit 7835 of 7840 MiB with 17 allocator retries per step (each retry flushed the
cache: rollout 17 s instead of 3 s), so the main run used **24 resident** + `expandable_segments` (0 retries).

| quantity | predicted | measured (main run, 100 steps) | gap explained |
|---|---|---|---|
| allocated peak | 7410 MiB at 24 resident, minus 233 MiB context = 7177 | **7232 MiB** | +55 MiB (0.8 %): `repeat_kv` and logits transients not in the ledger |
| reserved peak | — | 7542 MiB | allocator cache: +310 MiB over allocated |
| nvidia-smi peak (device-wide, sampled every 200 ms in every run) | 7603 at 25 | **7793 MiB** (≤ 7839 in any run) | = reserved 7542 + CUDA context 253 (re-measured with expandable_segments; 233 without) |
| host RSS | 10.05 GiB | 5.27 GiB | ledger counted a full shard while loading; we read tensor by tensor |
| decode, per weight pass | 128 ms at 25 resident (11 copies + resident-layer compute + per-layer overhead) | **128 ms** at 24 resident | 12 copies × 7.9 ms = 96 ms of PCIe per pass (`copy_ms`), the rest is resident-layer compute and launch overhead; copy hidden only 35 % |
| decode passes / step | 128 (always to max length) | mean 75.6, max 127 | generation stops when the last sequence ends: one long completion sets the step |
| prefill | 0.56 s | 1.26 s | micro-batches of 2 × 96 tokens: GEMMs far below the 42 TFLOPS measured at 12k rows |
| train fwd + bwd | 5.3 s | 9.1 s | same: 32 micro-batches of 224 tokens per layer, ×3 passes; FLOP model assumed peak GEMM rate |
| step | 22.2 s, 76 % generation | mean 20.1 s, median 26.7 s, **55 % generation** | shorter decode, slower training |
| merge GPU | 0.51 GiB | 441 MiB allocated, 661 MiB nvidia-smi | context; one tensor at a time |

Copy/compute overlap (CUDA events per layer copy vs compute-stream stall): training hides **97 %** of copy
time (big GEMMs), rollout only **35 %** (a decode pass has ~1 ms of compute per 7.9 ms copy). The L2L
bottleneck is PCIe during decode, as predicted; one more resident layer would save ~0.9 s/step.

## 2. L2L, streaming, merge

- **L2L engine** (`engine.py`): pinned host store (exact-size `cudaHostRegister`), resident slots + 2 rotating
  prefetch slots, copies on a separate stream ordered by events, activations checkpointed at layer boundaries
  (pinned host), layer-by-layer backward by recompute. One layer loop serves prefill, decode (KV cache),
  scoring and training. GRPO step: rollout (prefill + 1 pass per token) → disk → train forward → backward.
- **Equivalence** (`equivalence.py`, on the GPU, bf16, truth = fp32): logits streamed vs resident
  **bit-identical**; LoRA gradients pass `err(stream) ≤ 1.5 · err(reference)` against the fp32 truth, per
  tensor and in aggregate (worst ratio 1.20–1.26; Qwen3-0.6B 28 layers all streamed, and Qwen3-4B 4 layers).
  The tolerance is the reference's own bf16 error, so streaming may not add more than half again of it. On
  CPU 26 tests are bit-exact, with controls that must fail (swapped layers: error 1.13 vs 5e-7).
- **Deterministic resume, on the GPU** (`runs/resume_demo`): SIGKILL after step 3's update, after step 6's
  rollout, after step 8's checkpoint; 4 attempts. Sample ledger identical to the uninterrupted main run,
  0 duplicates, 0 gaps, and **rollout tokens and rewards identical for all 10 steps**. CPU tests also cover
  an epoch boundary and bit-identical final LoRA.
- **Merge** (`merge.py`): tensor by tensor, shard by shard; 15 s; GPU peak above; host 8.6 GiB.
- **DFlash**: not done (optional per the team). The budget predicts it pays only above τ ≈ 2.06 accepted
  tokens per pass, because the resident drafter (1 GB plus its buffers) displaces ~10 resident layers (25 → 15).

## 3. Did the model actually train?

`verify.py` (rules written before any run): base, adapter, merged, shuffled-reward control (same pipeline and
steps; rewards permuted across the batch), plus a random-direction control (`random_adapter.py`: same
per-tensor ‖ΔW‖ as the trained adapter, random direction). 500 held-out items, greedy, max 128 tokens, paired
bootstrap 95 % CIs. Full output: `task2_l2l/results/analysis.md`, `runs/verify/verify_report.json`.

| arm | pass | +/− items (360) | needs × or ÷ (140) | pass, first `<answer>` | >1 `<answer>` | truncated | mean tokens |
|---|---|---|---|---|---|---|---|
| base | 0.318 | 0.392 | 0.129 | 0.316 | 1.6 % | 2.0 % | 19 |
| adapter | 0.430 | 0.583 | 0.036 | 0.370 | 18.0 % | 10.8 % | 34 |
| shuffled control | 0.404 | 0.553 | 0.021 | 0.404 | 0 % | 55.8 % | 119 |
| merged (bf16) | 0.502 | 0.681 | 0.043 | 0.488 | 7.0 % | 4.2 % | 23 |
| random direction, attached | 0.320 | 0.394 | 0.129 | 0.316 | 1.4 % | 1.6 % | 19 |
| random direction, merged | 0.320 | 0.394 | 0.129 | 0.316 | 1.6 % | 1.8 % | 19 |

| rule | result | |
|---|---|---|
| R1 adapter − base > 0 | +0.112 [+0.060, +0.162] | pass |
| R2 adapter − control > 0 | +0.026 [−0.018, +0.070] | **fail** |
| R3 merged − adapter ∋ 0 | +0.072 [+0.040, +0.104]; same text on 65 % of items | **fail** |
| R4 not format only | every arm answers in the tag; +0.112 | pass |

What the numbers say:
1. **The control gained almost as much.** With shuffled rewards the policy drifted to long step-by-step
   reasoning (19 → 119 tokens): when it finishes it is right 91 % of the time, but 56 % of its answers are
   cut at 128 tokens. Random advantages are not neutral: under sequence-normalised GRPO, a long completion
   spreads its negative advantage over more tokens, so length drifts up (the length bias described in the
   Dr. GRPO paper), and on this model longer = more reasoning = more correct. A reward-free change of style
   moves held-out accuracy by +8.6 pts [+3.6, +13.4], so a +11 pt gain is not evidence of learning from the reward.
2. **Half the adapter's gain is hedging.** 18 % of adapter outputs give an answer, a "Note: this is wrong…",
   then a second answer. The reward grades the last tag. Graded on the first answer (what a user reads),
   adapter − base is +0.054 [0.000, +0.108] and adapter − control −0.034 [−0.080, +0.012].
3. **It got worse at × and ÷.** Items that need × or ÷: 0.129 → 0.036 (−0.093 [−0.157, −0.029]). The policy
   collapsed to +/− expressions, which win often on the easy items. Training reward stayed flat
   (0.40 → 0.41; 72 % of groups had zero reward variance and gave no gradient).
4. **The merged model is a different model.** The mean LoRA update is 0.08 % of a weight, below bf16's
   0.39 % step: 82 % of elements round back to the base value, and the merge keeps only 50–92 % of each
   tensor's delta (`runs/merge/merge_report.json`). The merged model then scores *higher* than the adapter it
   came from (+7.2 pts), with fewer hedged answers (7 % vs 18 %). This is not eval noise: the random-direction
   control, with the same per-tensor ‖ΔW‖, moves nothing (+0.002 [−0.010, +0.014], 94 % identical text to
   base), and its merge scores the same as its adapter (+0.000 [−0.008, +0.008], 94 % identical text). Rounding the *trained* delta, whose surviving elements are
   the large ones along a direction the model is sensitive to, produces a different policy. A shipped merged
   model would not be the policy that was trained or evaluated.
5. **The adapter is alive and specific.** The trained update changes greedy text on 84 % of items; the
   random one of equal size on 6 %. So "the adapter did nothing" is ruled out; what it learned is the problem.

What this check does **not** detect: contamination of the held-out set in pre-training; a bias shared by
every arm's scorer; regressions outside Countdown (no general benchmark was run); effects below the CI
half-width (~±5 pts at n = 500); sampling-time behaviour (greedy only); seed variance (one seed per run).

## 4. Silent failures (each one finishes without an error)

| stage | what goes wrong silently | how we checked / evidence | caught by Soup? |
|---|---|---|---|
| data | train/held-out overlap inflates the held-out gain | `prepare_countdown.py` asserts disjoint keys; prepare_data drops 7 MATH train items found in test | data doctor: no (template/EOS/roles/input-truncation only) |
| data → reward | task too easy (GSM8K 0.94): no gradient; or the reward measures length (MATH: 41 % truncated, finished answers 96 % right) | GPU pilots with k = 8 before training: pass rate, zero-variance groups, truncation | no; `soup advise` checks memory/ROI, not reward signal |
| template | instruction ignored (answer-only GSM8K: 88 % truncated); thinking mode on | pilots; `enable_thinking=False` asserted in train/eval; prompt-style mismatch aborts eval | data doctor checks the chat template renders, not the model's compliance |
| reward | parser too strict (`\$18`, `1{,}000` scored 0) or too loose; **last-tag rule rewards hedging** (18 % of adapter outputs) | 32 + Countdown adversarial unit tests; first-tag rescoring (§3.2) | no |
| GRPO | zero-variance groups (72 %) silently give no gradient; random advantages still drift length | logged per step; shuffled-reward control (§3.1) | no |
| GRPO / policy | gain on easy items hides a regression (× ÷ −9 pts) | per-subset paired CIs | `soup ship` compares one aggregate number |
| L2L memory | pinned memory 2× the ledger (power-of-two rounding: 9.2 GB vs 3.1); allocator retries (17/step, rollout 5.7× slower) visible only in `torch.cuda.memory_stats`; CUDA sees 7840 of 8192 MiB | review + ledger; retry counter per phase; nvidia-smi at 200 ms | no |
| L2L sync | prefetch not ordered after the previous pass's compute: hidden by incidental syncs | review; fixed with persistent events; GPU equivalence test | no |
| on-policy | rollout log-prob ≠ training log-prob (bf16 decode vs recompute): mean 0.008, max 1.6 nats | logged every step (`rollout_vs_train_*`) | no. Harmless at ratio = 1; would bias PPO clipping or a KL term |
| streaming | duplicate samples when killed before the first checkpoint | test (fixed); GPU kill/resume demo identical to the uninterrupted run | no |
| merge | **bf16 rounding swallows most of the LoRA delta**: the merged model ≠ the adapter | `merge_report.json` delta_kept / changed_frac; merged vs adapter arm; random-direction control | `soup ship` measures `AdapterChange` but it is "never part of the decision" |
| eval | cached arm reused across prompt styles | guard in `verify.py` (asserts style and max_new) | no |
| deps | transformers 5.x changed Qwen3 rotary/layer signatures | pinned 4.52.3 | no |

`soup ship` (read from its source, `utils/ship_verdict.py`): SHIP ⇔ tuned > base on the task metric
(strict, point estimate unless a noise floor is configured) and no general-benchmark regression. Our adapter
passes that leg (0.430 > 0.318). Nothing in it compares against a negative control, checks that the merged
artifact matches the adapter, or rescores the answer the way a user reads it.

## 5. Verdict: DON'T SHIP

The engineering is sound: under 8 GB on nvidia-smi in every phase, equivalence and resume proven, the time
model right where it is transfer-bound. The model is not: (1) no gain over a reward-free control; (2) the
measured gain partly comes from multi-answer hedging and a collapse to +/−; (3) the artifact users would get
(merged bf16) is not the evaluated policy. What must change first:
1. Reward: exactly one `<answer>` (0 otherwise), and a length/format penalty or a token-level (Dr. GRPO)
   loss so random advantages cannot drift length.
2. Signal: harder-balanced prompts or larger G so fewer groups have zero variance; report × ÷ separately as
   a gate; more steps, ≥ 3 seeds per arm, and the shuffled control as a required arm.
3. Ship the adapter unmerged, or merge into fp32 target weights, and gate on merged ≡ adapter (text identity
   ≥ 99 % at greedy).
4. Soup: make the negative control and merged-vs-adapter checks part of `soup ship`'s decision.

## Surprises and concerns

The biggest surprise was the shuffled-reward control: it produced the largest behaviour change of any run
and a held-out gain close to the "real" one. Without it this run would look like a clean success (+11 pts,
CI excluding 0). The second was the merge: a delta of 0.08 % of each weight is mostly rounded away in bf16,
and the result is a different model that happens to score better here. What still concerns me: one seed per
arm, so the control's length drift may be one outcome of several; greedy only; max 128 tokens cuts the
control's reasoning, so its 0.404 is a lower bound; and no general benchmark, so regressions outside the task
are unmeasured.

## AI tools

Claude Code (Opus) wrote most of the code, ran the GPU jobs and drafted this report. A second, fresh-context
instance reviewed the engine: 6 findings, all fixed (pinned memory 2×, a missing stream dependency, a
wrong-device stash, reward parser gaps, resume duplicates). Human decisions and corrections are in the git
history: dataset choice and the rejection of GSM8K/MATH, pre-registered rules, 24 resident layers, keeping the
merge. Tools were unreliable where they predicted instead of measured (the 5.3 s training estimate; the first
dataset picks); every number in this report comes from a log in the repo.

Evidence: `logs/` (timestamped), `task2_l2l/runs/*/logs/*.csv` and `runs/verify/*.nvsmi*.csv` (nvidia-smi
every 200 ms, device-wide and per process), `task2_l2l/verify.py` (Part 3 check), `task2_l2l/equivalence.py`
+ `tests/` (CPU test suite), `task2_l2l/results/analysis.md` (all numbers above, regenerated by `analyze.py`).


## Added after the draft (2026-11-18 evening)

### fp32 merge check (R3 follow-up): `task2_l2l/fp32_check.py` → `results/fp32_check.md`

Question: is "merged ≠ adapter" a bug in `merge.py` or the bf16 cast? Same 500 items and settings as `runs/verify`,
fp32 weights and compute, 8 resident layers (nvidia-smi peak 7001 MiB in the smoke run; reserved 6672 MiB).
The in-memory merge (`store.merge_lora`, `evaluate.py --merge-in-memory`) is bit-identical to `merge.py` for
fp32 and bf16 bases (`tests/test_merge.py::test_in_memory_merge_matches_merge_py`); fp32 kept 99.9999 % of every delta.

| arm | pass | >1 `<answer>` | truncated | tokens |
|---|---|---|---|---|
| adapter, bf16 | 0.430 | 18.0 % | 10.8 % | 34 |
| merged, bf16 | 0.502 | 7.0 % | 4.2 % | 23 |
| adapter, fp32 | 0.374 | 27.0 % | 20.2 % | 46 |
| merged, fp32 | 0.370 | 27.0 % | 19.8 % | 45 |

| pair | diff | 95 % CI | identical text |
|---|---|---|---|
| merged fp32 − adapter fp32 | −0.004 | [−0.010, +0.000] | 97.6 % |
| adapter fp32 − adapter bf16 | −0.056 | [−0.084, −0.030] | 64.0 % |
| merged bf16 − merged fp32 | +0.132 | [+0.094, +0.170] | 48.0 % |

Reading: `merge.py` is correct (R3 passes in fp32). But the trained policy itself is precision-fragile: the
same weights run in fp32 instead of bf16 change 36 % of greedy outputs and 5.6 pts of score. The base model's
precision sensitivity was not measured (no fp32 base arm), so "adapter fp32 − base bf16 = +0.056" mixes two
precisions. The shipped artifact must be evaluated at the precision users will run.

### H6 A/B, dynamic vs regular residency: `task2_l2l/h6_ab.py` → `results/h6_ab.md`

Pre-registered (task1_budget/README.md): adopt dynamic only if step time improves ≥ 2 % with CI excluding 0,
rollouts identical, nvidia-smi under the cap. A-B-B-A blocks (regular, dynamic, dynamic, regular), 8 steps each
on the same prompts and seed, steps 0–1 discarded, 12 paired steps.

| block | mean step | resident in decode | decode ms/pass | layer copies/step | nvidia-smi peak | retries |
|---|---|---|---|---|---|---|
| reg1 | 19.61 s | 24 | 128.2 | 894 | 7791 | 0 |
| dyn1 | 19.59 s | 25 | 128.1 | 862 | 7839 | 6 |
| dyn2 | 19.48 s | 25 | 127.8 | 862 | 7839 | 6 |
| reg2 | 19.49 s | 24 | 128.6 | 894 | 7791 | 0 |

Step-time saving +0.1 % [−0.4 %, +0.8 %]; rollouts identical (tokens, prompts, rewards). **Rejected: keep regular.**
Why the −11.5 % prediction failed: it was made for the GSM8K shape (20 regular resident layers, decode to 512
tokens, large KV). On Countdown, regular already holds 24 layers and the KV cache is small (576 MiB paged vs 958
static), so dynamic could add only one layer. It did remove one copy per pass (894 → 862 copies), but decode time
per pass did not move, so the paged-KV gather overhead cancelled it; and it ran at 7839 of 7840 MiB with
allocator retries. Lesson: the gain of dynamic residency depends on the task shape; re-predict after the task changes.
