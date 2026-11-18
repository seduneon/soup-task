# GRPO + LoRA on Qwen3-4B under an 8 GB cap

**Verdict: DON'T SHIP the trained artifact** (the LoRA adapter and its merged model). The streamed engine
works: every phase stays under 8 GB on nvidia-smi, L2L matches a non-streamed reference, resume is exact. The
model does not. **Why it failed:** (1) a control trained on *shuffled* rewards gained almost as much (+8.6 of
+11.2 pts), so the gain is not shown to come from the reward; (2) the reward graded the *last* `<answer>` tag:
18 % of outputs hedge with two answers, and graded on the first answer half the gain disappears; (3) it got worse
on × ÷ items (0.129 → 0.036); (4) the gain is not robust to how the model is run: the merged bf16 model scores
0.502, the adapter 0.430 in bf16 and 0.374 in fp32. Fixes in §4; full detail in `docs/APPENDIX.md`.

**Setup.** Vast.ai RTX 3070 (8 GB physical, so the cap is the hardware; CUDA sees 7840 MiB), PCIe 4.0 x16 (25.5 GB/s
measured), 31 GB RAM. Qwen3-4B non-thinking, bf16; LoRA r = 16 on all 7 linears × 36 layers (33 M params, fp32).
GRPO: 4 prompts × G = 8, 100 steps, lr 2e-5, β = 0, sequence-normalised loss. **Task: Countdown-3** (reach a target
with 3 numbers and + − × ÷; exact-fraction reward; 1500 train / 500 held-out, generated, disjoint), chosen on GPU
pilots: ~20-token answers (each decoded token costs a 128 ms pass over streamed weights), base pass 0.34 at T = 1.
GSM8K (0.94, no signal) and MATH 3–5 (41 % truncated) were rejected (§3). All engine/GRPO/streaming/merge code is
ours (`task2_l2l/`); `transformers` supplies only the Qwen3 layer classes and tokenizer.

## 1. Memory, L2L, streaming (Parts 1–2)

The budget (`task1_budget/`, committed before any run) predicted 25 resident / 11 streamed layers and a 7603 MiB
peak. At 25 resident the smoke run hit 7835 of 7840 MiB with 17 allocator retries per step (rollout 5.7× slower),
so the runs used **24 resident** + `expandable_segments` (0 retries in every run).

| quantity | predicted | measured (main run) | why the gap |
|---|---|---|---|
| allocated peak | 7177 MiB (24 res.) | 7232 MiB | +0.8 %: `repeat_kv` and logits transients not in the ledger |
| nvidia-smi peak (200 ms samples, every run) | 7603 MiB (25 res.) | **7793 MiB** (≤ 7839 any run) | reserved 7542 + CUDA context 253 |
| decode, per weight pass | 128 ms (25 res.) | **128 ms** | 12 copies × 7.9 ms (96 ms, PCIe) + resident compute |
| prefill / train fwd+bwd | 0.56 s / 5.3 s | 1.26 s / 9.1 s | small micro-batches run GEMMs far below the peak rate assumed |
| step; generation share | 22.2 s; 76 % | 20.1 s mean; **55 %** | generation stops at the longest completion (75.6 of 128 passes) |
| host RSS | 10.05 GiB | 5.27 GiB | weights read tensor by tensor, not a full shard |

- **L2L** (`engine.py`): exact-size pinned host buffers; resident slots + 2 rotating prefetch slots; copies on a
  separate stream ordered by events; boundary checkpoints in pinned host memory; layer-by-layer backward by
  recompute. One layer loop serves prefill, decode, scoring and training. Overlap: training hides **97 %** of
  copy time, decode **35 %** (1 ms compute per 7.9 ms copy). **Bottleneck: PCIe in decode, as predicted.**
- **Equivalence** (`equivalence.py`, GPU, bf16 vs fp32 truth): streamed logits **bit-identical** to resident;
  LoRA gradients pass err(stream) ≤ 1.5 × err(reference) per tensor (worst 1.26) on Qwen3-0.6B (28 layers
  streamed) and Qwen3-4B (4 layers). The tolerance is the reference's own bf16 error. A swapped-layer control fails.
- **Resume** (GPU, `runs/resume_demo`): SIGKILL at 3 points (after an update, mid-rollout, after a checkpoint):
  sample ledger = uninterrupted run, 0 duplicates, 0 gaps, identical rollout tokens and rewards for all 10 steps.
- **H6 dynamic-residency A/B** (A-B-B-A, 12 paired steps, identical rollouts): +1 resident layer only, its saved
  copy eaten by paged-KV overhead: **+0.1 % [−0.4, +0.8]** < 2 % rule → keep regular. H1 (gen ≥ 70 %) also failed (55 %).
- **Merge** (`merge.py`): tensor by tensor, 15 s, 661 MiB nvidia-smi. **DFlash: not done** (optional per the
  team); the budget says it pays only above τ = 2.06, since the 1 GB drafter displaces ~10 resident layers.

## 2. Did the model actually train? (Part 3)

A rising reward proves nothing (GRPO's loss is ~0 by construction; here reward was flat anyway, 0.40 → 0.41, with
72 % of groups at zero variance). `verify.py` (rules fixed before any run) compares five arms on 500 held-out items,
greedy, paired bootstrap 95 % CIs: base, adapter, merged, a **shuffled-reward control** (same pipeline, rewards
permuted) and a **random-direction adapter** (same per-tensor ‖ΔW‖).

| arm | pass | × ÷ items (140) | pass on 1st `<answer>` | >1 `<answer>` | truncated | tokens |
|---|---|---|---|---|---|---|
| base | 0.318 | 0.129 | 0.316 | 1.6 % | 2.0 % | 19 |
| adapter | 0.430 | 0.036 | 0.370 | 18.0 % | 10.8 % | 34 |
| shuffled control | 0.404 | 0.021 | 0.404 | 0 % | 55.8 % | 119 |
| merged (bf16) | 0.502 | 0.043 | 0.488 | 7.0 % | 4.2 % | 23 |
| random direction | 0.320 | 0.129 | 0.316 | 1.4 % | 1.6 % | 19 |

Rules: **R1** adapter − base > 0: +0.112 [+0.060, +0.162], pass. **R2** adapter − control > 0: +0.026
[−0.018, +0.070], **fail**. **R3** merged − adapter ∋ 0: +0.072 [+0.040, +0.104], 65 % identical text, **fail** (passes in fp32, item 4).
**R4** not format-only: every arm uses the tag, pass.

1. **A reward-free run gains almost as much** (control − base +8.6 pts [+3.6, +13.4]). Random advantages are not
   neutral under sequence normalisation: a long completion spreads its negative advantage over more tokens, so
   length drifts up (Dr. GRPO's length bias; 19 → 119 tokens), and here longer reasoning is more often right.
2. **Half the gain is hedging**: an answer, "Note: this is wrong…", a second answer; the reward reads the last.
   Graded on the first answer (what a user reads): adapter − base +0.054 [0.000, +0.108]; vs control −0.034.
3. **Worse at × ÷** (−0.093 [−0.157, −0.029]): it won the easy +/− items and the aggregate hides the regression.
4. **The merge changed the model, and so does precision.** The update is ~0.08 % of a weight, below bf16's
   half-step (0.39 %): 82 % of elements round back to base. Not eval noise: the random-direction adapter changes
   6 % of outputs (ours 84 %), and its merge equals its adapter. An **fp32 rerun** (`fp32_check.py`): merged ≡
   adapter (−0.004 [−0.010, 0.000], 97.6 % identical), so `merge.py` is right; but the *same* adapter scores 0.374
   in fp32 vs 0.430 in bf16 (−5.6 pts [−8.4, −3.0]). Its greedy outputs flip with numeric precision.

**Not detected:** pre-training contamination; scorer bias shared by all arms; regressions outside Countdown;
effects below ~±5 pts; sampling behaviour (greedy only); seed variance (one run per arm).

## 3. Silent failures (Part 4): each one finishes without an error

| stage | what breaks silently | how checked · evidence | Soup check catches it? |
|---|---|---|---|
| data | too easy (GSM8K 0.94) or reward measures length (MATH 41 % truncated); instruction ignored (88 % truncated) | GPU pilots: pass rate, zero-var groups, truncation | no: data doctor checks template/EOS/roles |
| reward | **last tag graded: hedging pays (18 %)**; parser too strict (`\$18` → 0) | first-tag rescoring; 32 parser tests | no |
| GRPO | 72 % zero-variance groups; random advantages drift length | per-step log; shuffled control | no |
| policy | easy-item gain hides × ÷ regression | per-subset CIs | no: one aggregate |
| L2L memory | pinned memory 2× ledger (power-of-two rounding); allocator retries, rollout 5.7× slower | exact-size `cudaHostRegister`; retry counter | no |
| L2L sync | prefetch not ordered after previous compute | review; events; GPU equivalence | no |
| on-policy | rollout ≠ training log-prob (max 1.6 nats) | logged; harmless at ratio 1 | no |
| streaming | duplicates if killed before 1st checkpoint | test (fixed); GPU kill demo | no |
| merge | **bf16 rounding swallows the delta** | `merge_report.json`; merged arm | no: `AdapterChange` never decides |

`soup ship` (`utils/ship_verdict.py`): SHIP if tuned > base with no general regression. **This adapter passes
it** (0.430 > 0.318): nothing there compares to a negative control, checks merged ≡ adapter, or rescores the
answer a user reads.

## 4. Verdict: DON'T SHIP, and what must change first (in priority order)

1. **Reward:** exactly one `<answer>` tag, else 0. **Loss:** token-level normalisation (Dr. GRPO), so random
   advantages cannot drift length. These remove the two effects behind the measured gain.
2. **Signal:** keep only prompts with base pass rate in (0, 1) at k = 8 (72 % of groups were wasted); gate on
   × ÷ separately; ≥ 3 seeds per arm, with the shuffled control as a required arm.
3. **Artifact:** ship the adapter unmerged or merged in fp32 (verified ≡ adapter); gate on merged ≡ adapter,
   and evaluate at the precision users will run, since this policy's score moves 5.6 pts between bf16 and fp32.
4. **Soup:** make the negative control and the merged-vs-adapter check part of `soup ship`'s decision.

**Surprises and concerns.** The shuffled control produced the largest behaviour change of any run and nearly
the "real" gain; without it this run looks like a clean +11-pt success. Second, numeric precision alone (bf16
merge, or fp32 compute) moves this policy's score by 6–13 pts: its greedy outputs sit on near-ties. Concerns: one
seed per arm; greedy only; 128 tokens cut the control's reasoning (0.404 is a lower bound); no general benchmark.

**AI tools.** Claude Code (Opus) wrote most code, ran the GPU jobs and drafted this report; a fresh-context
instance reviewed the engine (6 findings, all fixed, e.g. pinned memory 2×, a missing stream dependency, resume
duplicates). Human decisions are in git (dataset, pre-registered rules, 24 resident, keeping the merge). The
tools failed where they predicted rather than measured (training time, first dataset picks).

**Evidence.** `logs/`, nvidia-smi CSVs (200 ms, every run), `verify.py`, `equivalence.py`, `tests/`; numbers in `task2_l2l/results/`.
