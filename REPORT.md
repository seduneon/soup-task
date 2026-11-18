# GRPO + LoRA on Qwen3-4B under an 8 GB cap

**Verdict: DON'T SHIP the trained artifact** (the LoRA adapter and its merged model). The streamed engine
works: every phase stays under 8 GB on nvidia-smi, L2L matches a non-streamed reference, resume is exact. The
model does not. **Why it failed:** (0) training had almost no signal: 72 % of GRPO groups had zero reward
variance (~1 informative group per step), and the training reward never moved (0.40 → 0.41); (1) a control trained
on *shuffled* rewards gained almost as much on held-out data (+8.6 of +11.2 pts), so the gain is not shown to come
from the reward; (2) the reward graded the *last* `<answer>` tag: 18 % of outputs hedge with two answers, and
graded on the first answer half the gain disappears; (3) × ÷ items regressed (0.129 → 0.036), as in the control;
(4) the score depends on how the model is run: merged bf16 0.502, adapter 0.430 in bf16 and 0.374 in fp32.

**Setup.** Vast.ai RTX 3070 (8 GB physical, so the cap is the hardware; CUDA sees 7840 MiB), PCIe 4.0 x16 (25.5 GB/s
measured), 31 GB RAM. Qwen3-4B non-thinking, bf16; LoRA r = 16 on all 7 linears × 36 layers (33 M params, fp32).
GRPO: 4 prompts × G = 8, 100 steps, lr 2e-5, β = 0, sequence-normalised loss. **Task: Countdown-3** (reach a target
with 3 numbers and + − × ÷; exact-fraction reward; 1500 train / 500 held-out, generated, disjoint), chosen on GPU
pilots (GSM8K and MATH rejected, §3): ~20-token answers (each decoded token costs a 128 ms pass over streamed
weights), base pass 0.34 at T = 1. **Stack: our own** (`task2_l2l/`): Soup refuses `stream_layers` with rollout
tasks (`config/schema.py`); `transformers` supplies only the Qwen3 layer classes and tokenizer.

## 1. Memory, L2L, streaming (Parts 1–2)

The budget (`task1_budget/`, committed before any run) predicted 25 resident / 11 streamed layers and a 7603 MiB
peak. At 25 resident the smoke run hit 7835 of 7840 MiB (up to 17 allocator retries in a step), so the runs used
**24 resident** + `expandable_segments` (0 retries in every training and eval run). Predictions below are at 24.

| quantity | predicted | measured (main run) | why the gap |
|---|---|---|---|
| allocated peak | 7177 MiB | 7232 MiB | +0.8 %: `repeat_kv` and logits transients not in the ledger |
| nvidia-smi peak (200 ms samples, every run) | 7410 MiB | **7793 MiB** (≤ 7839 any run) | +383: allocator cache (reserved 7542) and context 253 vs 233 assumed |
| decode, per weight pass | 135 ms | **128 ms** | 12 copies × 7.9 ms = 96 ms of PCIe; overhead slightly overestimated |
| prefill / train fwd+bwd | 0.56 s / 5.3 s | 1.26 s / 9.1 s | **real modelling error**: GEMMs at micro-batch 1 × 224 tokens run far below peak |
| step; generation share | 23.1 s; 76 % | 20.1 s; **55 %** | totals agree only by cancellation: decode stops at the longest answer (75.6 of 128 passes) |
| host RSS (train / merge) | 10.05 / 10.55 GiB | 5.27 / 8.44 GiB | weights read tensor by tensor, not a full shard |

- **L2L** (`engine.py`): pinned host buffers; resident slots + 2 prefetch slots, copies on a separate stream
  ordered by events; boundary checkpoints; layer-by-layer backward by recompute; one layer loop for prefill, decode,
  scoring, training. Copy hidden: train **97 %**, decode **35 %**. **Bottleneck: PCIe in decode, as predicted.**
- **Equivalence** (`equivalence.py`, GPU): streamed logits **bit-identical** to resident; LoRA gradients within
  1.5 × the non-streamed bf16 reference's own error vs fp32 (worst 1.26), Qwen3-0.6B (all streamed) and Qwen3-4B
  (4 layers). CPU tests: a swapped-layer control fails.
- **Resume** (GPU, `runs/resume_demo`): SIGKILL at 3 points (after an update, mid-rollout, after a checkpoint):
  sample ledger = uninterrupted run, 0 duplicates, 0 gaps, identical rollout tokens and rewards for all 10 steps.
- **H6 dynamic-residency A/B** (pre-registered; A-B-B-A, 12 paired steps, identical rollouts): **+0.1 % [−0.4, +0.8]**
  < 2 % rule → keep regular. It added only 1 layer (short KV) at 7839 MiB with retries; the loss of the saved copy is
  unattributed. H1 (generation ≥ 70 %) also failed (55 %).
- **Merge** (`merge.py`): tensor by tensor, 15 s, 661 MiB nvidia-smi; merged eval 6729 MiB. **DFlash: not done**
  (optional); predicted to pay only above τ = 2.06, as drafter + 16-token verify buffers displace ~10 layers.

## 2. Did the model actually train? (Part 3)

A rising reward proves nothing (GRPO's loss is ~0 by construction). `verify.py` compares arms on 500 held-out
items, greedy, paired bootstrap 95 % CIs: base, adapter, merged, a **shuffled-reward control** (rewards permuted
across the batch) and a **random-direction adapter** (same per-tensor ‖ΔW‖; added post hoc).

| arm | pass | × ÷ items (140) | pass on 1st `<answer>` | >1 `<answer>` | truncated | tokens |
|---|---|---|---|---|---|---|
| base | 0.318 | 0.129 | 0.316 | 1.6 % | 2.0 % | 19 |
| adapter | 0.430 | 0.036 | 0.370 | 18.0 % | 10.8 % | 34 |
| shuffled control | 0.404 | 0.021 | 0.404 | 0 % | 55.8 % | 119 |
| merged (bf16) | 0.502 | 0.043 | 0.488 | 7.0 % | 4.2 % | 23 |
| random direction | 0.320 | 0.129 | 0.316 | 1.4 % | 1.6 % | 19 |

Pre-registered rules: **R1** adapter − base > 0: +0.112 [+0.060, +0.162], pass. **R2** adapter − control > 0:
+0.026 [−0.018, +0.070], **fail**. **R3** merged − adapter ∋ 0: +0.072 [+0.040, +0.104], 65 % identical text,
**fail** (passes in fp32, item 4). **R4** not format-only: every arm uses the tag, pass. Items 1–4 are post hoc:

1. **A reward-free run gains almost as much** (control − base +8.6 pts [+3.6, +13.4]). With ratio 1 and no KL,
   shuffled advantages have zero expected gradient, yet length drifted 19 → 119 tokens: Adam turns noise into
   fixed-size steps, so the control moved its weights almost as far (‖ΔW‖ 0.85 vs 1.00), and longer reasoning
   happens to score here. One seed; the mechanism is unproven.
2. **Half the gain is hedging**: an answer, "Note: this is wrong…", a second answer; the reward reads the last.
   Graded on the first answer (what a user reads): adapter − base +0.054 [0.000, +0.108]; vs control −0.034.
3. **Worse at × ÷** (−0.093 [−0.157, −0.029]); the control regressed too (−0.107), so this is not reward-driven either.
4. **Precision changes the policy.** The update is ~0.08 % of a weight (Frobenius), below bf16's 0.2–0.39 % step:
   82 % of updated elements round back to base. Not explained by perturbation size: the random-direction adapter
   changes 6 % of outputs (ours 84 %) and its merge equals its adapter. **fp32 rerun** (`fp32_check.py`): merged ≡
   adapter (−0.004 [−0.010, 0.000], 97.6 % identical), so `merge.py` is right; but the *same* adapter scores 0.374
   in fp32 vs 0.430 in bf16 (−5.6 pts [−8.4, −3.0]). No fp32 base arm, so adapter vs model fragility is open.

**Not detected:** contamination; scorer bias shared by all arms; regressions outside Countdown; effects below
~±5 pts; sampled (T = 1) behaviour; seed variance; multi-answer gaming (found only post hoc, R4 misses it).

## 3. Silent failures (Part 4): each one finishes without an error

| stage | what breaks silently | how checked · evidence | Soup check catches it? |
|---|---|---|---|
| data | too easy (GSM8K 0.94); reward measures length (MATH 41 % truncated) | GPU pilots: pass rate, zero-var groups, truncation | no: data doctor checks template/EOS/roles |
| template | thinking mode on; instruction ignored (answer-only GSM8K 88 % truncated) | `enable_thinking=False` asserted; pilots | no |
| reward | **last tag graded: hedging pays (18 %)**; parser too strict (`\$18` → 0) | first-tag rescoring; 32 parser tests | no |
| GRPO | 72 % zero-variance groups: no gradient; 50 % of steps clipped | per-step log; shuffled control | no |
| policy | easy-item gain hides × ÷ regression | per-subset CIs | no: one aggregate |
| L2L | pinned memory 2× ledger (power-of-two rounding); prefetch not ordered after previous compute | exact-size `cudaHostRegister`; events; GPU equivalence | no |
| on-policy | rollout ≠ training log-prob (mean 0.008, max 1.6 nats): uncorrected at a few tokens | logged per step | no |
| streaming | duplicates if killed before 1st checkpoint | test (fixed); GPU kill demo | no |
| merge | **bf16 rounding swallows the delta** | `merge_report.json`; merged arm; fp32 rerun | no: `AdapterChange` never decides |
| inference | **bf16 vs fp32 flips 36 % of greedy outputs** | fp32 arms | no |

`soup ship` (read in source, not run): SHIP if tuned > base and no general regression. Its tuned > base leg
passes (0.430 > 0.318); we had no general benchmark for the other. Nothing compares to a negative control, checks
merged ≡ adapter, or rescores the answer a user reads.

## 4. Verdict: DON'T SHIP, and what must change first (in priority order)

1. **Signal:** keep only prompts with base pass rate in (0, 1) at k = 8; ≥ 3 seeds per arm; the shuffled control
   (permuted *within* groups, so it keeps main's zero-variance pattern) as a required arm; a T = 1 eval as well.
2. **Reward:** exactly one `<answer>` tag, else 0 (removes the hedging); gate on × ÷ separately. Token-level loss
   normalisation is a hypothesis to A/B, not a proven fix for the control's drift.
3. **Artifact:** ship the unmerged adapter (fp32 LoRA); gate on merged ≡ adapter at the precision users run.
4. **Soup:** make the negative control, merged ≡ adapter and a CI lower bound > 0 part of `soup ship`'s decision.

**Surprises and concerns.** The shuffled control gave the largest behaviour change and nearly the "real" gain;
without it this is a clean +11-pt success. Precision alone moves the score 6–13 pts. Concerns: one seed per arm;
greedy only; 128 tokens cut the control (0.404 is a lower bound); no general benchmark; lr never swept.

**AI tools.** Claude Code (Opus) wrote most code, ran the GPU jobs and drafted this report. Fresh-context reviews
of the engine (6 findings fixed: pinned memory 2×, a missing stream dependency, resume duplicates…) and of this
report (corrected our length-bias explanation and retry numbers; checked against the logs).
Human decisions are in git (dataset, pre-registered rules, 24 resident, the merge and fp32 checks). Evidence:
`logs/`, nvidia-smi CSVs for every run, `verify.py`, `equivalence.py`, `tests/`, numbers in `task2_l2l/results/`.
