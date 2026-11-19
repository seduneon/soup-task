# GRPO + LoRA on Qwen3-4B under an 8 GB GPU cap

## Verdict: DON'T SHIP

**The engineering works; the model does not.** Training, generation, the merge and evaluation all stayed under
8 GB on nvidia-smi (peak 7839 of 8192 MiB), our layer streaming reproduces an unstreamed model, and a killed
run resumes exactly. The trained model beats the base on held-out problems (0.43 vs 0.32), but a control
trained on shuffled rewards gains as much. Half of its gain is hedging: two answers, and the second is graded.
What it did learn from the reward is a shortcut that breaks × ÷ problems, and its score moves from 0.37 to 0.50
with numerical precision. **Root cause:** on 72 % of prompts all 8 samples got the same reward, leaving
about one useful prompt per step.

## Setup

- **Hardware:** RTX 3070 on Vast.ai, 8 GB, so the cap is the card; PCIe 4.0 at 25.5 GB/s; 31 GB RAM.
- **Training:** Qwen3-4B non-thinking, bf16, no SFT; LoRA rank 16 on all linear layers (33 M parameters); GRPO
  with 4 prompts × 8 samples per step, 100 steps, lr 2e-5, no KL term.
- **Task: Countdown-3.** Reach a target with three numbers and + − × ÷ (14, 2, 62 → 78). Reward 1 for a valid,
  exact expression in `<answer>`, else 0. 1500 generated training problems, 500 held-out. GPU pilots ruled out
  GSM8K (base 94 %, too easy) and MATH (41 % of answers cut off). Countdown answers are ~20 tokens (each token
  costs a 128 ms pass over streamed weights) and the base solves 34 %. But the pilot also showed 66 % of prompts
  with no reward variance, over our own 50 % limit. We trained anyway: the main mistake.
- **Stack: our own code** (`task2_l2l/`): the task asks for own streaming; Soup's `stream_layers` rejects
  GRPO rollouts. `transformers` supplies only the Qwen3 layers and tokenizer.

## 1. Memory and time (Parts 1 and 2a)

`task1_budget/` adds up every GPU tensor in each phase of a step. Committed before any GPU run, it was re-run
before training with the card's measured bandwidth and context. It allowed 25 resident layers; at 25 a smoke
run hit 7835 of 7840 usable MiB with silent allocator retries, so we used **24 resident, 12 streamed**.

**Predicted peak, 7410 MiB (generation):** CUDA context 233, embedding and output head 742, 24 layers 4620, two
streaming buffers 385, LoRA and Adam 378, KV cache 1008, activations and logits 44. Training swaps the KV cache for
gradients, checkpoints (mostly in host RAM) and a 519 MiB logits chunk (152 k vocabulary). No KL, so no
reference log-probs; no DFlash drafter (1025 MiB).

| | predicted | measured | why they differ |
|---|---|---|---|
| GPU peak, nvidia-smi | 7410 MiB | **7793 MiB** | allocator cache (7232 allocated, 7542 reserved), larger context |
| GPU peak, merge | 518 MiB | 661 MiB | the same two effects |
| host RAM GiB, train / merge | 10.1 / 10.6 | 5.3 / 8.4 | weights read tensor by tensor, not whole files |
| decode pass | 135 ms | 128 ms | PCIe: 12 layer copies × 7.9 ms, plus compute |
| training, forward + backward | 5.3 s | 9.1 s | small matrix multiplies run far below peak |
| step, generation share | 23.1 s, 76 % | 20.1 s, 55 % | decoding stops early; the two errors cancel |

**How a step splits:** 20 s = 11 s generating + 9 s training. Generation is PCIe-bound: each token copies the 12
streamed layers again, and only 35 % hides behind compute. Training copies each layer twice per step (all 32
sequences pass through a layer before the next), so 97 % is hidden.

## 2. Layer streaming and data streaming (Part 2)

- **L2L** (`engine.py`): weights in pinned host memory, copied one layer ahead into two GPU buffers on a separate
  CUDA stream, with events so a buffer is never overwritten in use; layer-boundary checkpoints, recomputed in the
  backward pass. One loop serves every pass.
- **Equivalence** (`equivalence.py`, `tests/`): logits bit-identical to an unstreamed run (which matches Hugging
  Face); LoRA gradients (0.6B, 4B) within 1.5× the unstreamed bf16 run's own rounding error. CPU tests repeat it
  in fp32 vs fp64, including our mixed resident/streamed setup; a layer-swapped engine fails.
- **Resume:** killed three times mid-run on the GPU; it read the same samples as an uninterrupted run, with no
  duplicates or gaps, and produced identical rollouts.
- **Merge** (`merge.py`): tensor by tensor, 15 s, 661 MiB; evaluating the merged model: 6729 MiB.
- **Dynamic residency** (the team's suggestion): extra layers on the GPU while the KV cache is small. 0.1 % faster
  [−0.4 %, +0.8 %], under our 2 % bar, so we kept regular streaming.
- **DFlash: not built** (optional). Predicted to pay off only above ~2 accepted tokens per pass, because the
  drafter would push ~10 layers off the GPU.

## 3. Did the model actually train? (Part 3)

A rising reward proves nothing: GRPO's loss is ~0 by design, and rewards can be gamed. `verify.py` compares five
versions on 500 held-out problems (greedy, paired bootstrap 95 % CIs): base, adapter, merged, a **control**
trained the same way on shuffled rewards, and a **random adapter** of equal size.

| | solved | × ÷ problems (140) | 2+ answer tags | cut off at 128 tokens | average tokens |
|---|---|---|---|---|---|
| base | 0.318 | 0.129 | 2 % | 2 % | 19 |
| adapter | 0.430 | 0.036 | 18 % | 11 % | 34 |
| shuffled control | 0.428\* | 0.107\* | 0 % | 56 % | 119 |
| merged (bf16) | 0.502 | 0.043 | 7 % | 4 % | 23 |
| random adapter | 0.320 | 0.129 | 1 % | 2 % | 19 |

\* With 12 correct `÷`/`×` answers counted; the reward rejects them (0.404 and 0.021).

**Rules set before the runs:** R1 adapter beats base: +0.112 [+0.060, +0.162], **pass**. R2 adapter beats the
control: +0.026 [−0.018, +0.070] (+0.004 counting `÷`), **fail**. R3 merged equals adapter: +0.072 [+0.040, +0.104],
**fail** in bf16, pass in fp32. R4 (not just format): pass, but uninformative.

**What we found by looking closer** (after the fact, so exploratory):

1. **Noise gives the same gain.** Shuffled rewards carry no information, yet the control's answers grew from 19 to
   119 tokens of reasoning, which helps here. Likely Adam's fixed-size steps on noise (unproven).
2. **Half the gain is hedging.** 18 % of outputs give an answer, "Note: this is wrong…", then another; the last is
   graded. On the first answer, the one a user reads, the gain halves (+0.054).
3. **What it learned from the reward is a shortcut.** It nearly stopped using × and ÷ (37 % → 13 % on problems that
   need them; `46 - 30 / 2` became `46 - (30 - 2)`). The control did not. Rarely solved, × ÷ problems gave no
   signal.
4. **Precision changes the answers.** The update is ~0.08 % of each weight, below bf16's step (0.2–0.4 %), so a bf16
   merge rounds most of it away. In fp32 the merge is exact, so `merge.py` is right, but the adapter itself scores
   0.374 in fp32 vs 0.430 in bf16.
5. **Sampled answers did not improve.** Each training prompt was new, so the training reward is an unseen-prompt
   score at temperature 1. It stayed flat (0.40 → 0.41).

**Not detected:** contamination, scorer bias shared by all versions, other-task regressions, effects under ~5
points, seed variance. It missed the hedging and the `÷` bias until we read outputs.

## 4. Silent failures (Part 4)

All finish without an error. Soup's checks (data doctor, pre-flight, `soup ship`) were read, not run.

| stage | what breaks silently | how we checked | caught by Soup? |
|---|---|---|---|
| data | too easy (GSM8K); reward measures length (MATH) | GPU pilots | no |
| template | thinking mode on; "answer only" ignored | asserted; pilots | no |
| reward | hedging pays (last answer graded); `÷` rejected | rescoring; 33 tests | no |
| GRPO | 72 % of prompts give no signal | logged per step; control | no |
| policy | +/− gain hides a × ÷ regression | scores per problem type | no: one score |
| L2L | pinned memory 2× the plan; a buffer overwritten in use | events; equivalence tests | no |
| on-policy | generation and training log-probs differ (bf16) | logged: 0.008 nats | no |
| streaming | duplicates after an early kill; a torn log line | tests; kill demo | no |
| DFlash | verifier without LoRA; drafts change the output | not built | — |
| merge | bf16 rounding erases the update | merged vs adapter; fp32 | no (ignored) |
| inference | bf16 vs fp32 changes 36 % of answers | fp32 runs | no |

`soup ship`'s main test, tuned beats base, passes here. Nothing in it compares against a negative control, checks
merged = adapter, or grades the answer a user reads.

## 5. What must change before shipping

1. **Give GRPO a signal:** train only on prompts the base sometimes solves (22 of 64 in the pilot); tune the
   learning rate (2e-5 left the update below bf16 precision); ≥ 3 seeds; evaluate sampling too.
2. **Fix the reward:** exactly one `<answer>` tag, else 0; accept `÷ ×`; score × ÷ problems separately.
3. **Ship the adapter, not the merged model**; require merged = adapter at the users' precision.
4. **In Soup:** gate `soup ship` on a negative control, merged = adapter and a CI above 0.

**What surprised me and what still concerns me.** The control trained on meaningless rewards changed the model more
than the real run and matched its score; without it, this looks like a clean +11-point success. Precision alone
moves the score 6 to 13 points. Still concerning: one seed per run, greedy evaluation only, the 128-token limit
(the control's score is a lower bound), no general eval.

**AI tools.** Claude Code (Opus) wrote most of the code, ran the GPU jobs and drafted this report. I kept its code only
after tests and GPU checks passed, and rejected its first dataset picks after pilots. My decisions are in git: the
task, the pre-registered rules, 24 resident layers, the merge, fp32 and random-adapter checks.
Independent AI reviews, checked against raw files, found six engine bugs (e.g. pinned memory 2×, a missing stream
dependency) and the scorer rejecting `÷`, which had caused a wrong claim in a draft. Least
reliable: unmeasured predictions and confident explanations.

**Where things are** (in `task2_l2l/` unless noted): logs in `/logs` and `runs/*/train_log.jsonl`; nvidia-smi every
200 ms in `runs/**/*nvsmi*.csv`; Part 3 check `verify.py`; equivalence test `equivalence.py` and `tests/`; all
numbers in `results/analysis.md`; full version in `/docs/APPENDIX.md`.
