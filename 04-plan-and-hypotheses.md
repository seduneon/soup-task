# 04 — Plan, hypotheses, A/B methodology, checklists

## 1. Stack decision

| Component | Choice |
|---|---|
| Layer streaming + L2L engine (train, decode, verify) | Own code: PyTorch; `transformers` only for Qwen3 layer classes, config, tokenizer |
| GRPO loop | Own (~50 lines of loss); not TRL — rollouts must come from own engine |
| DFlash (bonus) | Drafter's `trust_remote_code` model, called from own loop |
| Data streaming + resume | Own |
| Merge | Own, tensor by tensor; compare with Soup's `utils/lowmem_merge.py` |
| Silent-failure checks | Install Soup; run data doctor, pre-flight (`soup advise`), `soup ship` and report what they catch/miss |
| Baseline arm | Regular streaming (+ L2L order); dynamic KV-aware residency as the research arm (see `task1_budget/README.md`) |

## 2. Build order (each step gates the next)

1. **Before renting:** commit the part-1 budget script (`task1_budget/`, predictions) and the pre-registration (section 4 below and `task1_budget/README.md`).
2. Qwen3-0.6B (same family, tied embeddings, fits resident) locally / cheap GPU:
   - streamed forward → logits `torch.equal` vs resident;
   - backward → every LoRA grad vs resident (randomise `lora_B` first; assert copied count > 0);
   - L2L order over k micro-batches → grads vs plain GA (fp32 bit-exact; bf16 tolerance rule).
3. Streamed decode with per-layer KV cache → token-identical to HF `generate` (greedy). Then the paged KV cache for the dynamic arm → token-identical to the regular arm.
4. Data stream + deterministic resume test (kill / resume / compare sample IDs).
5. Move to Qwen3-4B under the 8 GB cap; measure bandwidth, GEMM ceiling, per-layer overhead, context size.
6. Reward function + base pass rate on candidate dataset (target 30–70%).
7. GRPO loop: few steps, then full run; log per-phase time and memory; nvidia-smi sampler running throughout. A/B regular vs dynamic streaming (H6).
8. Negative control run (shuffled rewards or lr=0).
9. Merge (optional) + merged-vs-adapter check.
10. Part 3 evaluation (held-out, CIs), part 4 checks incl. Soup tools.
11. Bonus: DFlash (greedy identity → sampling distribution check → τ vs drift → A/B step time).
12. Report + logs + "what surprised / concerns me" paragraph + AI-tools note.

## 3. Methodology rules (from Soup's internal practice)

- Write and **commit the decision rule before the first run**. Rules written after the numbers don't count.
- Run arms in **A-B-B-A** blocks on the same instance, same warm-up. Separate setup time from steady-state step time.
- Label every number: GPU, CPU, measured H2D bandwidth, SM clock, host load, n.
- An arm with a failed/void validity check → "unknown / no verdict", never "slower" or "worse".
- Correctness first, timing second: a throughput number before correctness is established measures nothing.
- Every positive needs a **control that could have failed** (e.g. a deliberately broken arm that the check catches).
- Report n and spread; prefer medians over single runs; state the resolution limit ("no difference detected at X" ≠ "no difference").
- Keep failed and withdrawn readings in the record.

## 4. Hypothesis table (pre-registration skeleton)

Fill metric, threshold and n **before** running.

| # | Hypothesis | Measurement | Decision rule (draft) | Task part |
|---|---|---|---|---|
| H1 | Generation dominates the GRPO step (predicted 83 %, plain decode) | Per-phase wall timers (cuda-synced at phase boundaries) | Median share over ≥ 5 steps ≥ 0.7 → accept | 1, 2a |
| H2 | Decode is transfer-bound: pass time ≈ linear in streamed bytes (~9 ms/streamed layer at 22 GB/s) | Vary resident layers 0…N, fit pass time vs streamed layers | Slope within ±30% of measured copy time per layer, R² > 0.9 | 1, 2a |
| H3 | Resident layers help generation far more than training (D2 saw only −2.39% on training) | A/B per phase at equal VRAM | Gen gain per layer ≥ 5× train gain per layer | 1 |
| H4 | L2L over G micro-batches cuts layer loads per step by ≈ k with grads equal to GA | Load counter; grad diff vs GA | Loads ≈ 1/k of GA; fp32 bit-exact, bf16 ≤ 1.5× GA-vs-GA floor | 2a |
| H5 | Fusing the reference pass into the policy layer visit saves one stream per step with identical log-probs | A/B step time; `torch.equal` on ref log-probs | Saving ≥ 80% of one pass time; log-probs equal | 2a |
| H6 | Dynamic (KV-aware) residency beats the best fixed plan: extra layers stay resident while the decode KV cache is short and are evicted as it grows (predicted −11.5 % step time; basis: 10Cache priority placement, applied to the decode phase it did not evaluate) | A-B-B-A regular vs dynamic, same B | Win ≥ 2 % (D2 threshold) with CI excluding 0, greedy tokens identical, nvidia-smi ≤ 8192 | 1, 2b (research) |
| H7 | Pageable host store silently serialises copies | Bandwidth + overlap (compute stream blocked time) pinned vs pageable | Blocked-time fraction materially higher for pageable | 2a |
| H8 | The adapter really learned | Held-out accuracy base vs adapter vs merged vs control, bootstrap / paired CIs | Adapter − base CI excludes 0; control − base CI includes 0; merged ≈ adapter | 3 |
| H9 | Reward gain is not gaming | Separate format-compliance and correctness metrics; inspect samples | Correctness improves, not only format | 3, 4 |
| H10 | Prediction matches measurement | Part-1 predicted vs allocated/reserved/nvidia-smi and time | Each term within stated tolerance; every gap explained | 1 |
| H11 *(bonus)* | DFlash speeds up the whole step only if batch τ > 1.77 (> 2.06 to beat the dynamic arm), because the drafter costs 9 resident layers | τ per pass and per batch at the sampling temperature; A/B step time | Step gain ≥ 2 % with CI excluding 0 and on-policy checks pass | 2c |
| H12 *(bonus)* | DFlash acceptance falls as policy drifts | τ vs KL(policy ‖ base) over training | Negative correlation, report slope | 2c |
| H13 *(bonus)* | Greedy DFlash is token-identical; sampling preserves distribution | Token compare; χ²/KS on next-token / sequence stats vs plain sampling | 100% identical greedy; test not rejected at chosen α with enough samples to have power | 2c |

## 5. Part 3 — "did it really train?" design notes

- Arms: base, adapter-attached, merged, negative control (shuffled rewards or lr=0), same seeds/decoding.
- Held-out set never seen in training; fixed; large enough for the CI you need (state the resolution).
- Paired comparisons per item; bootstrap CIs.
- Adapter liveness: assert adapter delta norm > 0; outputs differ from base on a fixed probe set.
- Merged vs adapter: max logit diff within justified bf16 tolerance.
- What the check would NOT detect: e.g. learning a dataset-specific shortcut that also exists in the held-out split; regressions outside the task (general capability); format overfitting if the scorer is lenient; small effects below the resolution limit.

## 6. Part 4 — silent-failure checklist (start list)

| Stage | What can silently break | How to check |
|---|---|---|
| Data | duplicates between train and held-out; answer leakage in prompt | hash overlap; grep |
| Tokenizer/template | thinking mode on; different template in rollout vs train vs eval; missing EOS | decode and diff rendered prompts; assert `enable_thinking=False` |
| Reward | parser too lenient / always 1; fenced output not parsed | unit tests with adversarial outputs; reward on known-wrong answers |
| GRPO | prompt tokens in loss; zero-variance groups; wrong sign; ratio ≠ 1 on-policy | mask audit; % zero-advantage groups; sign test on toy |
| Layer streaming / L2L | severed graph (detach/no_grad); aliasing pooled buffers; wrong layer order; pageable fallback | grad-vs-resident; owner tripwire; `is_pinned()`; blocked-time |
| Data streaming | duplicate/skip on resume; shuffle RNG not restored | kill/resume ID log diff |
| DFlash | wrong hidden-state layer index; verifier without LoRA; KV not cropped; distribution bias | compare features to HF; greedy identity; distribution test |
| Merge | adapter keys mismatch → merged = base; wrong α/r scaling | merged vs adapter logits; merged vs base must differ |
| Inference/eval | eval loads base instead of merged; different decoding params | checksum loaded weights; log decoding config |
| Memory cap | allocator limit ≠ nvidia-smi; spill not visible | nvidia-smi sampler for whole run |

For each: does Soup's data doctor / pre-flight / `soup ship` catch it? Record evidence either way.

## 7. Day plan (5 days, ~12–16 h)

- **Day 1:** budget script + pre-registration committed; send questions to the team; streamed forward/backward + equivalence on 0.6B.
- **Day 2:** L2L order + GA equivalence; streamed decode with KV; data resume test.
- **Day 3:** rent GPU; measurements (bandwidth, ceiling, overhead, context); reward + dataset; first GRPO steps under cap.
- **Day 4:** full run + negative control; merge (optional); part 3 eval; A/B for H1–H7 (incl. regular vs dynamic streaming).
- **Day 5:** part 4 checks incl. Soup tools; DFlash bonus if time; report and deliverables.

## 8. Live-discussion prep

- How did you prove training happened? What does your check miss?
- Do you trust tok/s, acceptance length, VRAM? (Same-session ceiling, clock recorded, nvidia-smi vs allocator.)
- Where is the bottleneck, and was the prediction right?
- Did DFlash really speed up the step (not just decode)?
- Why is training compute-bound in the paper but rollouts transfer-bound here?
- What would you change in Soup so this failure can't pass `soup ship` again?
- 3 days on H100s: what first?
