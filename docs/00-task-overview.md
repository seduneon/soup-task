# 00 — Task overview (Soup AI Engineer take-home, hard track)

Personal reference notes. Sources: the task PDF (`team/Soup-Task-01-AI-Engineer-v5.pdf`), the Exact Layer Streaming paper v3 (Makazhan, Aug 2026), the follow-up message from the team, and three internal docs they shared (dynamic streaming ×2, L2L context).

## The task in one paragraph

Train **Qwen/Qwen3-4B (non-thinking mode)** with **GRPO + LoRA** (no SFT stage) on a task with an automatic, verifiable reward (exact-match math, JSON validity, unit-tested code, constraint following), 1–2k prompts. Everything — training, rollout generation, merge, evaluation — must stay **≤ 8 GB GPU memory as shown by nvidia-smi** (CUDA context included). bf16 weights alone are ~8 GB, so the model must be **streamed**. Then decide: **SHIP or DON'T SHIP**.

- Time: 12–16 h of work, 5-day deadline.
- Hardware: rented Ampere+ GPU on Vast.ai (RTX 3060 / 3060 Ti / 3070), host RAM ≥ 16 GB, credit provided.
- If the card has > 8 GB (RTX 3060 = 12 GB), enforce the cap yourself and **prove it with nvidia-smi**; an allocator limit is not a cap.
- No quantization required.

## The five parts

1. **Memory and time budget** — predict GPU + CPU RAM per term and step time from PCIe bandwidth *before* running; then compare with measured allocated / reserved / nvidia-smi and time; explain the gaps. Decide how many layers stay resident. → code in `task1_budget/`, reasoning in `02-memory-time-budget.md`.
2. **Make it fit**
   - (a) **L2L** — own implementation: weights in pinned CPU memory, current + prefetched layer on GPU, checkpoint at layer boundaries, copy/compute overlap on a separate CUDA stream; also for generation and eval. **Prove equivalence** (logits + gradients vs a non-L2L reference on a small config, justified tolerance). Report how a GRPO step splits between generation and training, and why.
   - (b) **Streaming** — prompts/rollout data streamed from disk with **deterministic resume** (kill mid-epoch, resume, same samples, no duplicates, no gaps). Merge (optional per the team) tensor-by-tensor / shard-by-shard; show the peak.
   - (c) **DFlash rollouts (bonus per the team)** — drafter `z-lab/Qwen3-4B-DFlash-b16` (~1 GB, resident) integrated into own L2L loop. Report acceptance length and step time with/without. Show on-policy: verifier uses current LoRA, greedy is token-identical to plain decoding, distribution check for sampling. Measure acceptance vs policy drift.
3. **Prove the model actually trained** — rising reward proves nothing (GRPO loss ≈ 0 by construction; reward can be gamed). Compare base / adapter-attached / merged (must match adapter up to noise) / negative control (shuffled rewards or lr=0). Held-out set, confidence intervals. State what the check would NOT detect.
4. **Silent failures** — across data → tokenizer/template → reward → GRPO → L2L → streaming → DFlash → merge → inference. For each: what can go wrong, how checked, whether Soup's checks (data doctor, pre-flight, `soup ship`) catch it, evidence.
5. **Verdict** — one call, SHIP or DON'T SHIP, backed by evidence. If DON'T SHIP: exactly what must change.

## Deliverables

Report (≤ 3 pages) · raw timestamped logs · raw nvidia-smi samples for the whole run · part-3 verification script · L2L equivalence test · finished notebook/code · one paragraph: what surprised you, what still concerns you. Note where AI tools were used, what was accepted, verified, changed. Don't hide failed results — a clear negative beats a quiet success.

## Scoring

| Area | Weight |
|---|---|
| Proving the model is real | 25% |
| Silent failures | 20% |
| Memory, L2L, streaming, DFlash engineering | 30% |
| Live discussion (40 min, after submission) | 25% |

Likely live questions: how did you prove training happened; what does your check miss; do you trust tok/s, acceptance length, VRAM; where is the L2L bottleneck and was the prediction right; did DFlash really speed up the step; what would you do with 3 days of H100s; what would you change in Soup so this failure can't pass `soup ship` again.

## Scope update from the team (follow-up message)

- **DFlash is optional** — a bonus for A/B testing and extra research.
- **Merge is optional** (still cheap; part 3 references the merged model, so probably do it).
- Choose the streaming type yourself: **regular** (Soup's `stream_layers`) or **dynamic** (D2). For dynamic, use **10Cache, arXiv 2511.14124** as the basis.
  - **Decision:** regular streaming is the baseline; **KV-aware dynamic residency** (extra layers resident while the decode KV cache is short, evicted as it grows) is the research arm, adopted only if it wins ≥ 2 % in A/B. Reasoning in `task1_budget/README.md` and `03-prior-work-soup.md` §E.
- What they really want to see: **how you think, how you form hypotheses, how you run A/B, and whether you have enough hypotheses to close all research questions.**
- Going beyond the task is welcome; documenting all improvements is a plus.
- Questions are welcome; not everything working is OK.
- After the task: accelerated internship (< 1 month), then offer decision. Pay is reviewed based on activity and contribution. Two projects: B2C (their product) and B2B (datacenter optimization).

## Open questions to send the team

1. Should "L2L" be read in Soup's internal sense (layer-major over k micro-batches, equal to gradient accumulation) on top of my own layer streaming, or is the task-text definition (streaming + checkpointing + overlap) enough?
2. May I use Soup's `stream_layers` as an A/B reference arm, given the task says implement my own?
3. How much Vast.ai credit, and should I confirm the budget before renting?
4. The doc links point to private `SoupTeam/soup-rnd` — will I get read access (D2 profile code, `benchmarks/METHOD.md`)?
5. Is the report still ≤ 3 pages, with extra documentation in an appendix?

## File index

- `00-task-overview.md` — this file
- `01-concepts.md` — transformers, prefill/decode, KV cache, GRPO, LoRA/merge, pinned memory, layer streaming vs L2L, DFlash
- `02-memory-time-budget.md` — part 1 broken into steps, with formulas and Qwen3-4B numbers
- `03-prior-work-soup.md` — the paper's lessons and traps, Soup repo findings, internal D2 / L2L docs, 10Cache
- `04-plan-and-hypotheses.md` — build order, hypothesis table, A/B methodology, checklists
- `../task1_budget/` — Part 1 code: `budget.py` (ledger, time model, arms), `bench.py` (GPU measurements), `config.yaml`, `results/predicted.md`, README with the pre-registered hypotheses
- `DFlash.md` — DFlash notes
- `team/` — the task PDF and the team's internal docs (Russian): `dynamic_stream.md`, `dynamic_stream_2.md`, `l2l_context.md`
- `../task2_l2l/` — Part 2 code: streamed L2L engine, GRPO loop, data streaming with resume, merge, Part 3 verification
