# GRPO with LoRA and streaming on 8GB GPU

Soup AI Engineer take-home (hard track): GRPO + LoRA on Qwen3-4B (non-thinking) with own layer streaming and
L2L, everything under 8 GB of GPU memory on nvidia-smi.

## Layout

| Path | What |
|---|---|
| `task1_budget/` | Part 1: memory and time budget (`budget.py`), GPU measurements (`bench.py`), committed prediction (`results/predicted.md`), pre-registered hypotheses and A/B rules (`README.md`) |
| `task2_l2l/` | Part 2/3: streamed L2L engine, GRPO + LoRA loop, data streaming with deterministic resume, merge, verification (`README.md` has the run commands) |
| `docs/00-task-overview.md` | The task, the team's scope update, open questions, file index |
| `docs/01-concepts.md` | Background: prefill/decode, KV cache, GRPO, LoRA/merge, pinned memory, streaming vs L2L, DFlash |
| `docs/02-memory-time-budget.md` | Part 1 reasoning step by step (numbers come from `task1_budget/`) |
| `docs/03-prior-work-soup.md` | Exact Layer Streaming paper, Soup repo, internal D2 / L2L docs, 10Cache |
| `docs/04-plan-and-hypotheses.md` | Build order, hypothesis table, A/B methodology, silent-failure checklist |
| `docs/DFlash.md` | DFlash notes (bonus arm) |
| `docs/team/` | Task PDF and the team's internal docs (dynamic streaming, L2L context; Russian) |

## Scope

- **Required:** own L2L + layer streaming, data streaming with deterministic resume, GRPO under the 8 GB cap,
  proof that the model trained, silent-failure analysis, SHIP / DON'T SHIP verdict.
- **Streaming:** regular streaming is the baseline; KV-aware dynamic residency is the research arm (A/B).
- **Optional:** DFlash rollouts (bonus A/B arm) and the merge.

## Part 1 prediction (assumed hardware, before any GPU run)

20 resident / 16 streamed layers, peak 7641 MiB (prefill) under a 7680 MiB limit, **111.8 s per GRPO step**
(83 % generation). The dynamic arm is predicted at 99.0 s (−11.5 %). Details and decision rules:
[task1_budget/README.md](task1_budget/README.md).
