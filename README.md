# GRPO with LoRA and streaming on 8GB GPU

Soup AI Engineer take-home (hard track): GRPO + LoRA on Qwen3-4B (non-thinking) with own layer streaming and
L2L, everything under 8 GB of GPU memory on nvidia-smi.

## Layout

| Path | What |
|---|---|
| `task1_budget/` | Part 1: memory and time budget (`budget.py`), GPU measurements (`bench.py`), committed prediction (`results/predicted.md`), pre-registered hypotheses and A/B rules (`README.md`) |
| `00-task-overview.md` | The task, the team's scope update, open questions, file index |
| `01-concepts.md` | Background: prefill/decode, KV cache, GRPO, LoRA/merge, pinned memory, streaming vs L2L, DFlash |
| `02-memory-time-budget.md` | Part 1 reasoning step by step (numbers come from `task1_budget/`) |
| `03-prior-work-soup.md` | Exact Layer Streaming paper, Soup repo, internal D2 / L2L docs, 10Cache |
| `04-plan-and-hypotheses.md` | Build order, hypothesis table, A/B methodology, silent-failure checklist |
| `DFlash.md` | DFlash notes (bonus arm) |

## Scope

- **Required:** own L2L + layer streaming, data streaming with deterministic resume, GRPO under the 8 GB cap,
  proof that the model trained, silent-failure analysis, SHIP / DON'T SHIP verdict.
- **Streaming:** regular streaming is the baseline; KV-aware dynamic residency is the research arm (A/B).
- **Optional:** DFlash rollouts (bonus A/B arm) and the merge.

## Part 1 prediction (assumed hardware, before any GPU run)

20 resident / 16 streamed layers, peak 7641 MiB (prefill) under a 7680 MiB limit, **111.8 s per GRPO step**
(83 % generation). The dynamic arm is predicted at 99.0 s (−11.5 %). Details and decision rules:
[task1_budget/README.md](task1_budget/README.md).
