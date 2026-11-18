# Status and next steps (2026-11-18, 11:55 UTC)

Start here in a new session. Decisions and results are in `task2_l2l/README.md` (dataset section) and the git log.

## Running on the Vast RTX 3070 box (tmux, survives closing the AI session)

| tmux session | What | Log (box: `/workspace/soup-task/logs/`) | Output (`/workspace/soup-task/task2_l2l/`) |
|---|---|---|---|
| `train` | main run, 100 steps (started 11:44, ~35 min), then the shuffled-reward control, 100 steps (~35 min) | `main.log`, `control.log` | `runs/main/`, `runs/control/` |
| `verify` | waits for `EXIT_CODE=` in `control.log`, then base vs adapter vs control, greedy, 500 held-out (~15 min) | `verify.log` | `runs/verify/verify_report.json` |

Each log ends with `EXIT_CODE=<n>` when its job finishes. Expected all done around 13:15 UTC.
Check: `ssh … 'tail -3 /workspace/soup-task/logs/{main,control,verify}.log'`.
The box's disk is NOT persistent across destroy: pull `runs/` and `logs/` to the Mac before destroying it:
`rsync -az -e "ssh -p PORT" root@HOST:/workspace/soup-task/task2_l2l/runs/ task2_l2l/runs/`, same for `logs/`.

## Settled decisions
- Dataset: **Countdown-3** (`data/countdown3`, `reward: countdown`, `prompt_style: plain`, `max_new: 128`).
  Why, plus the rejected options (GSM8K CoT too easy, MATH 3–5 length-confounded, GSM8K answer-only ~10 %,
  DAPO-Math needs 1k+ tokens): `task2_l2l/README.md`. Documented deviation: zero-variance groups 0.66 > 0.5.
- 24 resident layers, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (default in train.py/evaluate.py).
- Merge and DFlash are optional per the team and are **skipped**: no merged arm in verify.
- transformers pinned to 4.52.3 (`requirements.txt`).
- evaluate/verify for Countdown always need `--prompt-style plain --reward countdown --max-new 128`
  (an adapter trained with another prompt style aborts by design).

## Next steps, in order
1. Pull `runs/` and `logs/` from the box (above). Commit small evidence: summaries, train_log.jsonl,
   verify_report.json, nvidia-smi CSVs. `logs/` and `runs/` are gitignored, so use `git add -f` on the chosen files.
2. **Part 3:** read `runs/verify/verify_report.json`. Compare adapter vs base AND adapter vs control directly
   (Spurious Rewards, arXiv 2506.10947: random rewards can also help Qwen models). Split +/- items from */÷ items
   (`needs_muldiv` field in the data). Report format compliance (`has_box`) separately from accuracy.
   Plot the reward curve from `train_log.jsonl` against the control's.
3. **Part 1 comparison:** predicted vs measured, from `task1_budget/results/countdown_measured_hw.md` and
   train_log `phases`.
   - memory: predicted 7603 MiB at 25 resident, measured 7835 (+232 MiB);
   - time: predicted 22.2 s/step, measured ~23 s mean;
   - generation vs train split: per-phase `s` in train_log.
   - Also re-measure the CUDA context with expandable_segments: `bench.py ctx` rewrites `measured_hw.yaml`,
     so back it up first.
4. **Part 2b:** deterministic resume on the GPU. Kill main mid-run with `L2L_KILL_AT=<step>:rollout`, resume,
   compare `samples.jsonl` IDs. CPU tests already pass; one GPU demo is needed.
5. **Part 4:** silent-failure table. Findings so far:
   - pinned-memory 2× ledger;
   - transformers 5.x breaking change;
   - allocator retries that only show on stderr;
   - CUDA total 7840 vs 8192;
   - CoT length confound on MATH;
   - instruction non-compliance on GSM8K answer-only;
   - spurious-reward risk;
   - `verify.py` cache reuse across prompt styles (now guarded).
6. **Part 5:** SHIP / DON'T SHIP verdict. Write the ≤3-page report, the surprises/concerns paragraph, and the
   AI-tools note (Claude Code wrote and ran most code; human decisions and corrections are in the git history).

## Not done / open
- Overlap evidence (`overlap` in train_log), residency sweep (H2/H3), regular-vs-dynamic A/B (H6): optional
  research arms, only if time allows.
- `logs/` gitignored: raw logs and nvidia-smi samples are deliverables, so they must be force-added or the ignore changed.
