# Status and next steps (2026-11-18, 12:15 UTC)

Start here in a new session. Decisions and results are in `task2_l2l/README.md` (dataset section) and the git log.

## Running on the Vast RTX 3070 box (tmux, survives closing the AI session)

| tmux session | What | Log (box: `/workspace/soup-task/logs/`) | Output (`/workspace/soup-task/task2_l2l/`) |
|---|---|---|---|
| `train` | main run, 100 steps (started 11:44, ~35 min), then the shuffled-reward control, 100 steps (~35 min) | `main.log`, `control.log` | `runs/main/`, `runs/control/` |
| `verify` | waits for `EXIT_CODE=` in `control.log`, then base vs adapter vs control, greedy, 500 held-out (~15 min) | `verify.log` | `runs/verify/verify_report.json` |

| `post` | `task2_l2l/post_queue.sh`: waits for `EXIT_CODE=` in `verify.log`, then (1) merge → `/dev/shm/merged` (root disk has 5 GB free, merged bf16 is ~8 GB; report copied to `runs/merge/`), (2) verify again with `--merged` (other arms reused), (3) GPU kill/resume demo `runs/resume_demo` (SIGKILL at 3:train, 6:rollout, 8:after_ckpt, 10 steps, same seed as main), (4) `bench.py ctx` with expandable_segments → `task1_budget/results/measured_hw_ctx_expandable.yaml` (original yaml restored) | `merge.log`, `verify_merged.log`, `resume_demo.log`, `bench_ctx_expandable.log`; `post_queue.done` at the end | `runs/merge/`, `runs/verify/`, `runs/resume_demo/` |

Each log ends with `EXIT_CODE=<n>` when its job finishes. main/control/verify expected done ~13:10 UTC, post queue ~13:40 UTC.
A background waiter in the AI session watches main/control/verify markers (lost if that session dies: just check the logs).
Merge is optional per the team but kept (decided 2026-11-18 with the user): Part 3 asks for the merged arm and
`verify.py` gives "NO VERDICT (missing arm)" without it. RAM check: 31 GB box, /dev/shm 7.7 GB + merge peak
~10.5 GiB (budget) ≈ 19 GB. If merge fails, report it as attempted and why.
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
1. Pull `runs/` and `logs/` from the box (above; skip `checkpoint.pt`, keep `rollouts/` of main and resume_demo
   for the resume comparison). Commit small evidence: summaries, train_log.jsonl, samples.jsonl, attempts.jsonl,
   verify_report.json, merge_report.json, nvidia-smi CSVs. `logs/` and `runs/` are gitignored, so use `git add -f`.
   Then run `cd task2_l2l && python3 analyze.py` → `results/analysis.md`, `analysis.json`, `reward_curves.png`
   (Part 3 table split +/- vs */÷, format vs correctness, LoRA ‖ΔW‖ main vs control, Part 1 measured vs
   predicted, Part 2b resume vs main). Tested on a partial snapshot of main.
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
   - `verify.py` cache reuse across prompt styles (now guarded);
   - rollout vs training log-prob mismatch (bf16 decode vs recompute): mean 0.008 nats, max 1.6 nats per token
     (`rollout_vs_train_*` in train_log); harmless for ratio = 1 single-update GRPO, would bias PPO-clip / KL;
   - decode stops when the longest sequence ends: prediction assumed C = 128 passes, measured mean ~72;
     one rambling completion sets the whole batch's step time.
   - Soup's checks (read from public source, MakazhanAlpamys/Soup @ 7d33145): `soup data doctor` = chat template,
     template render, generation markers, EOS in labels, BOS duplication, system/unknown roles, input truncation
     (`utils/data_doctor.py` CHECKS). No train/held-out leakage, reward, or generation-length check.
     `soup ship` (`utils/ship_verdict.py::decide_ship`): SHIP ⇔ tuned > base (strict, point estimate unless a
     noise floor is configured) AND no general-benchmark regression. No negative control, no merged-vs-adapter
     check; `AdapterChange` warns below 1 % changed elements but "never part of the decision".
     `soup advise` = pre-flight ROI/memory on the data file.
6. **Part 5:** SHIP / DON'T SHIP verdict. Write the ≤3-page report, the surprises/concerns paragraph, and the
   AI-tools note (Claude Code wrote and ran most code; human decisions and corrections are in the git history).

## Not done / open
- Overlap evidence (`overlap` in train_log), residency sweep (H2/H3), regular-vs-dynamic A/B (H6): optional
  research arms, only if time allows.
- `logs/` gitignored: raw logs and nvidia-smi samples are deliverables, so they must be force-added or the ignore changed.
