# Part 2 — open items

State on 2026-11-17: everything that runs without a CUDA GPU is built and tested (87 CPU tests pass;
train → evaluate → merge smoke-run on Apple MPS). Nothing in `task2_l2l/` is committed yet.

## Triage of 2026-11-17 (what was done now, what waits and why)

| Item | Decision | Status |
|---|---|---|
| Harder dataset (MATH L3–5) | do now: Qwen3-4B very likely > 70 % on GSM8K, so the pre-registered rule will probably reject it; building the fallback on the GPU would waste paid time | **done**: `data/math35` (1500/500, numeric gold only, 20 % dropped, 7 test-overlap questions excluded); parser fixed for `\frac12`, `-\frac{1}{14}`; GSM8K output verified byte-identical |
| Part 3 verification script | do now: required deliverable, fully CPU-testable | **done**: `verify.py` (4 arms, paired CIs, rule R1–R4 fixed before runs, limits stated) + 7 tests |
| Negative control | do now: `verify.py` needs it; lr = 0 is vacuous here (B = 0 at init → identical to base) | **done**: `reward_shuffle` (seeded permutation per step, exact resume, true reward still logged) + test |
| Copy/compute overlap evidence | do now (instrumentation only): the GPU run must produce the evidence without another code round | **done**: `engine.profile` / `overlap_report()` → `overlap` in `train_log.jsonl`; CUDA-only, so unverified until the GPU |
| Backward re-copy of the last layer | defer: perf only, 2 of 16 copies; measure first (H1) | open |
| `repeat_kv` transient | defer: changes attention numerics/path; only if the measured decode peak needs it | open |
| LR | defer: needs real rollouts | open |
| DFlash | defer: bonus; needs the 1 GB drafter download and a GPU to mean anything | open — ask before starting |
| Part 4 / 5 / report | defer: need GPU evidence; Part 4 rows for what is already checked can be drafted any time | open |

## 1. Code paths no CPU test can reach

| Item | Why it is open | First check on the GPU |
|---|---|---|
| Copy stream, events, non-blocking H2D in `engine.visit()` | `self.cuda` branches are dead on CPU and MPS | `equivalence.py --model Qwen/Qwen3-0.6B --layers 28 --device cuda --dtype bfloat16 --resident 0` |
| Exact-size pinning (`cudaHostRegister`), release of resident host copies | CUDA only | `store.pin` asserts `is_pinned()`; watch host RSS in `train_log.jsonl` |
| Checkpoint arena offload with non-blocking D2H/H2D | CPU test covers the logic, not the concurrency | grads in `equivalence.py` (offload is on by default on CUDA) |
| `memory_plan` (dynamic arm) | uses `mem_get_info` / allocator state; the CPU test uses a synthetic schedule | `train.py --set steps=2 streaming=dynamic out_dir=runs/smoke_dyn`, check `resident_min_max` and `over_cap` |
| Overlap profiling (`overlap_report`) | CUDA events only | smoke run: `copies` == `layer_copies`, `h2d_gbps` close to `bench.py pcie`; compare step time with `profile_overlap=false` once (event overhead) |

## 2. Measurements (need the rented GPU)

- `task1_budget/bench.py` → measured H2D, VRAM bandwidth, TFLOPS, launch overhead, context size.
- Per-phase memory (allocated / reserved / nvidia-smi) vs the Part 1 ledger (H10); fill the tables in
  `task1_budget/README.md`.
- Generation / training split of a step (H1), decode PCIe-bound (H2: `stall_ms` vs `copy_ms` in decode),
  residency sweep (H3).
- Regular vs dynamic A/B, A-B-B-A blocks (H6).
- Merge GPU and host peak on 4B.
- Cap evidence: nvidia-smi device CSV + per-process `.apps.csv` for train, eval and merge.
- Part 3: main run, `reward_shuffle` control run (same steps), merge, `verify.py` on the chosen held-out set.

## 3. Decisions pending a measurement

- **Dataset.** Rule: keep a dataset only if the sampled base pass rate on its train split (`evaluate.py --k 8
  --temperature 1`) is 30–70 % and fewer than half the groups have zero reward variance. Measure GSM8K and
  math35; prefer GSM8K if it passes, else math35; if neither passes, report it and pick the closer one.
- **Learning rate** (`2e-5` in `train_config.yaml`): tune on the first steps.
- **`repeat_kv` transient** (~200 MiB in decode at B = 16, full length): keep HF SDPA unless the measured
  decode peak needs it; the alternative is SDPA with `enable_gqa=True`.

## 4. Known, deliberately not fixed yet

- Backward re-copies the last layer the forward just used (2 of 16 copies per train step). Perf only; measure
  first.
- Expected on the GPU, not bugs: after kill/resume the final LoRA may differ in the last bits (non-deterministic
  SDPA backward) while rollouts stay identical; in bf16 a few of the ~400 per-tensor ratios may exceed 1.5
  while the aggregate passes (look at `worst_tensors`).
- math35 keeps only numeric answers: the set is biased towards numeric problems (report it). MATH test is
  public and Qwen3 may have seen it (contamination; listed in `verify.py` as not detected).

## 5. Not started

- DFlash rollouts (bonus): drafter resident, hidden features from the target layers, greedy identity,
  sampling distribution check, acceptance vs policy drift, step-time A/B (H11–H13).
- Part 4 silent-failure table, Part 5 verdict, report, "what surprised me" paragraph, AI-tools note.
- Commit and push of `task2_l2l/`.
