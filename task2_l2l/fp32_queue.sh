#!/bin/bash
# R3 follow-up: is merged != adapter caused by the bf16 cast? Same 500 items and settings as runs/verify, fp32 weights.
# Ran on the Vast box 2026-11-18 17:50-18:53 UTC in tmux; log: logs/verify_fp32.log. Analysis: fp32_check.py.
source /venv/main/bin/activate
cd /workspace/soup-task/task2_l2l
C="--model Qwen/Qwen3-4B --adapter runs/main/adapter.pt --data data/countdown3/heldout --n 500 --k 1 --temperature 0.0 --max-new 128 --reward countdown --prompt-style plain --device cuda --dtype float32 --resident 8 --batch 16 --seed 0"
date -u; python evaluate.py $C --out runs/verify_fp32/adapter || exit 1
date -u; python evaluate.py $C --merge-in-memory --out runs/verify_fp32/merged || exit 1
date -u
