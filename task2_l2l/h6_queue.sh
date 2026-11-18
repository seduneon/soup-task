#!/bin/bash
# H6 A/B (pre-registered in task1_budget/README.md): dynamic vs regular residency, A-B-B-A blocks on one instance.
# Same seed, same prompts (fresh out_dir => same first 8 steps), 8 steps per block, first 2 discarded (warm-up).
# Rule: adopt dynamic only if step time gain >= 2 % with CI excluding 0, rollouts identical, nvidia-smi <= cap.
source /venv/main/bin/activate
cd /workspace/soup-task/task2_l2l
until grep -q EXIT_CODE /workspace/soup-task/logs/verify_fp32.log; do sleep 30; done   # GPU free first
for b in reg1:regular dyn1:dynamic dyn2:dynamic reg2:regular; do
  name=${b%%:*}; mode=${b##*:}
  date -u; echo "== block $name ($mode)"
  python train.py --set steps=8 streaming=$mode profile_overlap=false out_dir=runs/h6/$name
  echo "block $name exit $?"
done
date -u
