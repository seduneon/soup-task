#!/bin/bash
# GPU jobs after verify (run from task2_l2l/ in tmux). Each job appends EXIT_CODE=<n> to its own log.
# 1. merge into /dev/shm (root disk has 5 GB free, the merged bf16 model is ~8 GB); 2. merged arm of verify
# (base/adapter/control summaries are reused); 3. kill/resume demo on the GPU vs the main run;
# 4. CUDA context re-measured with expandable_segments.
source /venv/main/bin/activate
L=../logs
until grep -q EXIT_CODE= $L/verify.log 2>/dev/null; do sleep 30; done

python merge.py --base Qwen/Qwen3-4B --adapter runs/main/adapter.pt --out /dev/shm/merged --device cuda > $L/merge.log 2>&1
echo EXIT_CODE=$? >> $L/merge.log
mkdir -p runs/merge && cp /dev/shm/merged/merge_report.json /dev/shm/merged/merge_nvsmi*.csv runs/merge/ 2>/dev/null

python verify.py --model Qwen/Qwen3-4B --adapter runs/main/adapter.pt --merged /dev/shm/merged \
  --control runs/control/adapter.pt --data data/countdown3/heldout --n 500 --prompt-style plain \
  --reward countdown --max-new 128 --resident 24 --out runs/verify > $L/verify_merged.log 2>&1
echo EXIT_CODE=$? >> $L/verify_merged.log

# Kill/resume: SIGKILL at 3 points (after rollout, after train, after checkpoint), resume each time.
# Same seed/config as main, so steps 0-9 must reproduce main's sample ledger and rollouts.
for i in 1 2 3 4 5 6; do
  L2L_KILL_AT=3:train,6:rollout,8:after_ckpt python train.py --set steps=10 out_dir=runs/resume_demo >> $L/resume_demo.log 2>&1
  rc=$?; echo "attempt $i rc=$rc" >> $L/resume_demo.log
  [ $rc -eq 0 ] && break
done
echo EXIT_CODE=$rc >> $L/resume_demo.log

cp ../task1_budget/results/measured_hw.yaml ../task1_budget/results/measured_hw.yaml.bak
(cd ../task1_budget && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python bench.py ctx) > $L/bench_ctx_expandable.log 2>&1
echo EXIT_CODE=$? >> $L/bench_ctx_expandable.log
cp ../task1_budget/results/measured_hw.yaml ../task1_budget/results/measured_hw_ctx_expandable.yaml
mv ../task1_budget/results/measured_hw.yaml.bak ../task1_budget/results/measured_hw.yaml
echo ALL_DONE >> $L/post_queue.done
