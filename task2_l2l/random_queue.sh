#!/bin/bash
# Random-direction control (random_adapter.py): attached and merged, same eval settings as verify.py.
source /venv/main/bin/activate
L=../logs
COMMON="--data data/countdown3/heldout --n 500 --k 1 --temperature 0.0 --max-new 128 --reward countdown \
  --prompt-style plain --device cuda --dtype bfloat16 --resident 24 --batch 16 --seed 0"
python random_adapter.py --adapter runs/main/adapter.pt --out runs/random/adapter.pt --seed 0 > $L/random.log 2>&1 &&
python evaluate.py --model Qwen/Qwen3-4B --adapter runs/random/adapter.pt --out runs/verify/random_adapter $COMMON >> $L/random.log 2>&1 &&
rm -rf /dev/shm/merged &&
python merge.py --base Qwen/Qwen3-4B --adapter runs/random/adapter.pt --out /dev/shm/merged_random --device cuda >> $L/random.log 2>&1 &&
cp /dev/shm/merged_random/merge_report.json runs/random/merge_report.json &&
python evaluate.py --model /dev/shm/merged_random --out runs/verify/random_merged $COMMON >> $L/random.log 2>&1
echo EXIT_CODE=$? >> $L/random.log
