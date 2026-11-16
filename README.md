# GRPO with LoRA and streaming on 8GB GPU

## Memory and time budget (before run)

Estimate GPU and CPU-RAM use for the GRPO step and for the merge: weights,
per-layer buffers, activation checkpoints, logits (mind the ~151k vocab), rollout KV cache, reference log-probs, LoRA and optimizer states, the
drafter, and the peak during the merge. Decide how many layers stay resident on the GPU and how many are streamed, and justify it. Also
estimate step time from PCIe bandwidth for both training and generation. Show the calculation, compare with the actual peak (allocated,
reserved, nvidia-smi) and measured time, and explain the gaps.
