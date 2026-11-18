# H6: dynamic vs regular residency (A-B-B-A, steps 2-7 of 8 per block)

| block | mean step s | resident min-max (decode) | nvidia-smi peak MiB |
|---|---|---|---|
| reg1 | 19.61 | [24, 24] | 7791 |
| dyn1 | 19.59 | [25, 25] | 7839 |
| dyn2 | 19.48 | [25, 25] | 7839 |
| reg2 | 19.49 | [24, 24] | 7791 |

- step-time saving of dynamic: **+0.1%** [-0.4%, +0.8%] over 12 paired steps
- rollout-time saving: -0.4% [-0.7%, +0.1%]
- rollouts identical across all blocks (tokens, prompts, rewards): True
- rule (gain >= 2 %, CI excl. 0, identical, <= 7840 MiB): **keep regular**
