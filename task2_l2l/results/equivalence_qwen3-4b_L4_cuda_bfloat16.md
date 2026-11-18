# L2L equivalence: Qwen/Qwen3-4B, 4 layers, cuda, bfloat16

B=4, S=42, 46 loss tokens, chunk 16, LoRA r=16 on 7 linears, 0 resident / 4 streamed, truth torch.float32. Rule: err(stream) ≤ 1.5 × err(ref) vs truth, per tensor and in aggregate.

- resident engine vs HF logits: max |Δ| 0.00e+00, argmax agree 1.0000
- streamed vs reference logits: bit-exact **True**, max |Δ| 0.00e+00

| run | pass | bit-exact | err stream | err ref | worst ratio | all grads nonzero | copies/pass |
|---|---|---|---|---|---|---|---|
| full batch | **True** | False | 1.65e-02 | 1.63e-02 | 1.20 | True | 4 of 4 |
| mb=1 | **True** | False | 1.64e-02 | 1.60e-02 | 1.26 | True | 4 of 4 |
