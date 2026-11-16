# Claude explains DFlash

DFlash does use a second, small model, so I put that badly last time. It's just not an independent model. I read DFlash's actual code (`dflash.py` in the drafter repo), so the explanation below follows what it really does.

## DFlash

**There are two models:**
- **Target:** Qwen3-4B, the big model you're training with LoRA.
- **Drafter:** `z-lab/Qwen3-4B-DFlash-b16`, 537M params (5 layers), kept on the GPU.

**What makes the drafter special:** it can't run on its own. It has no embedding table or output head of its own. The drafter reuses the target's embedding and LM head. Its main input is also not text but the target's internal hidden states. Think of it as a small extra module that plugs into the big model rather than a separate model.

### One round of DFlash

Say the text so far is "The answer is" and the last confirmed token is "is".

**Step 1: the target's pass gives the drafter its context.** Whenever the target runs (on the prompt, or during verification), you save the hidden states at layers 1, 9, 17, 25 and 33. They're glued together and squeezed to one vector per token: `fc` maps `len(target_layer_ids) * hidden_size` to `hidden_size`. These vectors summarize what the big model "understood" about each previous token.

**Step 2: the drafter fills in a whole block at once.** It sets up 16 slots: The block's first slot holds the last confirmed token. The remaining `block_size - 1` slots are masks to be filled:

```
["is"] [MASK] [MASK] [MASK] ... [MASK]     ← 16 slots
```

In one forward pass, the drafter predicts all 15 masked slots at the same time. This is the "block diffusion" part: fill in the blanks in parallel instead of one token after another. Inside its attention, queries come only from the noise block, while keys and values combine the target's context features with the block itself. So each blank can look at the big model's understanding of the past plus the other blanks.

```
draft: ["is"] ["4"] ["2"] ["."] ["\n"] ["So"] ...
```

**Step 3: the target checks all 16 in one pass.** This is the expensive part, and under L2L it means one stream of all 36 layers. The target computes what it would have produced at each position, given the draft tokens before it.

**Step 4: keep the matching prefix.** The code counts how many draft tokens agree with the target, stopping at the first disagreement. It keeps those, plus the target's own token at the first mismatch (a free "bonus" token):

```
draft:   "4"  "2"  "."  "\n"  "So"
target:  "4"  "2"  "."  "\n"  "Thus"   ← mismatch at position 5
keep:    "4"  "2"  "."  "\n"  "Thus"   = 5 tokens from ONE pass over the weights
```

The KV cache is then cut back to the accepted length (rejected entries are removed). The hidden states captured in step 3 become the drafter's context for the next round. Then the loop repeats.

**Why this matters for you:** without DFlash, each streamed pass over the 36 layers gives you 1 token. With DFlash, it gives you "accepted + 1" tokens, which is the *acceptance length* you must report. If that averages 4, generation needs about 4× fewer weight streams.

### Three details in the reference code that affect your work

1. **The sampling scheme is exact matching, not the textbook ratio test.** Acceptance is exact token matching, not probability-ratio rejection sampling. The target samples its own token at each position, and a draft token is kept only if it equals the target's sample. Every kept token is therefore something the target itself sampled, so the output distribution is still exactly the target's. It accepts somewhat less often than the textbook rule would. You still need the "distribution check" the task asks for, because you'll reimplement this logic yourself.
2. **It's written for batch size 1** (`acceptance_length ... [0].item()`). In GRPO you generate a group of 8 at once, and each sequence accepts a different number of tokens per round. Generalizing this, with per-sequence KV lengths, is your work.
3. **The layer-index offset is a silent-failure trap.** In HuggingFace, `hidden_states[0]` is the embedding output, not layer 0's output. If your L2L engine captures "layer 9" with an off-by-one, the drafter still runs and still drafts tokens. Acceptance just quietly drops, and nothing errors. Compare your captured features with HF's `output_hidden_states` on the same input.

Also note: the drafter was trained against the **base** model's hidden states. As your LoRA changes the target, those hidden states shift, and acceptance should fall. The task asks you to measure that.

## Pinned memory

**Background:** your OS manages RAM in pages (~4 KB chunks) and is allowed to move them around or swap them to disk behind your program's back. That's normal "pageable" memory.

**The problem:** the GPU copies data from RAM using its own copy engine (DMA). It reads physical RAM addresses directly, without asking the CPU. If the OS could move a page mid-copy, the GPU would read garbage. So CUDA refuses to DMA from pageable memory directly. Instead it:
1. copies your data, using the CPU, into a hidden locked staging buffer, then
2. DMAs from there to the GPU.

That's an extra copy, it's slower, and it ties up the CPU. Worst of all, **`non_blocking=True` silently stops being asynchronous**, so your copy/compute overlap disappears. No error, just slowness. The paper hit exactly this (§5.2): a store that couldn't be pinned dropped the 3B run to a pageable, synchronous copy.

**Pinned (page-locked) memory** is RAM where you ask the OS: "never move or swap these pages." Then the GPU can DMA straight from it, asynchronously, at full PCIe speed, while the CPU and GPU keep working.

```python
cpu_w = torch.empty(shape, dtype=torch.bfloat16, pin_memory=True)  # allocate pinned
cpu_w.copy_(weights_from_safetensors)        # fill it once (trap 3 in the paper: avoid extra copies)
assert cpu_w.is_pinned()

with torch.cuda.stream(copy_stream):
    gpu_slot.copy_(cpu_w, non_blocking=True) # truly async, runs alongside compute
    ready_event.record()
```

**Costs:**
- Pinned RAM is taken away from the OS, so pinning too much can starve the system.
- Allocating it is slow, so do it once at startup.
- Containers often cap it. Check `ulimit -l` on your Vast box, because pinning 7 GB may fail there.
- Never overwrite or free a pinned buffer until the copy reading from it has finished. Wait on the event first.

**A good first experiment on the GPU box:** copy a 200 MB tensor to the GPU from pageable memory and from pinned memory, and time both. You'll see the speed difference, and you get the measured PCIe bandwidth that part 1's time model needs.
