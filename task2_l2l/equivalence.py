#!/usr/bin/env python3
"""L2L equivalence on real Qwen3 weights: streamed engine vs non-L2L autograd reference vs high-precision truth.

    python equivalence.py                                  # Qwen3-0.6B, first 8 layers, CPU (fits 8 GB RAM)
    python equivalence.py --layers 28 --device cuda        # full 0.6B on the GPU: bf16 vs bf16 ref vs fp32 truth

Checks (rule in compare.py): resident engine == HF logits; streamed logits == reference logits;
streamed LoRA grads within RATIO x the reference's own rounding error, full batch and L2L micro-batches;
one copy per streamed layer per pass. Writes results/equivalence_<tag>.{json,md}.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from compare import RATIO, compare, lora_grads
from engine import IGNORE, StreamedQwen3
from lora import LoRAParams, lora_shapes
from store import LayerStore

HERE = Path(__file__).parent
PROMPTS = ["What is 17 * 23?", "Name the capital of Kazakhstan.", "Write a JSON object with keys a and b.",
           "Is 221 a prime number? Answer yes or no."]
ANSWERS = ["17 * 23 = 391.", "Astana.", '{"a": 1, "b": 2}', "No, 221 = 13 * 17."]


def batch(tok):
    """Templated prompts (non-thinking) + answers, left padded; loss on answer tokens only."""
    seqs, starts = [], []
    for p, a in zip(PROMPTS, ANSWERS):
        prompt = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        pi = tok(prompt, add_special_tokens=False).input_ids
        ai = tok(a + "<|im_end|>", add_special_tokens=False).input_ids
        seqs.append(pi + ai)
        starts.append(len(pi))
    S = max(map(len, seqs))
    ids = torch.full((len(seqs), S), tok.pad_token_id)
    mask = torch.zeros(len(seqs), S, dtype=torch.long)
    targets = torch.full((len(seqs), S), IGNORE)
    for b, (s, st) in enumerate(zip(seqs, starts)):
        off = S - len(s)
        ids[b, off:], mask[b, off:] = torch.tensor(s), 1
        targets[b, off + st - 1:S - 1] = torch.tensor(s[st:])  # position t predicts token t+1
    return ids, mask, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--resident", type=int, default=0, help="resident layers in the streamed engine")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM

    from train import model_dir

    dev, dt = torch.device(args.device), getattr(torch, args.dtype)
    truth_dt = torch.float64 if dev.type == "cpu" else torch.float32
    repo = model_dir(args.model)  # local directory or HF repo id
    cfg = AutoConfig.from_pretrained(repo)
    cfg.num_hidden_layers, cfg._attn_implementation = args.layers, "sdpa"
    tok = AutoTokenizer.from_pretrained(repo)
    ids, mask, targets = batch(tok)
    g = torch.Generator().manual_seed(0)
    weights = torch.randn(ids.numel(), generator=g, dtype=torch.float64)  # stand-in for GRPO advantages
    n_tok = int((targets != IGNORE).sum())
    loss_fn = lambda logp, idx: -(weights.to(logp.device)[idx].to(logp.dtype) * logp).sum() / n_tok

    t0 = time.time()
    store = LayerStore.from_safetensors(repo, args.layers, dt)
    lora = LoRAParams(lora_shapes(cfg), args.layers, r=args.rank, alpha=2 * args.rank, device=dev)
    lora.randomize_B(std=0.02)
    report = dict(model=args.model, layers=args.layers, device=str(dev), dtype=args.dtype, truth=str(truth_dt),
                  B=ids.shape[0], S=ids.shape[1], loss_tokens=n_tok, chunk=args.chunk, ratio=RATIO,
                  layer_MiB=store.layer_bytes / 2**20, torch=torch.__version__)

    def grads(eng, mb=None, reference=False):
        eng.lora.zero_grad(set_to_none=True)
        if reference:
            loss = eng.autograd_grads(ids, mask, targets, loss_fn, mb=mb)
        else:
            loss = eng.train_grads(ids, mask, targets, loss_fn, mb=mb, chunk=args.chunk)
        return loss.item(), lora_grads(eng.lora)

    ref = StreamedQwen3(cfg, store, lora, dev, resident=range(args.layers))
    stream = StreamedQwen3(cfg, store, lora, dev, resident=range(args.resident))

    # 1. my resident driver vs HF, LoRA off (B = 0 adds exactly 0)
    lora0 = LoRAParams(lora_shapes(cfg), args.layers, r=args.rank, alpha=2 * args.rank, device=dev)
    ours = StreamedQwen3(cfg, store, lora0, dev, resident=range(args.layers)).logits(ids, mask).float()
    hf = Qwen3ForCausalLM.from_pretrained(repo, config=cfg, torch_dtype=dt).to(dev).eval()
    with torch.no_grad():
        theirs = hf(ids.to(dev), attention_mask=mask.to(dev),
                    position_ids=(mask.cumsum(-1) - 1).clamp(min=0).to(dev)).logits.float()
    del hf
    v = mask.bool().to(dev)
    report["hf_max_abs_diff"] = (ours[v] - theirs[v]).abs().max().item()
    report["hf_argmax_agree"] = (ours[v].argmax(-1) == theirs[v].argmax(-1)).float().mean().item()
    del ours, theirs

    # 2. forward: streamed vs resident reference
    lr, ls = ref.logits(ids, mask), stream.logits(ids, mask)
    report["logits_bitexact"] = bool(torch.equal(lr[v], ls[v]))
    report["logits_max_abs_diff"] = (lr[v].float() - ls[v].float()).abs().max().item()
    del lr, ls

    # 3. gradients, full batch and L2L micro-batches
    truth_store = LayerStore.from_safetensors(repo, args.layers, truth_dt)
    truth_eng = StreamedQwen3(cfg, truth_store, lora.clone_as(truth_dt).to(dev), dev, resident=range(args.layers))
    truth_loss, truth = grads(truth_eng, reference=True)
    del truth_eng, truth_store
    report["truth_loss"] = truth_loss
    report["grads"] = {}
    for mb in (None, 1):
        ref_loss, rg = grads(ref, mb=mb, reference=True)
        loads0, passes0 = stream.loads, stream.passes
        s_loss, sg = grads(stream, mb=mb)
        rep = compare(sg, rg, truth)
        rep.update(ref_loss=ref_loss, stream_loss=s_loss,
                   copies_per_pass=(stream.loads - loads0) / (stream.passes - passes0),
                   streamed_layers=args.layers - args.resident)
        worst = sorted(rep.pop("rows"), key=lambda r: -r["err_stream"] / (r["err_ref"] + 1e-30))[:5]
        rep["worst_tensors"] = worst
        report["grads"]["full batch" if mb is None else f"mb={mb}"] = rep
    report["seconds"] = time.time() - t0

    tag = f"{Path(args.model).name.lower()}_L{args.layers}_{dev.type}_{args.dtype}"
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    json.dump(report, open(out / f"equivalence_{tag}.json", "w"), indent=1)
    md = [f"# L2L equivalence: {args.model}, {args.layers} layers, {dev.type}, {args.dtype}\n",
          f"B={report['B']}, S={report['S']}, {n_tok} loss tokens, chunk {args.chunk}, LoRA r={args.rank} on 7 linears, "
          f"{args.resident} resident / {args.layers - args.resident} streamed, truth {truth_dt}. Rule: "
          f"err(stream) ≤ {RATIO} × err(ref) vs truth, per tensor and in aggregate.\n",
          f"- resident engine vs HF logits: max |Δ| {report['hf_max_abs_diff']:.2e}, argmax agree {report['hf_argmax_agree']:.4f}",
          f"- streamed vs reference logits: bit-exact **{report['logits_bitexact']}**, max |Δ| {report['logits_max_abs_diff']:.2e}\n",
          "| run | pass | bit-exact | err stream | err ref | worst ratio | all grads nonzero | copies/pass |",
          "|---|---|---|---|---|---|---|---|"]
    for k, r in report["grads"].items():
        md.append(f"| {k} | **{r['pass_']}** | {r['bitexact']} | {r['agg_err_stream']:.2e} | {r['agg_err_ref']:.2e} | "
                  f"{r['worst_ratio']:.2f} | {r['all_nonzero']} | {r['copies_per_pass']:.0f} of {r['streamed_layers']} |")
    (out / f"equivalence_{tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
