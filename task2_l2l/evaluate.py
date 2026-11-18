#!/usr/bin/env python3
"""Pass rate of a model (base, base + adapter, or a merged directory) on a JSONL set, on the streamed engine.

    python evaluate.py --model Qwen/Qwen3-4B --data data/gsm8k/heldout --n 200            # base pass rate
    python evaluate.py --model Qwen/Qwen3-4B --adapter runs/main/adapter.pt --data ...     # policy
    python evaluate.py --model runs/main/merged --data ...                                 # merged

Greedy by default (k = 1, T = 0); --k 4 --temperature 1 for sampled pass rates. Writes per-item rows
(id, sample, reward, has_box, finished, length, text) and a summary with a bootstrap 95 % CI over items,
reported with format and truncation rates separately from correctness.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from data_stream import PromptStream
from decode import generate
from monitor import NvSmi, alloc_conf, alloc_counters
from reward import REWARDS
from train import build, encode


def bootstrap_ci(x, n=2000, seed=0):
    x = torch.as_tensor(x, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    means = x[torch.randint(0, len(x), (n, len(x)), generator=g)].mean(-1)
    return float(means.quantile(0.025)), float(means.quantile(0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16, help="sequences per decode batch")
    ap.add_argument("--max-prompt", type=int, default=256, help="P of the memory plan; longer prompts abort")
    ap.add_argument("--reward", default="math_boxed")
    ap.add_argument("--prompt-style", default="cot", choices=["cot", "direct", "plain"],
                    help="must match the training run; an adapter trained with another style aborts")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--resident", type=int, default=20)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output prefix: <out>.jsonl and <out>.summary.json")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    nvsmi = NvSmi(f"{args.out}.nvsmi.csv")  # the cap applies to evaluation too
    try:
        evaluate(args)
    finally:
        nvsmi.stop()


def evaluate(args):
    ad = torch.load(args.adapter, weights_only=False, map_location="cpu") if args.adapter else None
    if ad:  # evaluating with another prompt than training used is a silent Part 3 failure
        trained = ad.get("config", {}).get("prompt_style", "cot")
        assert trained == args.prompt_style, f"adapter trained with prompt_style={trained}, eval asked {args.prompt_style}"
    cfg = dict(model=args.model, layers=args.layers, device=args.device, dtype=args.dtype, seed=args.seed,
               lora_r=ad["r"] if ad else 16, lora_alpha=ad["alpha"] if ad else 32, resident=args.resident, n_buffers=2)
    mcfg, tok, eng, lora = build(cfg)
    if ad:
        lora.load_state_dict(ad["lora"])
    eng.adapter_enabled = ad is not None
    stream = PromptStream(args.data, shuffle=False)
    recs = stream.next(min(args.n, stream.N))
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    gen = torch.Generator(device=eng.device).manual_seed(args.seed)
    reward_fn = REWARDS[args.reward]
    too_long = {r["id"]: n for r in recs if (n := len(encode(tok, r["prompt"], args.prompt_style))) > args.max_prompt}
    assert not too_long, f"prompts longer than --max-prompt={args.max_prompt} tokens: {too_long}"
    jobs = [(r, j) for r in recs for j in range(args.k)]
    rows, t0 = [], time.time()
    for a in range(0, len(jobs), args.batch):
        chunk = jobs[a:a + args.batch]
        ro = generate(eng, [encode(tok, r["prompt"], args.prompt_style) for r, _ in chunk], args.max_new, eos, tok.pad_token_id,
                      temperature=args.temperature, generator=gen)
        for b, (r, j) in enumerate(chunk):
            text = tok.decode(ro.ids[b, ro.P:][ro.comp[b, ro.P:]], skip_special_tokens=True)
            rew, info = reward_fn(text, r["answer"])
            rows.append({"id": r["id"], "sample": j, "reward": rew, "has_box": info.get("has_box", False),
                         "finished": bool(ro.finished[b]), "length": int(ro.comp[b].sum()), "text": text})
        retries = alloc_counters(eng.device)["alloc_retries"] if eng.cuda else 0
        print(f"batch {a // args.batch + 1}/{-(-len(jobs) // args.batch)}: {time.time() - t0:.0f} s, "
              f"pass so far {sum(x['reward'] for x in rows) / len(rows):.3f}, alloc retries {retries}", flush=True)
    per_item = {}
    for x in rows:
        per_item.setdefault(x["id"], []).append(x["reward"])
    item_means = [sum(v) / len(v) for v in per_item.values()]
    summary = {"model": args.model, "adapter": args.adapter, "data": args.data, "n_items": len(per_item), "k": args.k,
               "temperature": args.temperature, "prompt_style": args.prompt_style, "max_new": args.max_new,
               "pass_rate": sum(item_means) / len(item_means),
               "ci95": bootstrap_ci(item_means), "has_box_rate": sum(x["has_box"] for x in rows) / len(rows),
               "truncated_rate": sum(not x["finished"] for x in rows) / len(rows),
               "mean_length": sum(x["length"] for x in rows) / len(rows), "seconds": time.time() - t0,
               # dataset rule: groups whose k rewards are all equal give GRPO zero advantage
               "zero_var_group_frac": sum(len(set(v)) == 1 for v in per_item.values()) / len(per_item)
               if args.k > 1 else None,
               "alloc_conf": alloc_conf()}
    if eng.cuda:
        summary.update(alloc_counters(eng.device))
        summary["gpu_alloc_peak_mib"] = torch.cuda.max_memory_allocated(eng.device) / 2**20
        summary["gpu_reserved_peak_mib"] = torch.cuda.max_memory_reserved(eng.device) / 2**20
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    json.dump(summary, open(f"{out}.summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
