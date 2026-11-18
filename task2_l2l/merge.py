#!/usr/bin/env python3
"""Merge a LoRA adapter into the base model tensor by tensor, shard by shard.

    python merge.py --base Qwen/Qwen3-4B --adapter runs/main/adapter.pt --out runs/main/merged --device cuda

For each input shard: read one tensor (mmap), and if it is a LoRA target move it to the device,
W' = W + (alpha/r)·B·A in fp32, cast back to the base dtype, bring it back; the output shard is written with
the same partition as the input. GPU peak = context + one tensor's working set; host peak = one output shard
plus page cache of the mmapped input. Never more than one tensor on the GPU.

Also reports how much of each delta survives the cast to bf16: a LoRA delta below half a bf16 step of W is
rounded away and the merged model silently equals the base there (checked in Part 3 as merged vs adapter).
"""
import argparse
import json
import re
import resource
import shutil
import sys
import time
from pathlib import Path

import torch

MiB = 2**20
TARGET_RE = re.compile(r"^model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.weight$")


def peak_rss_mib():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / MiB if sys.platform == "darwin" else r / 1024  # bytes on macOS, KiB on Linux


def merge(base_dir, adapter, out_dir, device="cpu"):
    from safetensors import safe_open
    from safetensors.torch import save_file
    base_dir, out_dir, dev = Path(base_dir), Path(out_dir), torch.device(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    ad = torch.load(adapter, weights_only=False, map_location="cpu") if not isinstance(adapter, dict) else adapter
    lora, scale, targets = ad["lora"], ad["alpha"] / ad["r"], set(ad["targets"])
    n_layers = 1 + max(int(k.split(".")[1].split("_")[0]) for k in lora)
    index = base_dir / "model.safetensors.index.json"
    files = sorted(set(json.load(open(index))["weight_map"].values())) if index.exists() else ["model.safetensors"]
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    t0, merged, stats = time.time(), 0, []
    for fn in files:
        out = {}
        with safe_open(base_dir / fn, "pt") as f:
            meta = f.metadata()
            for k in f.keys():
                t = f.get_tensor(k)
                m = TARGET_RE.match(k)
                if m and m[2] in targets and int(m[1]) < n_layers:
                    i, name = int(m[1]), m[2]
                    A = lora[f"p.{i}_{name}_A"].to(dev, torch.float32)
                    B = lora[f"p.{i}_{name}_B"].to(dev, torch.float32)
                    W = t.to(dev, torch.float32)
                    delta = scale * (B @ A)
                    new = (W + delta).to(t.dtype)
                    kept = (new.float() - W).norm() / delta.norm().clamp(min=1e-30)  # share of the delta that survives
                    stats.append({"tensor": k, "delta_rel": float(delta.norm() / W.norm()),
                                  "changed_frac": float((new != t.to(dev)).float().mean()), "delta_kept": float(kept)})
                    t = new.cpu()
                    merged += 1
                    del A, B, W, delta, new
                out[k] = t.contiguous()
        save_file(out, out_dir / fn, metadata=meta or {"format": "pt"})
        del out
        print(f"{time.strftime('%H:%M:%S')} shard {fn} written ({merged} tensors merged so far, "
              f"{time.time() - t0:.0f} s)", flush=True)
    for p in base_dir.iterdir():  # config, tokenizer, index
        if p.is_file() and p.suffix != ".safetensors":
            shutil.copy(p, out_dir / p.name)
    assert merged == n_layers * len(targets), f"merged {merged} tensors, expected {n_layers * len(targets)}"
    report = {"merged_tensors": merged, "seconds": time.time() - t0, "host_peak_rss_mib": peak_rss_mib(),
              "largest_input_shard_mib": max((base_dir / fn).stat().st_size for fn in files) / MiB,
              "changed_frac_min": min(s["changed_frac"] for s in stats),
              "changed_frac_mean": sum(s["changed_frac"] for s in stats) / len(stats),
              "delta_kept_min": min(s["delta_kept"] for s in stats), "per_tensor": stats}
    if dev.type == "cuda":
        report["gpu_alloc_peak_mib"] = torch.cuda.max_memory_allocated(dev) / MiB
        report["gpu_reserved_peak_mib"] = torch.cuda.max_memory_reserved(dev) / MiB
    json.dump(report, open(out_dir / "merge_report.json", "w"), indent=1)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    base = Path(args.base)
    if not base.exists():
        from huggingface_hub import snapshot_download
        base = Path(snapshot_download(args.base, allow_patterns=["*.json", "*.safetensors", "*.txt"]))
    from monitor import NvSmi
    Path(args.out).mkdir(parents=True, exist_ok=True)
    nvsmi = NvSmi(Path(args.out) / "merge_nvsmi.csv")  # the cap applies to the merge too
    try:
        rep = merge(base, args.adapter, args.out, args.device)
    finally:
        nvsmi.stop()
    print(json.dumps({k: v for k, v in rep.items() if k != "per_tensor"}, indent=1))


if __name__ == "__main__":
    main()
