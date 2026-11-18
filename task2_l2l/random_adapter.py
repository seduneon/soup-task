#!/usr/bin/env python3
"""Random-direction control for the merge: same per-tensor ‖ΔW‖ as a trained adapter, random direction.

    python random_adapter.py --adapter runs/main/adapter.pt --out runs/random/adapter.pt --seed 0

Each LoRA pair (A, B) is replaced by Gaussian (A', B') of the same shapes, rescaled so ‖B'A'‖_F = ‖BA‖_F.
Merged and evaluated like the real adapter, it measures how far a weight change of this size moves
greedy held-out accuracy when it carries no learned signal (the size of the bf16 merge perturbation
is of the same order: the merge keeps 50-92 % of each delta).
"""
import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ad = torch.load(args.adapter, weights_only=False, map_location="cpu")
    g = torch.Generator().manual_seed(args.seed)
    new = {}
    for k, A in ad["lora"].items():
        if not k.endswith("_A"):
            continue
        B = ad["lora"][k[:-2] + "_B"]
        Ar = torch.randn(A.shape, generator=g, dtype=torch.float32)
        Br = torch.randn(B.shape, generator=g, dtype=torch.float32)
        s = ((B.double() @ A.double()).norm() / (Br.double() @ Ar.double()).norm()).sqrt().float()
        new[k], new[k[:-2] + "_B"] = (Ar * s).to(A.dtype), (Br * s).to(B.dtype)
        got, want = (new[k[:-2] + "_B"].double() @ new[k].double()).norm(), (B.double() @ A.double()).norm()
        assert abs(got - want) <= 1e-3 * want, (k, float(got), float(want))
    out = dict(ad, lora=new, random_control={"source": str(args.adapter), "seed": args.seed})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"wrote {args.out}: {len(new) // 2} pairs, norms matched per tensor")


if __name__ == "__main__":
    main()
