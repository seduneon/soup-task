#!/usr/bin/env python3
"""Countdown-3 shards for GRPO: data/countdown3/{train,heldout}/*.jsonl with {id, prompt, answer, ...}.

    python prepare_countdown.py                 # 1500 train / 500 held-out, 3-number items only

Source: Jiayi-Pan/Countdown-Tasks-3to4 (TinyZero's Countdown set), rows {target, nums}. Only 3-number items are
kept: answers stay short (4-number search runs long and brings back the truncation confound we hit on MATH).

Every candidate is brute-forced (all orderings x all binary trees x all + - * / choices, exact fractions):
unsolvable items are dropped, and each item is tagged with n_solutions and needs_muldiv (no +/- only solution),
the difficulty knobs. Held-out and train are disjoint on the (sorted nums, target) pair. answer = "a,b,c|target",
what reward.countdown parses; the same check is run on one known solution per item before writing.
"""
import argparse
import itertools
import random
from fractions import Fraction
from pathlib import Path

import pandas as pd

from data_stream import write_jsonl_shards
from reward import countdown

HERE = Path(__file__).parent
PROMPT = ("Using the numbers {nums}, write an arithmetic expression that equals {target}. "
          "Use +, -, *, / and parentheses, and use each number exactly once. Keep any working brief. "
          "Put only the final expression inside <answer></answer>, for example <answer>(9 - 5) * 7</answer>.")
EXAMPLE_NUMS = (5, 7, 9)  # the numbers of the example inside PROMPT
OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
       "/": lambda a, b: a / b if b != 0 else None}


def solutions(nums, target):
    """All expressions (as strings) over the 3 numbers that equal target; both tree shapes, every order."""
    out = []
    for a, b, c in itertools.permutations(nums):
        for o1, o2 in itertools.product(OPS, repeat=2):
            x = OPS[o1](Fraction(a), Fraction(b))  # (a o1 b) o2 c
            if x is not None and (v := OPS[o2](x, Fraction(c))) is not None and v == target:
                out.append(f"({a} {o1} {b}) {o2} {c}")
            y = OPS[o2](Fraction(b), Fraction(c))  # a o1 (b o2 c)
            if y is not None and (v := OPS[o1](Fraction(a), y)) is not None and v == target:
                out.append(f"{a} {o1} ({b} {o2} {c})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-heldout", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", type=int, default=20000, help="3-number rows brute-forced before sampling")
    ap.add_argument("--out", default=HERE / "data" / "countdown3")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    df = pd.read_parquet(hf_hub_download("Jiayi-Pan/Countdown-Tasks-3to4", "data/train-00000-of-00001.parquet",
                                         repo_type="dataset"))
    rng = random.Random(args.seed)
    rows = df.index[df["nums"].map(len) == 3].tolist()
    print(f"source rows {len(df)}, 3-number rows {len(rows)}")
    pool, seen, n_unsolvable, n_dup = [], set(), 0, 0
    for i in rng.sample(rows, min(args.pool, len(rows))):
        nums, target = [int(x) for x in df.iloc[i]["nums"]], int(df.iloc[i]["target"])
        key = (tuple(sorted(nums)), target)
        if key[0] == EXAMPLE_NUMS:  # the prompt's worked example must never be a real item's answer
            continue
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        sols = solutions(nums, target)
        if not sols:
            n_unsolvable += 1
            continue
        plus_minus_only = any("*" not in s and "/" not in s for s in sols)
        pool.append({"nums": nums, "target": target, "n_solutions": len(sols), "needs_muldiv": not plus_minus_only,
                     "example_solution": sols[0]})
    print(f"brute-forced {len(seen)} unique items: {n_unsolvable} unsolvable dropped, {n_dup} duplicates skipped")
    need = args.n_train + args.n_heldout
    assert len(pool) >= need, f"only {len(pool)} solvable items, need {need}: raise --pool"
    picked = rng.sample(pool, need)  # unique keys already, so the two splits are disjoint by construction
    splits = {"heldout": picked[:args.n_heldout], "train": picked[args.n_heldout:]}
    keys = {s: {(tuple(sorted(r["nums"])), r["target"]) for r in v} for s, v in splits.items()}
    assert not keys["train"] & keys["heldout"], "train/held-out overlap"
    for name, items in splits.items():
        recs = []
        for j, it in enumerate(items):
            answer = f"{','.join(map(str, it['nums']))}|{it['target']}"
            r, _ = countdown(f"<answer>{it['example_solution']}</answer>", answer)
            assert r == 1.0, f"reward rejects a brute-forced solution: {it}"  # solver and scorer agree
            recs.append({"id": f"countdown3-{name}-{j}", "answer": answer,
                         "prompt": PROMPT.format(nums=", ".join(map(str, it["nums"])), target=it["target"]),
                         "n_solutions": it["n_solutions"], "needs_muldiv": it["needs_muldiv"]})
        write_jsonl_shards(recs, Path(args.out) / name, per_shard=500)
        md = sum(r["needs_muldiv"] for r in recs) / len(recs)
        ns = sorted(r["n_solutions"] for r in recs)
        print(f"{name}: {len(recs)} items, needs */÷ {md:.0%}, n_solutions median {ns[len(ns) // 2]}")
    print("->", args.out)


if __name__ == "__main__":
    main()
