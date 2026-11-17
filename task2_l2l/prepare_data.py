#!/usr/bin/env python3
"""Build the JSONL shards the stream reads: data/<name>/train/*.jsonl and data/<name>/heldout/*.jsonl.

    python prepare_data.py --source gsm8k --n-train 1500 --n-heldout 500
    python prepare_data.py --source math35                       # MATH levels 3-5, numeric answers only

Held-out comes from the source's test split; train is a seeded sample of its train split. Fails if a
normalised question appears in both (train/held-out leakage) or if a reference answer is not something the
reward can parse as a number (a gold answer the scorer cannot match makes the item always score 0).

math35 (fallback if GSM8K is too easy for the base model): MATH (Hendrycks et al.) levels 3-5, gold answer =
the last \\boxed{} of the reference solution. Items whose gold answer is not a number (expressions, \\sqrt,
\\pi, tuples, matrices, base-n) are dropped BEFORE sampling: a string comparison of LaTeX would give false
zeros (\\frac{\\sqrt3}{3} vs \\frac{1}{\\sqrt3}), i.e. reward noise. This biases the set towards numeric
problems; the dropped share is printed and must be reported.

Prompt length: every item whose templated prompt (chat template + PROMPT_SUFFIX, exactly what train.py feeds
the model) is longer than --max-prompt-tokens is dropped BEFORE sampling, in both splits. The memory budget
sizes the KV cache and activations for P = 256 prompt tokens (task1_budget/config.yaml), and a batch is padded
to its longest prompt, so one long prompt would push the whole step over the 8 GB cap. The dropped share is
printed (math35 loses its longest problems: a bias to report alongside the numeric-only one).
"""
import argparse
import hashlib
import random
import re
from pathlib import Path

import pandas as pd

from data_stream import write_jsonl_shards
from reward import as_number, boxed_all, normalize
from train import encode

HERE = Path(__file__).parent
SOURCES = {
    "gsm8k": dict(repo="openai/gsm8k", train="main/train-00000-of-00001.parquet",
                  test="main/test-00000-of-00001.parquet", question="question",
                  answer=lambda r: r["answer"].split("####")[-1].strip().replace(",", "")),
    "math35": dict(repo="DigitalLearningGmbH/MATH-lighteval", train="data/train-00000-of-00001.parquet",
                   test="data/test-00000-of-00001.parquet", question="problem",
                   keep=lambda r: r["level"] in ("Level 3", "Level 4", "Level 5"),
                   answer=lambda r: (boxed_all(r["solution"]) or [""])[-1].strip(), numeric_only=True,
                   extra=lambda r: {"level": r["level"], "type": r["type"]}),
}


def qhash(q):
    return hashlib.sha256(re.sub(r"\W+", " ", q.lower()).strip().encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="gsm8k", choices=sorted(SOURCES))
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-heldout", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=HERE / "data")
    ap.add_argument("--max-prompt-tokens", type=int, default=256, help="P of the memory budget")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B", help="any Qwen3 size: same tokenizer")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    src = SOURCES[args.source]
    frames = {s: pd.read_parquet(hf_hub_download(src["repo"], src[s], repo_type="dataset")) for s in ("train", "test")}
    rng = random.Random(args.seed)
    out = Path(args.out) / args.source
    test_q = {qhash(q) for q in frames["test"][src["question"]]}
    picked = {}
    for split, name, n in (("train", "train", args.n_train), ("test", "heldout", args.n_heldout)):
        df = frames[split]
        cand = [i for i in range(len(df)) if src.get("keep", lambda r: True)(df.iloc[i])]
        if src.get("numeric_only"):
            n0 = len(cand)
            cand = [i for i in cand if as_number(normalize(src["answer"](df.iloc[i]))) is not None]
            print(f"{split}: kept {len(cand)} of {n0} items with a numeric gold answer ({1 - len(cand) / n0:.1%} dropped)")
        n0 = len(cand)
        cand = [i for i in cand if len(encode(tok, df.iloc[i][src["question"]])) <= args.max_prompt_tokens]
        print(f"{split}: kept {len(cand)} of {n0} items with a templated prompt <= {args.max_prompt_tokens} tokens "
              f"({1 - len(cand) / n0:.1%} dropped)")
        if split == "train":  # a question that is also in the test split never enters train (MATH has some)
            n0 = len(cand)
            cand = [i for i in cand if qhash(df.iloc[i][src["question"]]) not in test_q]
            print(f"train: excluded {n0 - len(cand)} questions that also appear in the test split")
        idx = sorted(rng.sample(cand, min(n, len(cand))))
        recs = [{"id": f"{args.source}-{split}-{i}", "prompt": df.iloc[i][src["question"]],
                 "answer": src["answer"](df.iloc[i]), **src.get("extra", lambda r: {})(df.iloc[i])} for i in idx]
        bad = [r for r in recs if as_number(normalize(r["answer"])) is None]
        assert not bad, f"{len(bad)} reference answers the reward cannot parse, e.g. {bad[0]}"
        picked[name] = recs
    leak = {qhash(r["prompt"]) for r in picked["train"]} & {qhash(r["prompt"]) for r in picked["heldout"]}
    assert not leak, f"{len(leak)} questions appear in both train and held-out"
    for name, recs in picked.items():  # written only after every check passed
        write_jsonl_shards(recs, out / name, per_shard=500)
    print({k: len(v) for k, v in picked.items()}, "->", out)


if __name__ == "__main__":
    main()
