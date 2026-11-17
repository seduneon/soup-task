#!/usr/bin/env python3
"""Part 3: did the model actually train? Four arms on the SAME held-out items with the SAME decoding settings.

    python verify.py --model Qwen/Qwen3-4B --adapter runs/main/adapter.pt --merged runs/main/merged \
        --control runs/control/adapter.pt --data data/math35/heldout --n 500 --out runs/verify

Arms: base (adapter off), adapter (base + LoRA), merged (LoRA folded into the weights, bf16), control (an adapter
trained with --set reward_shuffle=true: same pipeline and step count, reward unrelated to the completion).
Each arm is one evaluate.py process (own CUDA context, own nvidia-smi log), skipped if its summary exists.

Rule, written before any run (all four must hold for "TRAINED"):
  R1 effect:      adapter - base, paired bootstrap 95 % CI over items, lower bound > 0.
  R2 not generic: adapter - control, CI lower bound > 0 (a gain from "any LoRA update / any GRPO noise" is not
                  learning from the reward).
  R3 merge:       merged - adapter CI contains 0 (merge is the same model up to bf16 rounding).
  R4 not format:  adapter - base restricted to items where BOTH arms boxed an answer, CI lower bound > 0, or
                  fewer than 30 such items (then reported as "cannot separate format from correctness").
Everything is paired by item id; the script refuses arms evaluated on different items or settings.

What this does NOT detect (stated in the report):
  - contamination: a held-out gain is not proof of skill if the held-out set leaked into pre-training;
  - reward-parser bias shared by all arms (the same scorer grades every arm);
  - gains outside the held-out distribution, or regressions elsewhere (no general benchmark here);
  - an effect smaller than the CI half-width (no difference detected != no difference);
  - for greedy k = 1, sampling-time behaviour (use --k 4 --temperature 1 for that).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
SETTINGS = ("data", "n_items", "k", "temperature")


def paired_ci(diff, n=4000, seed=0):
    d = torch.as_tensor(diff, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    means = d[torch.randint(0, len(d), (n, len(d)), generator=g)].mean(-1)
    return float(d.mean()), float(means.quantile(0.025)), float(means.quantile(0.975))


def load_arm(prefix):
    rows = [json.loads(l) for l in open(f"{prefix}.jsonl")]
    summary = json.load(open(f"{prefix}.summary.json"))
    items = {}
    for r in rows:
        items.setdefault(r["id"], []).append(r)
    return summary, items


def compare(arms, min_boxed=30):
    """arms: name -> (summary, {id: [rows]}). Returns the report dict (pure, unit-tested)."""
    names = list(arms)
    ref_s, ref_items = arms[names[0]]
    for nm in names[1:]:
        s, items = arms[nm]
        assert set(items) == set(ref_items), f"{nm} was evaluated on different items than {names[0]}"
        for key in SETTINGS:
            assert s.get(key) == ref_s.get(key), f"{nm}: {key}={s.get(key)!r} vs {ref_s.get(key)!r}"
    ids = sorted(ref_items)

    def score(nm, i):
        v = arms[nm][1][i]
        return sum(r["reward"] for r in v) / len(v)

    def boxed(nm, i):
        return all(r["has_box"] for r in arms[nm][1][i])

    rep = {"n_items": len(ids), "arms": {}, "pairs": {}}
    for nm in names:
        s = arms[nm][0]
        rep["arms"][nm] = {k: s.get(k) for k in ("pass_rate", "ci95", "has_box_rate", "truncated_rate", "mean_length")}

    def pair(a, b, subset=None):
        sub = [i for i in ids if subset is None or subset(i)]
        if not sub:
            return {"n": 0}
        d = [score(a, i) - score(b, i) for i in sub]
        mean, lo, hi = paired_ci(d)
        return {"n": len(sub), "diff": mean, "ci95": [lo, hi], "a_better": sum(x > 0 for x in d),
                "b_better": sum(x < 0 for x in d)}

    for a, b in (("adapter", "base"), ("adapter", "control"), ("control", "base"), ("merged", "adapter"),
                 ("merged", "base")):
        if a in arms and b in arms:
            rep["pairs"][f"{a}-{b}"] = pair(a, b)
    if "merged" in arms and "adapter" in arms:  # same model up to rounding: how often is the output text identical
        same = [all(x["text"] == y["text"] for x, y in zip(arms["merged"][1][i], arms["adapter"][1][i])) for i in ids]
        rep["merged_adapter_text_identical"] = sum(same) / len(ids)
    if "adapter" in arms and "base" in arms:
        rep["pairs"]["adapter-base|both_boxed"] = pair("adapter", "base", lambda i: boxed("adapter", i) and boxed("base", i))

    P = rep["pairs"]
    lo = lambda k: P[k]["ci95"][0] if P.get(k, {}).get("n") else None  # noqa: E731
    checks = {}
    if "adapter-base" in P:
        checks["R1_effect"] = lo("adapter-base") > 0
    if "adapter-control" in P:
        checks["R2_not_generic"] = lo("adapter-control") > 0
    if "merged-adapter" in P:
        a, b = P["merged-adapter"]["ci95"]
        checks["R3_merge"] = a <= 0 <= b
    fb = P.get("adapter-base|both_boxed")
    if fb is not None:
        checks["R4_not_format"] = (fb["ci95"][0] > 0) if fb["n"] >= min_boxed else None  # None: too few to tell
    rep["checks"] = checks
    required = ("R1_effect", "R2_not_generic", "R3_merge")
    if any(k not in checks for k in required):
        rep["verdict"] = "NO VERDICT (missing arm)"
    elif all(checks[k] for k in required) and checks.get("R4_not_format") is not False:
        rep["verdict"] = "TRAINED" + ("" if checks.get("R4_not_format") else " (format vs correctness not separable)")
    else:
        rep["verdict"] = "NOT SHOWN: " + ", ".join(k for k, v in checks.items() if v is False)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--merged")
    ap.add_argument("--control")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--reward", default="math_boxed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--resident", type=int, default=20)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = {"base": (args.model, None)}
    if args.adapter:
        arms["adapter"] = (args.model, args.adapter)
    if args.merged:
        arms["merged"] = (args.merged, None)
    if args.control:
        arms["control"] = (args.model, args.control)
    common = ["--data", args.data, "--n", str(args.n), "--k", str(args.k), "--temperature", str(args.temperature),
              "--max-new", str(args.max_new), "--reward", args.reward, "--device", args.device, "--dtype", args.dtype,
              "--resident", str(args.resident), "--batch", str(args.batch), "--seed", "0"]
    if args.layers:
        common += ["--layers", str(args.layers)]
    for nm, (model, adapter) in arms.items():
        if Path(f"{out / nm}.summary.json").exists():
            print(f"{nm}: reusing {out / nm}.summary.json", flush=True)
            continue
        cmd = [sys.executable, str(HERE / "evaluate.py"), "--model", model, "--out", str(out / nm), *common]
        if adapter:
            cmd += ["--adapter", adapter]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    rep = compare({nm: load_arm(out / nm) for nm in arms})
    json.dump(rep, open(out / "verify_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
