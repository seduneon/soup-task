#!/usr/bin/env python3
"""R3 follow-up: is "merged != adapter" the bf16 cast, or a merge bug?

    python fp32_check.py        # reads runs/verify (bf16) and runs/verify_fp32 (fp32), writes results/fp32_check.md

runs/verify_fp32 = the same 500 held-out items and settings as runs/verify, but fp32 weights and compute:
  adapter: fp32 base + the trained LoRA attached;  merged: the same LoRA folded into the fp32 weights in host memory
  (store.merge_lora, bit-identical to merge.py, tests/test_merge.py), evaluated with the adapter off.
Expectation if merge.py is correct and only bf16 rounding broke R3: fp32 merged == fp32 adapter (CI ∋ 0, text ~100 %).
"""
import json
from pathlib import Path

from verify import load_arm, paired_ci

HERE = Path(__file__).parent
ARMS = {"adapter_bf16": "runs/verify/adapter", "merged_bf16": "runs/verify/merged", "base_bf16": "runs/verify/base",
        "adapter_fp32": "runs/verify_fp32/adapter", "merged_fp32": "runs/verify_fp32/merged"}
PAIRS = [("merged_fp32", "adapter_fp32"), ("merged_bf16", "adapter_bf16"), ("adapter_fp32", "adapter_bf16"),
         ("merged_fp32", "adapter_bf16"), ("merged_bf16", "merged_fp32"), ("adapter_fp32", "base_bf16")]


def main():
    arms = {nm: load_arm(HERE / p) for nm, p in ARMS.items()}
    ids = sorted(arms["adapter_bf16"][1])
    for nm, (s, items) in arms.items():
        assert sorted(items) == ids, f"{nm}: different items"
        assert (s["max_new"], s["prompt_style"], s["temperature"]) == (128, "plain", 0.0), nm
    row = lambda nm, i: arms[nm][1][i][0]  # noqa: E731  (k = 1)
    out = {"arms": {}, "pairs": {}}
    lines = ["# fp32 merge check (R3 follow-up)", "", "| arm | pass | 95 % CI | >1 answer tag | truncated | mean tokens |",
             "|---|---|---|---|---|---|"]
    for nm in ARMS:
        s = arms[nm][0]
        multi = sum(row(nm, i)["text"].count("<answer>") > 1 for i in ids) / len(ids)
        out["arms"][nm] = {"pass": s["pass_rate"], "ci95": s["ci95"], "multi_answer": multi,
                           "truncated": s["truncated_rate"], "mean_length": s["mean_length"],
                           "dtype": s.get("dtype", "bfloat16"), "merge_stats": s.get("merge_stats")}
        lines.append(f"| {nm} | {s['pass_rate']:.3f} | [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] | {multi:.1%} | "
                     f"{s['truncated_rate']:.1%} | {s['mean_length']:.0f} |")
    lines += ["", "| pair (a − b) | diff | 95 % CI | identical text |", "|---|---|---|---|"]
    for a, b in PAIRS:
        d = [row(a, i)["reward"] - row(b, i)["reward"] for i in ids]
        mean, lo, hi = paired_ci(d)
        same = sum(row(a, i)["text"] == row(b, i)["text"] for i in ids) / len(ids)
        out["pairs"][f"{a}-{b}"] = {"diff": mean, "ci95": [lo, hi], "text_identical": same}
        lines.append(f"| {a} − {b} | {mean:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {same:.1%} |")
    p = out["pairs"]["merged_fp32-adapter_fp32"]
    out["R3_fp32"] = p["ci95"][0] <= 0 <= p["ci95"][1]
    lines += ["", f"R3 at fp32 (merged − adapter CI ∋ 0): **{'pass' if out['R3_fp32'] else 'fail'}**, "
              f"identical text {p['text_identical']:.1%}."]
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "results/fp32_check.json", "w"), indent=1)
    open(HERE / "results/fp32_check.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
