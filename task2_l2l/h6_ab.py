#!/usr/bin/env python3
"""H6: dynamic vs regular residency, A-B-B-A blocks (h6_queue.sh). Writes results/h6_ab.md and .json.

Pre-registered rule (task1_budget/README.md): adopt dynamic only if the step-time gain is >= 2 % with a 95 % CI
excluding 0, rollouts are identical (same tokens -> same work), and nvidia-smi stays under the cap.
Pairs: block reg1 <-> dyn1 and dyn2 <-> reg2, step by step (same prompts, same seed); steps 0-1 are warm-up.
"""
import csv
import glob
import json
from pathlib import Path

import torch

HERE = Path(__file__).parent
BLOCKS = ("reg1", "dyn1", "dyn2", "reg2")
WARMUP = 2


def load(name):
    rows = {}
    for l in open(HERE / f"runs/h6/{name}/train_log.jsonl"):
        r = json.loads(l)
        rows[r["step"]] = r  # last attempt wins (there is only one)
    peak = 0.0
    for f in glob.glob(str(HERE / f"runs/h6/{name}/logs/nvsmi-*.csv")):
        if f.endswith(".apps.csv"):
            continue
        for r in csv.DictReader(open(f)):
            k = next(c for c in r if "memory.used" in c)
            peak = max(peak, float(r[k].split()[0]))
    return rows, peak


def main():
    runs = {b: load(b) for b in BLOCKS}
    steps = sorted(set.intersection(*[set(runs[b][0]) for b in BLOCKS]))
    meas = [s for s in steps if s >= WARMUP]
    t = lambda b, s: runs[b][0][s]["phases"]["rollout"]["s"] + runs[b][0][s]["phases"]["train"]["s"]  # noqa: E731
    roll = lambda b, s: runs[b][0][s]["phases"]["rollout"]["s"]  # noqa: E731
    same = all(runs[b][0][s]["completion_tokens"] == runs["reg1"][0][s]["completion_tokens"]
               and runs[b][0][s]["sample_ids"] == runs["reg1"][0][s]["sample_ids"]
               and runs[b][0][s]["reward"]["reward_mean"] == runs["reg1"][0][s]["reward"]["reward_mean"]
               for b in BLOCKS for s in steps)
    pairs = [(t("reg1", s), t("dyn1", s)) for s in meas] + [(t("reg2", s), t("dyn2", s)) for s in meas]
    rpairs = [(roll("reg1", s), roll("dyn1", s)) for s in meas] + [(roll("reg2", s), roll("dyn2", s)) for s in meas]

    def gain(p, n=4000):  # relative step-time saving of dynamic, bootstrap over pairs
        x = torch.tensor(p, dtype=torch.float64)
        g = torch.Generator().manual_seed(0)
        idx = torch.randint(0, len(x), (n, len(x)), generator=g)
        boot = 1 - x[idx][..., 1].sum(-1) / x[idx][..., 0].sum(-1)
        return float(1 - x[:, 1].sum() / x[:, 0].sum()), float(boot.quantile(0.025)), float(boot.quantile(0.975))

    g, lo, hi = gain(pairs)
    rg, rlo, rhi = gain(rpairs)
    peak = {b: runs[b][1] for b in BLOCKS}
    res = {b: [runs[b][0][s].get("resident_min_max") for s in meas] for b in BLOCKS}
    adopt = g >= 0.02 and lo > 0 and same and max(peak.values()) <= 7840
    out = {"n_pairs": len(pairs), "step_gain": g, "step_gain_ci95": [lo, hi], "rollout_gain": rg,
           "rollout_gain_ci95": [rlo, rhi], "rollouts_identical": same, "nvsmi_peak_mib": peak,
           "resident_min_max": res, "adopt_dynamic": adopt,
           "mean_step_s": {b: sum(t(b, s) for s in meas) / len(meas) for b in BLOCKS}}
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "results/h6_ab.json", "w"), indent=1)
    md = ["# H6: dynamic vs regular residency (A-B-B-A, steps 2-7 of 8 per block)", "",
          "| block | mean step s | resident min-max (decode) | nvidia-smi peak MiB |", "|---|---|---|---|"]
    for b in BLOCKS:
        md.append(f"| {b} | {out['mean_step_s'][b]:.2f} | {res[b][0]} | {peak[b]:.0f} |")
    md += ["", f"- step-time saving of dynamic: **{g:+.1%}** [{lo:+.1%}, {hi:+.1%}] over {len(pairs)} paired steps",
           f"- rollout-time saving: {rg:+.1%} [{rlo:+.1%}, {rhi:+.1%}]",
           f"- rollouts identical across all blocks (tokens, prompts, rewards): {same}",
           f"- rule (gain >= 2 %, CI excl. 0, identical, <= 7840 MiB): **{'adopt dynamic' if adopt else 'keep regular'}**"]
    open(HERE / "results/h6_ab.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
