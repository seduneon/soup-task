#!/usr/bin/env python3
"""Turn the raw run outputs into the report numbers (CPU only, reads files, no model).

    python analyze.py --main runs/main --control runs/control --verify runs/verify --resume runs/resume_demo \
        --heldout data/countdown3/heldout --out results

Writes results/analysis.md, results/analysis.json and results/reward_curves.png:
  Part 3: arms, paired diffs split by +/- vs */÷ items, format (has <answer>) vs correctness, length,
          what changed in the answers, LoRA norms of the trained adapter vs the shuffled-reward control,
          training curves of the main run vs the control.
  Part 1: measured step time per phase vs the prediction, peaks per phase, copy/compute overlap.
  Part 2b: kill/resume demo vs the uninterrupted main run (sample ledger + rollout tokens).
"""
import argparse
import json
import statistics as st
from pathlib import Path

import torch

from verify import paired_ci


def jl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def by_step(rows):  # raw log keeps replayed attempts; the last line per step is the committed one
    d = {}
    for r in rows:
        d[r["step"]] = r
    return [d[k] for k in sorted(d)]


def ma(x, w=10):
    return [sum(x[max(0, i - w + 1):i + 1]) / len(x[max(0, i - w + 1):i + 1]) for i in range(len(x))]


def arm_items(path):
    items = {}
    for r in jl(path):
        items.setdefault(r["id"], []).append(r)
    return items


def part3(args, rep, md):
    vdir = Path(args.verify)
    heldout = {r["id"]: r for f in sorted(Path(args.heldout).glob("*.jsonl")) for r in jl(f)}
    arms = {nm: arm_items(vdir / f"{nm}.jsonl") for nm in ("base", "adapter", "control", "merged", "random_adapter", "random_merged")}
    arms = {k: v for k, v in arms.items() if v}
    if "base" not in arms:
        md.append("Part 3: no verify outputs yet.\n")
        return
    ids = sorted(arms["base"])
    score = lambda nm, i: sum(r["reward"] for r in arms[nm][i]) / len(arms[nm][i])  # noqa: E731
    subsets = {"all": ids, "+/- only": [i for i in ids if not heldout[i]["needs_muldiv"]],
               "needs * or /": [i for i in ids if heldout[i]["needs_muldiv"]]}
    out = {"arms": {}, "pairs": {}}
    md.append("## Part 3: held-out Countdown-3, greedy, 500 items\n")
    md.append("| arm | pass (all) | pass +/- | pass */÷ | has `<answer>` | truncated | mean len |")
    md.append("|---|---|---|---|---|---|---|")
    for nm, it in arms.items():
        rows = [r for i in ids for r in it[i]]
        a = {s: sum(score(nm, i) for i in sub) / len(sub) for s, sub in subsets.items()}
        a.update(has_box=sum(r["has_box"] for r in rows) / len(rows),
                 truncated=sum(not r["finished"] for r in rows) / len(rows),
                 mean_len=sum(r["length"] for r in rows) / len(rows))
        out["arms"][nm] = a
        md.append(f"| {nm} | {a['all']:.3f} | {a['+/- only']:.3f} | {a['needs * or /']:.3f} | {a['has_box']:.3f} "
                  f"| {a['truncated']:.3f} | {a['mean_len']:.1f} |")
    md.append(f"\nItems: {len(subsets['+/- only'])} +/- only, {len(subsets['needs * or /'])} need * or /.\n")
    md.append("Paired differences (bootstrap 95 % CI over items, 4000 resamples):\n")
    md.append("| pair | subset | n | diff | 95 % CI | a better | b better |")
    md.append("|---|---|---|---|---|---|---|")
    for a, b in (("adapter", "base"), ("adapter", "control"), ("control", "base"), ("merged", "adapter"),
                 ("merged", "base"), ("random_adapter", "base"), ("random_merged", "random_adapter"),
                 ("random_merged", "base")):
        if a not in arms or b not in arms:
            continue
        for s, sub in subsets.items():
            d = [score(a, i) - score(b, i) for i in sub]
            m, lo, hi = paired_ci(d)
            out["pairs"][f"{a}-{b}|{s}"] = {"n": len(sub), "diff": m, "ci95": [lo, hi]}
            md.append(f"| {a} − {b} | {s} | {len(sub)} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                      f"{sum(x > 0 for x in d)} | {sum(x < 0 for x in d)} |")
    # format vs correctness: among answered items, how often is the expression right
    md.append("\nCorrect given an `<answer>` was produced (format separated from correctness):\n")
    for nm, it in arms.items():
        rows = [r for i in ids for r in it[i] if r["has_box"]]
        if rows:
            md.append(f"- {nm}: {sum(r['reward'] for r in rows) / len(rows):.3f} of {len(rows)} answered")
    if "adapter" in arms:  # what changed: identical text vs base, flips
        same = sum(arms["adapter"][i][0]["text"] == arms["base"][i][0]["text"] for i in ids)
        flips_up = sum(score("adapter", i) > score("base", i) for i in ids)
        flips_dn = sum(score("adapter", i) < score("base", i) for i in ids)
        out["adapter_vs_base_text_identical"] = same / len(ids)
        md.append(f"\nAdapter output text identical to base on {same}/{len(ids)} items; "
                  f"{flips_up} items fixed, {flips_dn} broken.")
        if "control" in arms:
            same_c = sum(arms["control"][i][0]["text"] == arms["base"][i][0]["text"] for i in ids)
            out["control_vs_base_text_identical"] = same_c / len(ids)
            md.append(f"Control output text identical to base on {same_c}/{len(ids)} items.")
        ex = [i for i in ids if score("adapter", i) > score("base", i)][:3]
        for i in ex:
            md.append(f"  - `{i}` base: `{arms['base'][i][0]['text'][:80]}` → adapter: "
                      f"`{arms['adapter'][i][0]['text'][:80]}`")
    md.append("\nGreedy output text identical (fraction of items):\n")
    names = list(arms)
    md.append("| | " + " | ".join(names) + " |")
    md.append("|---" * (len(names) + 1) + "|")
    ident = {}
    for a in names:
        row = []
        for b in names:
            f = sum(arms[a][i][0]["text"] == arms[b][i][0]["text"] for i in ids) / len(ids)
            ident[f"{a}|{b}"] = f
            row.append(f"{f:.2f}")
        md.append(f"| {a} | " + " | ".join(row) + " |")
    out["text_identical"] = ident
    # The reward grades the LAST <answer>. A user (or any parser taking the first) sees the first one.
    from reward import _ANSWER, countdown
    first = {}
    for nm, it in arms.items():
        first[nm] = {}
        for i in ids:
            tags = _ANSWER.findall(it[i][0]["text"])
            first[nm][i] = countdown(f"<answer>{tags[0]}</answer>", heldout[i]["answer"])[0] if tags else 0.0
    md.append("\nRescored on the FIRST `<answer>` (the training reward takes the last one):\n")
    md.append("| arm | >1 `<answer>` tag | pass, last tag (reward) | pass, first tag |")
    md.append("|---|---|---|---|")
    out["first_tag"] = {}
    for nm, it in arms.items():
        multi = sum(len(_ANSWER.findall(it[i][0]["text"])) > 1 for i in ids) / len(ids)
        f = sum(first[nm].values()) / len(ids)
        out["first_tag"][nm] = {"multi_tag": multi, "pass_first": f}
        md.append(f"| {nm} | {multi:.3f} | {out['arms'][nm]['all']:.3f} | {f:.3f} |")
    for a, b in (("adapter", "base"), ("adapter", "control"), ("merged", "base"), ("random_adapter", "base")):
        if a in arms and b in arms:
            m, lo, hi = paired_ci([first[a][i] - first[b][i] for i in ids])
            out["first_tag"][f"{a}-{b}"] = {"diff": m, "ci95": [lo, hi]}
            md.append(f"- first tag, {a} − {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    rep_v = vdir / "verify_report.json"
    if rep_v.exists():
        v = json.load(open(rep_v))
        out["verify_checks"], out["verify_verdict"] = v.get("checks"), v.get("verdict")
        md.append(f"\nverify.py rules: `{v.get('checks')}` → **{v.get('verdict')}**")
        if "merged_adapter_text_identical" in v:
            md.append(f"Merged vs adapter text identical: {v['merged_adapter_text_identical']:.3f}")
    rep["part3"] = out


def lora_norms(path):
    if not Path(path).exists():
        return None
    ad = torch.load(path, weights_only=False, map_location="cpu")
    sc = ad["alpha"] / ad["r"]
    tot, n = 0.0, 0
    for k, v in ad["lora"].items():
        if k.endswith("_A"):
            B = ad["lora"][k[:-2] + "_B"]
            tot += float((sc * (B.float() @ v.float())).norm() ** 2)
            n += 1
    return {"delta_fro": tot ** 0.5, "pairs": n}


def curves(args, rep, md):
    main, ctrl = by_step(jl(Path(args.main) / "train_log.jsonl")), by_step(jl(Path(args.control) / "train_log.jsonl"))
    if not main:
        return
    md.append("\n## Training curves (true reward; the control trains on shuffled rewards but logs the true ones)\n")
    md.append("| run | steps | true reward, first 20 | last 20 | zero-var groups | grad norm | ‖ΔW‖ (LoRA, fro) |")
    md.append("|---|---|---|---|---|---|---|")
    res = {}
    for nm, rows, ad in (("main", main, Path(args.main) / "adapter.pt"), ("control", ctrl, Path(args.control) / "adapter.pt")):
        if not rows:
            continue
        r = [x["reward_true_mean"] for x in rows]
        zv = st.mean(x["reward"]["zero_std_groups"] for x in rows)
        gn = st.mean(x["grad_norm"] for x in rows)
        ln = lora_norms(ad)
        res[nm] = {"first20": st.mean(r[:20]), "last20": st.mean(r[-20:]), "zero_var": zv, "grad_norm": gn,
                   "lora": ln, "reward": r, "len": [x["len_mean"] for x in rows],
                   "has_box": [x["has_box_frac"] for x in rows]}
        md.append(f"| {nm} | {len(rows)} | {res[nm]['first20']:.3f} | {res[nm]['last20']:.3f} | {zv:.2f} | {gn:.2f} | "
                  f"{ln['delta_fro'] if ln else float('nan'):.3f} |")
    rep["curves"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("reward", "len", "has_box")} for k, v in res.items()}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        for nm, c in (("main", "#2563eb"), ("control", "#9ca3af")):
            if nm in res:
                ax[0].plot(res[nm]["reward"], color=c, alpha=0.25, lw=0.8)
                ax[0].plot(ma(res[nm]["reward"]), color=c, lw=2, label=f"{nm} (10-step mean)")
                ax[1].plot(ma(res[nm]["len"]), color=c, lw=2, label=nm)
        ax[0].set(xlabel="step", ylabel="true reward (batch mean)", title="Reward: main vs shuffled-reward control")
        ax[1].set(xlabel="step", ylabel="completion tokens (10-step mean)", title="Completion length")
        for a in ax:
            a.legend(frameon=False)
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(Path(args.out) / "reward_curves.png", dpi=130)
        md.append("\n![reward curves](reward_curves.png)\n")
    except ImportError:
        md.append("(matplotlib missing: no plot)")


def part1(args, rep, md):
    rows = by_step(jl(Path(args.main) / "train_log.jsonl"))
    if not rows:
        return
    roll = [x["phases"]["rollout"]["s"] for x in rows]
    train = [x["phases"]["train"]["s"] for x in rows]
    step = [a + b for a, b in zip(roll, train)]
    dp = [x["decode_passes"] for x in rows]
    pk = lambda ph, k: max(x["phases"][ph][k] for x in rows)  # noqa: E731
    retries = sum(x["phases"][ph]["alloc_retries"] for x in rows for ph in ("rollout", "train"))
    ov = lambda ph, k: st.mean(x["overlap"][ph][k] for x in rows if x.get("overlap"))  # noqa: E731
    per_pass = [x["decode_s"] / x["decode_passes"] for x in rows if x["decode_passes"]]
    out = {"steps": len(rows), "rollout_s_mean": st.mean(roll), "train_s_mean": st.mean(train),
           "step_s_mean": st.mean(step), "step_s_median": st.median(step), "gen_frac": sum(roll) / sum(step),
           "decode_passes_mean": st.mean(dp), "decode_passes_max": max(dp), "decode_s_per_pass": st.mean(per_pass),
           "prefill_s_mean": st.mean(x["prefill_s"] for x in rows),
           "device_used_peak_mib": max(pk("rollout", "device_used_mib"), pk("train", "device_used_mib")),
           "alloc_peak_rollout_mib": pk("rollout", "alloc_peak_mib"), "alloc_peak_train_mib": pk("train", "alloc_peak_mib"),
           "reserved_peak_mib": max(pk("rollout", "reserved_peak_mib"), pk("train", "reserved_peak_mib")),
           "host_rss_peak_mib": max(pk("rollout", "rss_mib"), pk("train", "rss_mib")), "alloc_retries_total": retries,
           "ooms": sum(x["phases"][ph]["ooms"] for x in rows for ph in ("rollout", "train")),
           "kv_mib": rows[-1]["kv_mib"], "layer_copies_mean": st.mean(x["layer_copies"] for x in rows)}
    for ph in ("rollout", "train"):
        out[f"overlap_{ph}"] = {k: ov(ph, k) for k in ("copy_ms", "stall_ms", "h2d_gbps", "hidden_frac")}
    out["rollout_vs_train_logp_max"] = max(x["rollout_vs_train_max"] for x in rows)
    out["rollout_vs_train_logp_mean"] = st.mean(x["rollout_vs_train_mean"] for x in rows)
    out["truncated_frac_mean"] = st.mean(x["truncated_frac"] for x in rows)
    rep["part1"] = out
    md.append("\n## Part 1: measured (main run) vs predicted (task1_budget/results/countdown_measured_hw.md)\n")
    md.append("| quantity | predicted | measured |")
    md.append("|---|---|---|")
    md.append(f"| step time | 22.2 s (decode to C = 128 every step) | mean {out['step_s_mean']:.1f} s, median {out['step_s_median']:.1f} s |")
    md.append(f"| generation share | 76 % | {100 * out['gen_frac']:.0f} % |")
    md.append(f"| decode passes / step | 128 | mean {out['decode_passes_mean']:.1f}, max {out['decode_passes_max']} (stops when every sequence ends) |")
    md.append(f"| decode time per pass | 16.38 s / 128 = 128 ms (PCIe-bound, 11 streamed layers × 7.9 ms) | {1000 * out['decode_s_per_pass']:.0f} ms |")
    md.append(f"| prefill | 0.56 s | {out['prefill_s_mean']:.2f} s |")
    md.append(f"| train (fwd + bwd) | 5.3 s | {out['train_s_mean']:.1f} s |")
    md.append(f"| GPU peak, nvidia-smi (device-wide) | 7603 MiB at 25 resident (24 used: −193) | {out['device_used_peak_mib']} MiB of 7840 visible |")
    md.append(f"| allocated peak rollout / train | — | {out['alloc_peak_rollout_mib']} / {out['alloc_peak_train_mib']} MiB |")
    md.append(f"| reserved peak | — | {out['reserved_peak_mib']} MiB |")
    md.append(f"| host RSS peak | 10.05 GiB | {out['host_rss_peak_mib'] / 1024:.2f} GiB |")
    md.append(f"| allocator retries / OOMs over the run | 0 | {out['alloc_retries_total']} / {out['ooms']} |")
    for ph in ("rollout", "train"):
        o = out[f"overlap_{ph}"]
        md.append(f"| {ph}: H2D GB/s, copy hidden behind compute | 25.5 GB/s | {o['h2d_gbps']:.1f} GB/s, "
                  f"{100 * o['hidden_frac']:.0f} % hidden (copy {o['copy_ms']:.0f} ms, stall {o['stall_ms']:.0f} ms) |")
    md.append(f"\nOn-policy check: |logp at rollout − logp at training| per token, mean {out['rollout_vs_train_logp_mean']:.4f}, "
              f"max {out['rollout_vs_train_logp_max']:.3f} nats.")


def part2b(args, rep, md):
    rd, mdir = Path(args.resume), Path(args.main)
    got, ref = jl(rd / "samples.jsonl"), jl(mdir / "samples.jsonl")
    if not got:
        return
    n = len(got)
    same_ledger = got == ref[:n]
    ids = [i for x in got for i in x["ids"]]
    tok_same, rew_same = [], []
    for s in range(n):
        a, b = rd / "rollouts" / f"step_{s:06d}.pt", mdir / "rollouts" / f"step_{s:06d}.pt"
        if a.exists() and b.exists():
            ra, rb = torch.load(a, weights_only=False), torch.load(b, weights_only=False)
            tok_same.append(bool(torch.equal(ra["ids"], rb["ids"])))
            rew_same.append(bool(torch.equal(ra["rewards"], rb["rewards"])))
    att = jl(rd / "attempts.jsonl")
    out = {"steps": n, "ledger_equal_to_main": same_ledger, "duplicates": len(ids) - len(set(ids)),
           "steps_contiguous": [x["step"] for x in got] == list(range(n)), "attempts": [a["event"] for a in att],
           "rollout_tokens_equal_per_step": tok_same, "rewards_equal_per_step": rew_same}
    rep["part2b"] = out
    md.append("\n## Part 2b: kill/resume on the GPU (SIGKILL after step 3 train, step 6 rollout, step 8 checkpoint)\n")
    md.append(f"- attempts: {out['attempts']}")
    md.append(f"- sample ledger of {n} steps identical to the uninterrupted main run: **{same_ledger}**; "
              f"duplicates {out['duplicates']}; steps contiguous {out['steps_contiguous']}")
    md.append(f"- rollout tokens identical to main per step: {tok_same}")
    md.append(f"- rewards identical per step: {rew_same}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="runs/main")
    ap.add_argument("--control", default="runs/control")
    ap.add_argument("--verify", default="runs/verify")
    ap.add_argument("--resume", default="runs/resume_demo")
    ap.add_argument("--heldout", default="data/countdown3/heldout")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    Path(args.out).mkdir(exist_ok=True)
    rep, md = {}, ["# Run analysis (generated by analyze.py from raw outputs)\n"]
    part3(args, rep, md)
    curves(args, rep, md)
    part1(args, rep, md)
    part2b(args, rep, md)
    Path(args.out, "analysis.md").write_text("\n".join(md) + "\n")
    json.dump(rep, open(Path(args.out, "analysis.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
