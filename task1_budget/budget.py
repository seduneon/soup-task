#!/usr/bin/env python3
"""Part 1: memory and time budget for one GRPO step and the merge, before running anything.

Every memory term is printed as (name, MiB, formula): the output table is the calculation.
Architecture comes from the checkpoint headers (cached in results/inventory.json), knobs from
config.yaml, and --hw swaps the assumed hardware numbers for bench.py's measurements.

    python budget.py                                    # -> results/predicted.{md,json}
    python budget.py --hw results/measured_hw.yaml --tag predicted_measured_hw
"""
import argparse, copy, functools, json, math, re
from pathlib import Path
from types import SimpleNamespace

import yaml

HERE = Path(__file__).parent
INV = HERE / "results" / "inventory.json"
MiB, GiB = 2**20, 2**30
BUFFERS = 2        # prefetch double buffering
DTYPE_B = {"BF16": 2, "F16": 2, "F32": 4}


# ---------------------------------------------------------------- architecture from checkpoint headers
def read_checkpoint(repo):
    """config.json + safetensors headers only; no weights are downloaded."""
    from huggingface_hub import get_safetensors_metadata, hf_hub_download
    conf = json.load(open(hf_hub_download(repo, "config.json")))
    files = get_safetensors_metadata(repo).files_metadata
    tensors = {n: t for f in files.values() for n, t in f.tensors.items()}
    size = lambda t: math.prod(t.shape) * DTYPE_B[t.dtype]
    layer = lambda n: re.search(r"(?:^|\.)layers\.(\d+)\.(.+)", n)
    layer0 = {layer(n)[2]: t for n, t in tensors.items() if layer(n) and layer(n)[1] == "0"}
    layer_params = sum(math.prod(t.shape) for t in layer0.values())
    n_layers = conf["num_hidden_layers"]
    assert sum(math.prod(t.shape) for n, t in tensors.items() if layer(n)) == n_layers * layer_params
    return dict(conf=conf, layer_params=layer_params, layer_bytes=sum(size(t) for t in layer0.values()),
                linears={k.split(".")[-2]: t.shape for k, t in layer0.items() if k.endswith("_proj.weight")},
                other={n: math.prod(t.shape) for n, t in tensors.items() if not layer(n)},
                total_bytes=sum(size(t) for t in tensors.values()),
                shards={fn: sum(size(t) for t in f.tensors.values()) for fn, f in files.items()})


def load_arch(cfg):
    if not INV.exists():
        inv = {k: read_checkpoint(cfg["models"][k]) for k in ("target", "drafter")}
        t, d, c = inv["target"], inv["drafter"], inv["target"]["conf"]
        h, hd, ff = c["hidden_size"], c["head_dim"], c["intermediate_size"]
        q, kv = c["num_attention_heads"] * hd, c["num_key_value_heads"] * hd
        # The hand derivation must agree with the real checkpoint.
        assert t["layer_params"] == 2 * h * q + 2 * h * kv + 3 * h * ff + 2 * h + 2 * hd, "formula != headers"
        assert c["tie_word_embeddings"] and "lm_head.weight" not in t["other"], "lm_head expected tied"
        assert not any("embed" in n or "lm_head" in n for n in d["other"]), "drafter expected to reuse target's"
        INV.parent.mkdir(exist_ok=True)
        json.dump(inv, open(INV, "w"), indent=1)
    inv = json.load(open(INV))
    t, d, c = inv["target"], inv["drafter"], inv["target"]["conf"]
    return SimpleNamespace(
        L=c["num_hidden_layers"], h=c["hidden_size"], V=c["vocab_size"], ff=c["intermediate_size"],
        q=c["num_attention_heads"] * c["head_dim"], kv=c["num_key_value_heads"] * c["head_dim"],
        layer_params=t["layer_params"], layer_B=t["layer_bytes"], linears=t["linears"],
        embed_B=2 * t["other"]["model.embed_tokens.weight"], shards=t["shards"], total_B=t["total_bytes"],
        Ld=d["conf"]["num_hidden_layers"], drafter_B=d["total_bytes"],
        drafter_params=d["layer_params"] * d["conf"]["num_hidden_layers"] + d["other"]["fc.weight"],
        n_feat=len(d["conf"]["dflash_config"]["target_layer_ids"]))


# ---------------------------------------------------------------- memory
def derived(a, c):
    df = c["dflash"]
    blk = c["block"] if df else 1
    B, S = c["prompts"] * c["G"], c["P"] + c["C"]
    lora_layer = sum(c["lora_rank"] * (o + i) for m, (o, i) in a.linears.items() if m in c["lora_targets"])
    return SimpleNamespace(
        B=B, G=c["G"], P=c["P"], C=c["C"], blk=blk, S=S, Skv=S + (blk if df else 0),
        pmb=min(c["prefill_mb"], B), lora_layer=lora_layer, lora=a.L * lora_layer,
        kv_tok=2 * a.kv * 2 * (a.L + a.Ld * df),  # K+V bytes per token per sequence, target + drafter
        bnd=B * S * a.h * 2,  # one boundary hidden state for all B sequences
        # elems/token live inside one no-grad layer: residual + normed (2h), q, k, v, attn out, gate/up/product
        act=2 * a.h + 2 * a.q + 2 * a.kv + 3 * a.ff,
        # elems/token saved for backward when recomputing one layer: norms in/out + residual (5h),
        # q, k, v pre- and post-RoPE + attn out (3q + 3kv), gate, up, silu(gate), act*up (4ff), LoRA A outputs
        saved=5 * a.h + 3 * a.q + 3 * a.kv + 4 * a.ff + c["lora_rank"] * len(c["lora_targets"]))


def gpu_memory(a, c, r):
    """Live tensors per phase. Rollout is split: prefill and decode buffers are never alive together."""
    d, df, mb, ch = derived(a, c), c["dflash"], c["train_mb"], c["logits_chunk"]
    kv_tok = lambda layers: 2 * layers * a.kv * 2  # K and V, bf16, per token
    always = [
        ("CUDA context + workspaces", c["hw"]["ctx_mib"] * MiB, "assumed / bench.py ctx"),
        ("embedding = tied lm_head", a.embed_B, "V·h·2B"),
        (f"resident layers ({r})", r * a.layer_B, "r · layer"),
        ("prefetch buffers", BUFFERS * a.layer_B * (r < a.L), "2 · layer"),
        ("LoRA weights, fp32", d.lora * 4, f"{d.lora / 1e6:.1f}M · 4B"),
        ("Adam m,v, fp32", d.lora * 8, f"{d.lora / 1e6:.1f}M · 8B"),
        ("DFlash drafter", a.drafter_B * df, "5 layers + fc, bf16"),
    ]
    kv_cache = [("target KV cache", d.B * d.Skv * kv_tok(a.L), "B·(S+blk)·2·L·kv·2B, full length"),
                ("drafter KV cache", d.B * d.Skv * kv_tok(a.Ld) * df, "B·(S+blk)·2·5·kv·2B")]
    phases = {
        "prefill": kv_cache + [
            ("prefill activations, one layer, chunk", d.pmb * d.P * d.act * 2, f"pmb·P·{d.act}·2B"),
            ("target features for drafter", d.B * d.P * a.n_feat * a.h * 2 * df, "B·P·5·h·2B"),
            ("last-token logits", d.B * a.V * 6, "B·V·(2+4)B"),
        ],
        "decode": kv_cache + [
            ("target features, one block", d.B * d.blk * a.n_feat * a.h * 2 * df, "B·blk·5·h·2B"),
            ("draft/verify logits + probs" if df else "decode logits", d.B * d.blk * a.V * (10 if df else 6),
             "B·blk·V·(2+4+4)B: bf16 logits, fp32 target + draft probs" if df else "B·V·(2+4)B"),
        ],
        "ref": [
            ("boundary hidden in+out", 2 * d.bnd, "2·B·S·h·2B"),
            ("activations, one layer, micro-batch", mb * d.S * d.act * 2, f"mb·S·{d.act}·2B"),
            ("logits chunk, no grad", ch * a.V * 10, "chunk·V·(2+4+4)B"),
        ],
        "train": [
            ("LoRA grads, fp32", d.lora * 4, "lora · 4B, set to None outside train"),
            ("boundary checkpoints on GPU", (2 if c["offload_checkpoints"] else a.L + 1) * d.bnd,
             "2·B·S·h·2B, rest on host" if c["offload_checkpoints"] else "(L+1)·B·S·h·2B"),
            ("dL/dh in+out", 2 * d.bnd, "2·B·S·h·2B"),
            ("recompute one layer, saved for backward", mb * d.S * d.saved * 2, f"mb·S·{d.saved}·2B"),
            ("logits chunk with grad", ch * a.V * 14,
             "chunk·V·14B: bf16 logits, fp32 upcast, log-softmax, grad (Exact Layer Streaming §5.6)"),
        ],
    }
    return {k: always + v for k, v in phases.items()}


def merge_and_host(a, c, r):
    d, base, shard = derived(a, c), c["hw"]["host_base_gib"] * GiB, max(a.shards.values())
    o, i = max((a.linears[m] for m in c["lora_targets"]), key=math.prod)
    return {
        "merge GPU": [("CUDA context", c["hw"]["ctx_mib"] * MiB, "assumed"),
                      ("one tensor: bf16 in, fp32, fp32 delta, bf16 out", 12 * o * i, f"{o}×{i}·(2+4+4+2)B")],
        "host, training": [("Python + torch + CUDA host", base, "assumed"),
                           ("pinned streamed layers", (a.L - r) * a.layer_B, "(L−r) · layer"),
                           ("pinned checkpoints", (a.L + 1) * d.bnd * c["offload_checkpoints"], "(L+1)·B·S·h·2B"),
                           ("one shard while loading", shard, "largest safetensors shard")],
        "host, merge": [("Python + torch", base, "assumed"),
                        ("input shard (mmap) + output shard buffer", 2 * shard, "2 · largest shard"),
                        ("LoRA fp32", d.lora * 4, "lora · 4B")],
    }


total = lambda rows: sum(b for _, b, _ in rows)


def fit(a, c):
    """Largest resident count for which every phase fits under cap − reserve (None if none fits)."""
    limit = (c["cap_mib"] - c["reserve_mib"]) * MiB
    ok = [r for r in range(a.L + 1) if max(map(total, gpu_memory(a, c, r).values())) <= limit]
    return max(ok) if ok else None


# ---------------------------------------------------------------- time
def decode_residency(a, c, r, passes, per_pass):
    """Resident count at each decode pass. regular: r throughout. dynamic: the KV cache grows from P to S tokens,
    and layers are kept resident in the memory it has not claimed yet, evicted as it grows (never below r)."""
    if c["streaming"] != "dynamic":
        return [r] * passes
    d, limit = derived(a, c), (c["cap_mib"] - c["reserve_mib"]) * MiB
    kv_full = d.B * d.Skv * d.kv_tok
    base = total(gpu_memory(a, c, 0)["decode"]) - BUFFERS * a.layer_B - kv_full  # decode phase w/o layers or KV
    out = []
    for i in range(passes):
        kv_now = d.B * min(d.Skv, d.P + (i + 1) * per_pass + d.blk) * d.kv_tok
        free = limit - base - kv_now
        rr = max(rr for rr in range(a.L + 1) if rr * a.layer_B + BUFFERS * a.layer_B * (rr < a.L) <= free) \
            if free >= BUFFERS * a.layer_B else 0
        out.append(max(r, rr))
    return out


def step_time(a, c, r):
    """Per layer visit: compute = max(FLOPs / TFLOPS, weight bytes / VRAM bandwidth) (roofline);
    streamed layers cost max(copy, compute), resident ones compute; every visit pays a launch overhead."""
    d, hw, df = derived(a, c), c["hw"], c["dflash"]
    vram, h2d, ovh, tf = hw["vram_gbps"] * 1e9, hw["h2d_gbps"] * 1e9, hw["overhead_ms"] / 1e3, hw["tflops"] * 1e12
    copy = a.layer_B / h2d
    head = lambda tok: max(2 * a.V * a.h * tok / tf, a.embed_B / vram)
    draft = (max(2 * a.drafter_params * d.B * d.blk / tf, a.drafter_B / vram) + a.Ld * ovh + head(d.B * d.blk)) if df else 0
    per_pass = c["tau"] if df else 1
    n_dec = math.ceil(d.C / per_pass)
    rows = [  # phase, tokens per pass, passes, FLOP/param/token, attention context, extra seconds per pass
        ("prefill", d.B * d.P, 1, 2, d.P / 2, head(d.B)),
        ("decode, DFlash verify" if df else "decode", d.B * d.blk, n_dec, 2, d.P + d.C / 2, head(d.B * d.blk) + draft),
        ("ref log-probs", d.B * d.S, 1, 2, d.S / 2, head(d.B * d.C)),
        ("train forward", d.B * d.S, 1, 2, d.S / 2, head(d.B * d.C)),
        ("train backward (recompute + dX)", d.B * d.S, 1, 4, d.S / 2, head(d.B * d.C)),
    ]
    out = []
    for name, tok, n, mult, ctx, extra in rows:
        flop = mult * (a.layer_params + 2 * a.q * ctx) * tok / tf
        mem = (mult / 2) * a.layer_B / vram  # weights read once per forward-equivalent
        comp = max(flop, mem)
        visit = lambda rr: (a.L - rr) * max(copy, comp) + rr * comp + a.L * ovh
        res = decode_residency(a, c, r, n, per_pass) if name.startswith("decode") else [r] * n
        t = sum(visit(rr) for rr in res) + n * extra
        bound = ("overhead" if a.L * ovh > visit(r) / 2 else "PCIe" if comp < copy and r < a.L
                 else "VRAM" if mem > flop else "FLOPs")
        out.append(dict(phase=name, tokens=tok, passes=n, mult=mult, s=t, bound=bound,
                        resident=f"{min(res)}–{max(res)}, mean {sum(res) / n:.1f}" if min(res) != max(res) else r,
                        layer_ms=f"copy {copy * 1e3:.1f} / FLOPs {flop * 1e3:.2f} / VRAM {mem * 1e3:.2f}"))
    return out


# ---------------------------------------------------------------- report
def evaluate(a, c, r=None):
    """Summary at resident count r (default: the largest that fits)."""
    r = fit(a, c) if r is None else r
    t = step_time(a, c, r or 0)
    peak = {k: total(v) / MiB for k, v in gpu_memory(a, c, r or 0).items()}
    gen = sum(x["s"] for x in t if x["phase"].startswith(("prefill", "decode")))
    step = sum(x["s"] for x in t)
    B = derived(a, c).B
    return dict(resident=r, peak_mib=peak, binding=max(peak, key=peak.get), time=t, step_s=step, gen_s=gen,
                prompts=c["prompts"], B=B, s_per_prompt=step / c["prompts"], s_per_seq=step / B)


def with_overrides(c, over):
    c = copy.deepcopy(c)
    for key, value in over.items():
        *path, last = key.split(".")
        functools.reduce(dict.__getitem__, path, c)[last] = value
    return c


def tau_breakeven(a, c):
    """Smallest τ at which DFlash (with its smaller residency) beats plain decode; None if never."""
    c = with_overrides(c, {"dflash": True})
    plain = evaluate(a, with_overrides(c, {"dflash": False}))["step_s"]
    step = lambda tau: evaluate(a, with_overrides(c, {"tau": tau}))["step_s"]
    lo, hi = 1.0, float(c["block"])
    if step(hi) >= plain:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if step(mid) >= plain else (lo, mid)
    return hi


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows] + [""])


mib = lambda b: f"{b / MiB:,.0f}"


def phase_table(mem):
    """One row per term, one column per phase: persistent terms show up in every column."""
    vals = {k: {n: b for n, b, _ in rows} for k, rows in mem.items()}
    formula = {n: f for rows in mem.values() for n, _, f in rows}  # ordered, de-duplicated
    body = [[n, *(mib(vals[k][n]) if n in vals[k] else "" for k in mem), f]
            for n, f in formula.items() if any(vals[k].get(n) for k in mem)]
    return table(["term", *mem, "formula"], body + [["**total**", *(f"**{mib(total(v))}**" for v in mem.values()), ""]])


def report(a, c, tag):
    e, d = evaluate(a, c), derived(a, c)
    r, limit = e["resident"] or 0, c["cap_mib"] - c["reserve_mib"]
    gain = e["step_s"] - evaluate(a, c, r + 1)["step_s"] if r < a.L else 0.0
    pcie = sum(x["passes"] for x in e["time"] if x["tokens"]) * (a.L - r) * a.layer_B
    be = tau_breakeven(a, c)
    scenarios = {n: evaluate(a, with_overrides(c, o)) for n, o in {"default": {}, **c.get("scenarios", {})}.items()}
    summary = lambda label, s: [label, "does not fit" if s["resident"] is None else s["resident"],
                                f"{s['peak_mib'][s['binding']]:,.0f} ({s['binding']})",
                                "yes" if max(s["peak_mib"].values()) <= limit else "no",
                                f"{s['step_s']:.1f}", f"{s['gen_s']:.1f}", f"{s['step_s'] - s['gen_s']:.1f}",
                                f"{s['s_per_prompt']:.1f}"]
    summary_head = ["", "resident", "peak MiB (binding)", "fits", "step s", "gen s", "train s", "s/prompt"]
    sweep = sorted({0, r // 2, r, r + 1, a.L} & set(range(a.L + 1)))
    batch = [evaluate(a, with_overrides(c, {"prompts": p})) for p in (1, 2, 3, 4)]
    batch_rows = [[x["prompts"], x["B"], "does not fit" if x["resident"] is None else x["resident"],
                   f"{x['peak_mib'][x['binding']]:,.0f} ({x['binding']})", f"{x['step_s']:.1f}",
                   f"{x['s_per_prompt']:.1f}", f"{x['s_per_seq']:.2f}",
                   f"{3600 / x['step_s']:.0f}"] for x in batch]
    weights = [("one decoder layer", a.layer_B), (f"{a.L} layers", a.L * a.layer_B), ("embedding = tied lm_head", a.embed_B),
               ("target total", a.total_B), ("DFlash drafter", a.drafter_B), ("largest shard", max(a.shards.values()))]
    md = [f"# Part 1 budget: {tag}\n",
          f"Cap {c['cap_mib']} MiB on nvidia-smi, {c['reserve_mib']} MiB reserve. Hardware: `{c['hw']}`.\n",
          f"Streaming `{c['streaming']}`, DFlash {'on' if c['dflash'] else 'off'}. "
          f"Symbols: prompts={c['prompts']}, G={d.G}, B=prompts·G={d.B} sequences, P={d.P}, C={d.C}, S=P+C={d.S}, "
          f"blk={d.blk}, τ={c['tau']}, L={a.L}, h={a.h}, kv={a.kv}, V={a.V}, layer={a.layer_B / MiB:.1f} MiB, "
          f"pmb={d.pmb}, mb={c['train_mb']}, chunk={c['logits_chunk']}.\n",
          "## Weights (from safetensors headers)\n", table(["", "MiB"], [[k, f"{b / MiB:,.1f}"] for k, b in weights]),
          f"## GPU memory per phase: **{r} resident / {a.L - r} streamed**, binding phase **{e['binding']}**\n",
          phase_table(gpu_memory(a, c, r))]
    for k, rows in merge_and_host(a, c, r).items():
        md += [f"### {k}: {total(rows) / GiB:.2f} GiB\n", table(["term", "MiB", "formula"], [[n, mib(b), f] for n, b, f in rows if b])]
    md += ["## Step time\n",
           table(["phase", "tokens/pass", "passes", "resident", "FLOP/param/tok", "per layer ms", "bound", "s"],
                 [[x["phase"], x["tokens"], x["passes"], x["resident"], x["mult"], x["layer_ms"], x["bound"],
                   f"{x['s']:.2f}"] for x in e["time"]]),
           f"**Step {e['step_s']:.1f} s, generation {100 * e['gen_s'] / e['step_s']:.0f}%, {e['s_per_prompt']:.1f} s/prompt.** "
           f"PCIe traffic {pcie / GiB:.0f} GiB/step; one more resident layer saves {gain:.2f} s/step. "
           + (f"DFlash arm break-even τ = {be:.2f} (assumed τ = {c['tau']}).\n" if be else
              "The DFlash arm never beats plain decode here.\n"),
           "## Batch sweep (prompts per step, best residency for each)\n",
           table(["prompts", "B", "resident", "peak MiB (binding)", "step s", "s/prompt", "s/seq", "steps/h"], batch_rows),
           "## Scenarios\n", table(summary_head, [summary(n, s) for n, s in scenarios.items()]),
           "## Residency sweep\n", table(summary_head[1:], [summary(rr, evaluate(a, c, rr))[1:] for rr in sweep])]
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"{tag}.md").write_text("\n".join(md))
    json.dump(dict(scenarios=scenarios, batch=batch, tau_breakeven=be, hw=c["hw"]),
              open(out / f"{tag}.json", "w"), indent=1)
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=HERE / "config.yaml")
    ap.add_argument("--hw", help="bench.py output; its hw section overrides the assumed numbers")
    ap.add_argument("--tag", default="predicted")
    args = ap.parse_args()
    c = yaml.safe_load(open(args.config))
    if args.hw:
        c["hw"].update(yaml.safe_load(open(args.hw))["hw"])
    report(load_arch(c), c, args.tag)


if __name__ == "__main__":
    main()
