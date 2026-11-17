#!/usr/bin/env python3
"""Measure the hardware numbers budget.py assumes, on the rented GPU, with nvidia-smi sampled throughout.

    python bench.py                 # ctx pcie vram gemm overhead -> results/measured_hw.yaml
    python bench.py pcie gemm       # a subset (ctx is only valid as the first CUDA use of a process)

Run it with the same environment variables as training (CUDA_MODULE_LOADING, PYTORCH_CUDA_ALLOC_CONF):
the context size depends on them.

ctx       CUDA context + cuBLAS/SDPA workspaces, as device-wide nvidia-smi memory.used (what the cap is judged on;
          per-PID numbers are often missing inside containers)
pcie      H2D of one real layer (192.5 MiB) on a side stream while GEMMs run: the overlap condition L2L sees.
          Median, not idle best. Sources: pinned, pageable, and a numpy memmap (how a safetensors slice looks
          when it is not copied into pinned memory first). host_ms = time until copy_() returns: ~0 means truly
          async, ~copy time means the "non_blocking" copy silently ran synchronously and overlap is gone.
          Also D2H (checkpoint offload) and the PCIe link generation/width while the link is busy.
vram      effective weight-read bandwidth at decode size (B = 16 tokens through every linear of one layer)
gemm      bf16 TFLOPS at the real layer shapes for 16 (decode) and 12288 (training = B·S) tokens, SM clock logged
overhead  launch + Python cost per layer visit, measured with TINY weights and LoRA on all 7 linears, so no
          weight read is included. A full-size decode layer is reported alongside: it should be close to
          overhead + layer bytes / vram_gbps, which checks the roofline in budget.py
"""
import datetime, statistics, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

OUT = Path(__file__).parent / "results"
H, NQ, NKV, HD, FF = 2560, 32, 8, 128, 9728
LAYER_ELEMS = 100_930_816
BF16 = torch.bfloat16
CAP_MIB = 8192
# (out, in, copies per layer): k~v and gate~up share shapes
SHAPES = {"q_proj": (4096, H, 1), "k_proj": (1024, H, 2), "o_proj": (H, 4096, 1),
          "gate_proj": (FF, H, 2), "down_proj": (H, FF, 1)}


def smi(query):
    out = subprocess.check_output(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"])
    return out.decode().strip().splitlines()[0].split(", ")


def used_mib():
    torch.cuda.synchronize()
    time.sleep(0.5)  # nvidia-smi lags the allocation
    return int(smi("gpu=memory.used")[0])


def clocks():
    return dict(zip(("sm_mhz", "temp_c", "pstate"), smi("gpu=clocks.sm,temperature.gpu,pstate")))


def rand(*shape):
    return torch.randn(*shape, device="cuda", dtype=BF16)


def gpu_ms(fn, n=50):
    for _ in range(3):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def ctx():
    stages = {"device_before": int(smi("gpu=memory.used")[0])}
    torch.zeros(1, device="cuda")
    stages["context"] = used_mib()
    a, w = rand(2048, H), rand(FF, H)
    (a @ w.T).sum().item()
    stages["first_bf16_matmul"] = used_mib()
    q, k = rand(8, NQ, 16, HD), rand(8, NKV, 768, HD)
    F.scaled_dot_product_attention(q, k, k, enable_gqa=True)
    stages["first_sdpa"] = used_mib()
    del a, w, q, k
    torch.cuda.empty_cache()
    stages["after_free"] = used_mib()  # what is left is not our tensors: context + workspaces (+ device baseline)
    return {"ctx_mib": stages["after_free"]}, stages


def pcie():
    pinned, pageable = torch.empty(LAYER_ELEMS, dtype=BF16, pin_memory=True), torch.empty(LAYER_ELEMS, dtype=BF16)
    dev, side = torch.empty(LAYER_ELEMS, dtype=BF16, device="cuda"), torch.cuda.Stream()
    a, w = rand(6144, H), rand(FF, H)

    def run(dst, src, busy=False, reps=30):
        gb, host = [], []
        for _ in range(reps):
            if busy:
                for _ in range(4):
                    a @ w.T  # queued on the default stream, runs while the copy runs
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            with torch.cuda.stream(side):
                s.record()
                dst.copy_(src, non_blocking=True)
                e.record()
            host.append((time.perf_counter() - t0) * 1e3)
            torch.cuda.synchronize()
            gb.append(2 * LAYER_ELEMS / (s.elapsed_time(e) / 1e3) / 1e9)
        return {"median_gbps": round(statistics.median(gb), 2), "best_gbps": round(max(gb), 2),
                "host_ms": round(statistics.median(host), 3), "src_pinned": src.is_pinned()}

    def link_under_load():  # links downshift at idle, so read gen/width while copies are queued
        with torch.cuda.stream(side):
            for _ in range(50):
                dev.copy_(pinned, non_blocking=True)
        cur = smi("gpu=pcie.link.gen.current,pcie.link.width.current")
        torch.cuda.synchronize()
        return {"current_gen_width": cur, "max_gen_width": smi("gpu=pcie.link.gen.max,pcie.link.width.max")}

    with tempfile.TemporaryDirectory() as tmp:
        mm = np.memmap(Path(tmp) / "layer.bin", dtype=np.int16, mode="w+", shape=(LAYER_ELEMS,))
        mm[:] = 1
        mm.flush()
        mapped = torch.from_numpy(np.memmap(mm.filename, dtype=np.int16, mode="c", shape=(LAYER_ELEMS,))).view(BF16)
        run(dev, pinned, reps=3)
        r = {"h2d_pinned_idle": run(dev, pinned), "h2d_pinned_under_gemm": run(dev, pinned, busy=True),
             "h2d_pageable_under_gemm": run(dev, pageable, busy=True, reps=10),
             "h2d_mmap_under_gemm (page cache warm)": run(dev, mapped, busy=True, reps=10),
             "d2h_pinned": run(pinned, dev), "link": link_under_load(), "clocks": clocks()}
        del mapped, mm
    return {"h2d_gbps": r["h2d_pinned_under_gemm"]["median_gbps"]}, r


def vram():
    w = {k: rand(o, i) for k, (o, i, _) in SHAPES.items()}
    layer_bytes = sum(n * o * i * 2 for o, i, n in SHAPES.values())
    r = {}
    x = {k: rand(16, i) for k, (o, i, _) in SHAPES.items()}
    ms = sum(n * gpu_ms(lambda k=k: x[k] @ w[k].T) for k, (_, _, n) in SHAPES.items())
    r["layer_read_gbps_16tok"] = round(layer_bytes / (ms / 1e3) / 1e9, 1)
    r["clocks"] = clocks()
    return {"vram_gbps": r["layer_read_gbps_16tok"]}, r


def gemm():
    before, tf = clocks(), {}
    for tok in (16, 12288):
        for name, (o, i, _) in SHAPES.items():
            x, w = rand(tok, i), rand(o, i)
            tf[f"{tok}/{name}"] = round(2 * tok * o * i / (gpu_ms(lambda: x @ w.T) / 1e3) / 1e12, 2)

    def layer_rate(tok):  # effective rate for a whole layer: total FLOPs / total time over its linears
        flops = sum(n * 2 * tok * o * i for o, i, n in SHAPES.values())
        secs = sum(n * 2 * tok * o * i / (tf[f"{tok}/{k}"] * 1e12) for k, (o, i, n) in SHAPES.items())
        return round(flops / secs / 1e12, 2)

    return ({"tflops": layer_rate(12288)},
            {"tflops": tf, "layer_tflops_16tok": layer_rate(16), "clocks_before": before, "clocks_after": clocks()})


def make_layer(h, nq, nkv, hd, ff, rank=0):
    """Qwen3 decoder layer shape with random weights, optional LoRA on all 7 linears."""
    shapes = dict(q=(nq * hd, h), k=(nkv * hd, h), v=(nkv * hd, h), o=(h, nq * hd), g=(ff, h), u=(ff, h), d=(h, ff))
    W = {k: rand(*s) * 0.02 for k, s in shapes.items()}
    A = {k: rand(rank, s[1]) * 0.02 for k, s in shapes.items()} if rank else {}
    Bm = {k: rand(s[0], rank) * 0.02 for k, s in shapes.items()} if rank else {}
    lin = lambda x, k: x @ W[k].T + ((x @ A[k].T) @ Bm[k].T if rank else 0)
    norm = lambda x: F.rms_norm(x, x.shape[-1:])

    def layer(x, cache):  # attention over a KV cache of len(cache)
        b, t, _ = x.shape
        y = norm(x)
        q = norm(lin(y, "q").view(b, t, nq, hd)).transpose(1, 2)
        k = torch.cat([cache, norm(lin(y, "k").view(b, t, nkv, hd)).transpose(1, 2)], 2)
        v = torch.cat([cache, lin(y, "v").view(b, t, nkv, hd).transpose(1, 2)], 2)
        x = x + lin(F.scaled_dot_product_attention(q, k, v, enable_gqa=True).transpose(1, 2).reshape(b, t, -1), "o")
        y = norm(x)
        return x + lin(F.silu(lin(y, "g")) * lin(y, "u"), "d")

    return layer, (nkv, hd), h


def wall_ms(spec, b, t, ctx_len, n=50):  # wall clock, so launch + Python cost is included
    layer, (nkv, hd), h = spec
    x, cache = rand(b, t, h), rand(b, nkv, ctx_len, hd)
    with torch.no_grad():
        for _ in range(3):
            layer(x, cache)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            layer(x, cache)
        torch.cuda.synchronize()
    return round((time.perf_counter() - t0) / n * 1e3, 3)


def overhead():
    tiny_lora, full = make_layer(64, 2, 1, 32, 128, rank=16), make_layer(H, NQ, NKV, HD, FF)
    r = {"tiny_lora_b16_t1_ctx512": wall_ms(tiny_lora, 16, 1, 512),
         "full_decode_b16_t1_ctx512": wall_ms(full, 16, 1, 512), "clocks": clocks()}
    return {"overhead_ms": r["tiny_lora_b16_t1_ctx512"]}, r


def main():
    order = ["ctx", "pcie", "vram", "gemm", "overhead"]
    which = [b for b in order if b in (sys.argv[1:] or order)]
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sampler = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,temperature.gpu,pcie.link.gen.current,pcie.link.width.current",
         "--format=csv", "-lms", "200"], stdout=open(OUT / "logs" / f"nvsmi-bench-{ts}.csv", "w"))
    path = OUT / "measured_hw.yaml"
    data = yaml.safe_load(open(path)) if path.exists() else {}
    name, total_mib, driver = smi("gpu=name,memory.total,driver_version")
    data["gpu"] = [name, total_mib, driver]
    # On a card with more than 8 GB the cap is ours to enforce and must be proven from nvidia-smi.
    data["cap_enforced_by_hardware"] = int(total_mib) <= CAP_MIB
    if not data["cap_enforced_by_hardware"]:
        print(f"WARNING: {name} has {total_mib} MiB; the {CAP_MIB} MiB cap is NOT enforced by the hardware")
    try:
        for b in which:
            hw, details = globals()[b]()
            print(f"== {b}: {hw}\n{details}")
            data.setdefault("hw", {}).update(hw)
            data.setdefault("details", {})[b] = {"at": datetime.datetime.now().isoformat(timespec="seconds"),
                                                 **details}
    finally:
        sampler.terminate()
        yaml.safe_dump(data, open(path, "w"), sort_keys=False)
    print(f"wrote {path}\nnext: python budget.py --hw results/measured_hw.yaml --tag predicted_measured_hw")


if __name__ == "__main__":
    main()
