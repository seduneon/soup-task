#!/usr/bin/env python3
"""GRPO + LoRA on the streamed engine, with deterministic resume.

    python train.py --config train_config.yaml                       # start, or resume if a checkpoint exists
    python train.py --config train_config.yaml --set steps=3 G=4     # overrides (YAML values)

One step: prompts from the disk stream -> rollouts (adapter on, sampled) -> rewards -> rollout batch
written to disk and read back -> reference log-probs (adapter off, only if beta > 0) -> layer-streamed
train pass -> Adam on LoRA -> log line -> ledger -> checkpoint. Phase times and memory are logged per step;
nvidia-smi runs for the whole job.

Out dir: checkpoint.pt (atomic), adapter.pt, train_log.jsonl (raw, append-only, every attempt), samples.jsonl
(consumed-sample ledger, committed steps only), attempts.jsonl (start / resume events), rollouts/,
logs/nvsmi-*.csv.

Negative control (Part 3): --set reward_shuffle=true permutes the rewards across the whole batch (seeded per
step, so resume is still exact) before the advantages; reward_true_mean logs the unshuffled reward. lr=0 is
NOT a usable control here: B starts at 0, so an lr=0 adapter is exactly the base model.

Test hook: L2L_KILL_AT="3:train,5:rollout" SIGKILLs the process once at that step and point.
"""
import argparse
import json
import os
import signal
import time
from pathlib import Path

# Before torch touches CUDA. A/B on the 3070 at 24 resident: allocator retries 2-6/step -> 0, rewards and grad
# norms identical, step time unchanged (README). Set here, not in a launch command, so no run can miss it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml

from data_stream import PromptStream, RolloutStore, atomic_torch_save, atomic_write_bytes
from decode import generate, memory_plan
from engine import StreamedQwen3
from grpo import completion_targets, diagnostics, group_advantages, make_loss, rollout_logp_grid, token_coef
from lora import LoRAParams, lora_shapes
from monitor import MiB, NvSmi, Phases
from reward import PROMPT_SUFFIXES, REWARDS
from store import LayerStore

HERE = Path(__file__).parent


def load_config(path, overrides):
    cfg = yaml.safe_load(open(path))
    for kv in overrides:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def model_dir(name):
    p = Path(name)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(name, allow_patterns=["*.json", "*.safetensors", "*.txt"]))


def build(cfg):
    from transformers import AutoConfig, AutoTokenizer
    root = model_dir(cfg["model"])
    mcfg = AutoConfig.from_pretrained(root)
    if cfg.get("layers"):
        mcfg.num_hidden_layers = cfg["layers"]
    mcfg._attn_implementation = "sdpa"
    dev, dt = torch.device(cfg["device"]), getattr(torch, cfg["dtype"])
    store = LayerStore.from_safetensors(root, mcfg.num_hidden_layers, dt)
    if cfg.get("merge_adapter"):  # in-memory merge into the store's dtype (evaluate.py --merge-in-memory)
        cfg["merge_stats"] = store.merge_lora(cfg["merge_adapter"])
    torch.manual_seed(cfg["seed"])  # LoRA A init
    lora = LoRAParams(lora_shapes(mcfg), mcfg.num_hidden_layers, r=cfg["lora_r"], alpha=cfg["lora_alpha"], device=dev)
    eng = StreamedQwen3(mcfg, store, lora, dev, resident=range(cfg["resident"]), n_buffers=cfg["n_buffers"],
                       release_host=True)
    tok = AutoTokenizer.from_pretrained(root)
    return mcfg, tok, eng, lora


def encode(tok, question, style="cot"):
    text = tok.apply_chat_template([{"role": "user", "content": question + PROMPT_SUFFIXES[style]}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False).input_ids


def kill_points():
    return {tuple(x.split(":")) for x in os.environ.get("L2L_KILL_AT", "").split(",") if x}


def maybe_kill(out, step, where, points):
    if (str(step), where) in points:
        marker = out / f".killed_{step}_{where}"
        if not marker.exists():
            marker.touch()
            os.kill(os.getpid(), signal.SIGKILL)


def say(msg):
    """Console progress for a person watching (tmux / log tail); train_log.jsonl stays the record."""
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=HERE / "train_config.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = Path(cfg["out_dir"])
    (out / "logs").mkdir(parents=True, exist_ok=True)
    nvsmi = NvSmi(out / "logs" / f"nvsmi-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.csv")
    try:
        run(cfg, out)
    finally:
        nvsmi.stop()


def run(cfg, out):
    stream = PromptStream(cfg["data_dir"], seed=cfg["seed"])  # before the model load: a bad path fails in seconds
    say(f"loading {cfg['model']} (resident {cfg['resident']}, streaming {cfg['streaming']}) -> {out}")
    mcfg, tok, eng, lora = build(cfg)
    dev = eng.device
    eng.profile = bool(cfg.get("profile_overlap")) and eng.cuda
    opt = torch.optim.AdamW(lora.parameters(), lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=0.0)
    gen = torch.Generator(device=dev).manual_seed(cfg["seed"])
    rstore, points = RolloutStore(out / "rollouts"), kill_points()
    phases = Phases(dev, cap_mib=cfg["cap_mib"], strict=cfg.get("strict_cap", False),
                    offset_mib=cfg.get("cap_offset_mib", 0))
    reward_fn = REWARDS[cfg["reward"]]
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    pad = tok.pad_token_id
    ckpt_path, ledger = out / "checkpoint.pt", out / "samples.jsonl"
    plan = None
    if cfg["streaming"] == "dynamic":
        assert dev.type == "cuda", "the dynamic residency plan reads CUDA allocator state"
        plan = memory_plan(eng, range(cfg["resident"]), (cfg["cap_mib"] - cfg["reserve_mib"]) * MiB,
                           cfg["headroom_mib"] * MiB)

    step0, ck_step = 0, -1
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        lora.load_state_dict(ck["lora"])
        opt.load_state_dict(ck["opt"])
        stream.load_state_dict(ck["stream"])
        gen.set_state(ck["gen"])
        step0 = ck_step = ck["step"]
        step0 += 1
    if ledger.exists():  # drop ledger lines of uncommitted steps (all of them if no checkpoint): they replay
        keep = [l for l in open(ledger) if json.loads(l)["step"] <= ck_step]
        atomic_write_bytes(ledger, "".join(keep).encode())
    append_jsonl(out / "attempts.jsonl", {"time": time.time(), "pid": os.getpid(),
                                          "event": "resume" if step0 else "start", "step": step0,
                                          "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
                                          "stream": stream.state_dict()})
    say(f"{'resume' if step0 else 'start'} at step {step0} of {cfg['steps']}"
        f"{' (reward_shuffle control)' if cfg.get('reward_shuffle') else ''}")
    t_run = time.time()

    for step in range(step0, cfg["steps"]):
        t_step = time.time()
        say(f"step {step}: rollout ({cfg['prompts']} prompts x G={cfg['G']}, max_new {cfg['max_new']})")
        recs = stream.next(cfg["prompts"])
        prompts = [encode(tok, r["prompt"], cfg.get("prompt_style", "cot")) for r in recs for _ in range(cfg["G"])]
        too_long = {r["id"]: len(p) for r, p in zip(recs, prompts[::cfg["G"]]) if len(p) > cfg["max_prompt"]}
        assert not too_long, (f"prompts longer than max_prompt={cfg['max_prompt']} tokens {too_long}: the memory "
                              f"plan assumes P <= max_prompt (filter with prepare_data.py --max-prompt-tokens)")
        loads0, passes0 = eng.loads, eng.passes
        with phases("rollout"):
            ro = generate(eng, prompts, cfg["max_new"], eos, pad, temperature=cfg["temperature"], top_p=cfg["top_p"],
                          generator=gen, prefill_mb=cfg["prefill_mb"], kv="paged" if plan else "static", plan=plan)
        ov_rollout = eng.overlap_report()
        if plan is not None:  # dynamic arm: back to the regular residency, so ref/train time is comparable (H6)
            eng.set_resident(range(cfg["resident"]))
        maybe_kill(out, step, "rollout", points)
        texts = [tok.decode(ro.ids[b, ro.P:][ro.comp[b, ro.P:]], skip_special_tokens=True) for b in range(len(prompts))]
        scored = [reward_fn(t, recs[b // cfg["G"]]["answer"]) for b, t in enumerate(texts)]
        rewards = torch.tensor([s[0] for s in scored], dtype=torch.float32)
        train_rewards = rewards
        if cfg.get("reward_shuffle"):  # Part 3 negative control: same pipeline, reward unrelated to the completion
            perm = torch.randperm(len(rewards), generator=torch.Generator().manual_seed(cfg["seed"] * 1000003 + step))
            train_rewards = rewards[perm]
        rstore.save(step, {"train_rewards": train_rewards, "ids": ro.ids, "attn": ro.attn, "comp": ro.comp, "logp": ro.logp, "P": ro.P,
                           "finished": ro.finished, "rewards": rewards, "texts": texts,
                           "sample_ids": [r["id"] for r in recs], "infos": [s[1] for s in scored]})
        data = rstore.load(step)  # the training passes read the rollout batch back from disk
        ids, attn, comp = data["ids"], data["attn"], data["comp"]
        targets = completion_targets(ids, comp)
        adv, adv_stats = group_advantages(data["train_rewards"], cfg["G"], scale_std=cfg["adv_std"])
        say(f"step {step}: rollout done in {time.time() - t_step:.0f} s, reward {float(rewards.mean()):.3f}; training")
        ref = None
        if cfg["beta"] > 0:
            with phases("ref"), eng.adapter_off():
                ref = eng.logprobs(ids, attn, targets, mb=cfg["train_mb"], chunk=cfg["chunk"]).cpu()
        stash = torch.zeros(ids.numel(), device=dev)  # same device as the indices the loss receives
        with phases("train"):
            lora.zero_grad(set_to_none=True)
            eng.train_grads(ids, attn, targets, make_loss(token_coef(targets, cfg["norm"]), adv, ref, cfg["beta"], stash),
                            mb=cfg["train_mb"], chunk=cfg["chunk"])
            grad_norm = float(torch.nn.utils.clip_grad_norm_(lora.parameters(), cfg["clip"]))
            opt.step()
            lora.zero_grad(set_to_none=True)  # grads exist only inside the train phase
        ov_train = eng.overlap_report()  # ref + train passes
        maybe_kill(out, step, "train", points)
        diag = diagnostics(targets, stash.view(ids.shape).cpu(), rollout_logp_grid(ro, ids.shape[1]), ref)
        lengths = comp.sum(-1).float()
        ph = phases.rec
        done, left = step - step0 + 1, cfg["steps"] - step - 1
        eta = (time.time() - t_run) / done * left
        say(f"step {step} done in {time.time() - t_step:.0f} s | reward {float(data['rewards'].mean()):.3f} "
            f"zero-var groups {adv_stats['zero_std_groups']:.2f} | grad {grad_norm:.3g} | "
            f"len {float(lengths.mean()):.0f} trunc {float((~data['finished']).float().mean()):.2f} | "
            f"peak {max((r.get('device_used_mib', 0) for r in ph.values()), default=0)} MiB "
            f"retries {sum(r.get('alloc_retries', 0) for r in ph.values())} | "
            f"ETA {eta / 60:.0f} min ({left} steps)")
        append_jsonl(out / "train_log.jsonl", {
            "time": time.time(), "pid": os.getpid(), "step": step, "epoch": recs[0]["_epoch"],
            "sample_ids": data["sample_ids"], "phases": phases.pop(),
            "prefill_s": ro.prefill_s, "decode_s": ro.decode_s, "decode_passes": ro.decode_passes,
            "layer_copies": eng.loads - loads0, "weight_passes": eng.passes - passes0,
            "overlap": {"rollout": ov_rollout, "train": ov_train},
            "resident_min_max": [min(ro.resident_trace), max(ro.resident_trace)], "kv_mib": ro.kv_bytes / MiB,
            "reward": adv_stats, "reward_true_mean": float(data["rewards"].mean()),
            "reward_shuffled": bool(cfg.get("reward_shuffle")), "grad_norm": grad_norm, "len_mean": float(lengths.mean()),
            "truncated_frac": float((~data["finished"]).float().mean()),
            "has_box_frac": sum(i.get("has_box", False) for i in data["infos"]) / len(texts), **diag})
        append_jsonl(out / "samples.jsonl", {"step": step, "ids": data["sample_ids"],
                                             "epoch": recs[0]["_epoch"]})
        if (step + 1) % cfg["save_every"] == 0 or step == cfg["steps"] - 1:
            atomic_torch_save({"step": step, "lora": lora.state_dict(), "opt": opt.state_dict(),
                               "stream": stream.state_dict(), "gen": gen.get_state(), "config": cfg},
                              out / "checkpoint.pt")
            say(f"step {step}: checkpoint saved")
        maybe_kill(out, step, "after_ckpt", points)
    atomic_torch_save({"lora": lora.state_dict(), "r": lora.r, "alpha": lora.alpha, "targets": lora.targets,
                       "config": cfg}, out / "adapter.pt")


if __name__ == "__main__":
    main()
