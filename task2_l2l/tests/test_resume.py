"""Deterministic resume: SIGKILL train.py mid-epoch (three times, at different points), resume, and compare
with an uninterrupted run: same samples in the same order, no duplicates, no gaps, and the same rollouts and
final LoRA weights bit for bit. Runs the real train.py on a tiny on-disk Qwen3 (CPU)."""
import json
import os
import subprocess
import sys

import pytest
import torch
import yaml

from common import ROOT, local_tokenizer_dir, save_tiny_model_dir
from data_stream import PromptStream, write_jsonl_shards

N_PROMPTS, PROMPTS_PER_STEP, STEPS = 10, 2, 8  # 5 steps per epoch: the run crosses into epoch 1

pytestmark = pytest.mark.skipif(local_tokenizer_dir() is None, reason="needs a Qwen3 tokenizer in the HF cache")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("resume")
    save_tiny_model_dir(root / "model", L=3)
    recs = [{"id": f"q{i:02d}", "prompt": f"What is {i} + {i}?", "answer": str(2 * i)} for i in range(N_PROMPTS)]
    write_jsonl_shards(recs, root / "data", per_shard=4)  # 3 shards
    base = yaml.safe_load(open(ROOT / "train_config.yaml"))
    base.update(model=str(root / "model"), device="cpu", dtype="float32", data_dir=str(root / "data"), resident=1,
                prompts=PROMPTS_PER_STEP, G=3, max_new=6, reward="toy_ascii", steps=STEPS, lr=1e-2, chunk=4)
    return root, base


def run_to_completion(root, cfg, name, kill_at=""):
    cfg = dict(cfg, out_dir=str(root / name))
    path = root / f"{name}.yaml"
    yaml.safe_dump(cfg, open(path, "w"))
    env = dict(os.environ, L2L_KILL_AT=kill_at, PYTHONPATH=str(ROOT))
    codes = []
    for _ in range(6):
        p = subprocess.run([sys.executable, str(ROOT / "train.py"), "--config", str(path)], env=env,
                           capture_output=True, text=True)
        codes.append(p.returncode)
        if p.returncode == 0:
            return root / name, codes
        assert p.returncode == -9, p.stderr[-3000:]  # only our SIGKILL is allowed to stop it
    raise AssertionError(f"did not finish: {codes}")


def ledger(out):
    return [json.loads(l) for l in open(out / "samples.jsonl")]


def test_stream_order_and_resume_unit(tmp_path):
    recs = [{"id": f"r{i}", "prompt": "p", "answer": "a"} for i in range(7)]
    write_jsonl_shards(recs, tmp_path, per_shard=3)
    a = PromptStream(tmp_path, seed=5)
    full = [r["id"] for r in a.next(20)]
    assert sorted(full[:7]) == sorted(r["id"] for r in recs) and sorted(full[7:14]) == sorted(full[:7])
    assert full[:7] != full[7:14]  # a new permutation each epoch
    b = PromptStream(tmp_path, seed=5)
    head = [r["id"] for r in b.next(9)]
    c = PromptStream(tmp_path, seed=5)
    c.load_state_dict(b.state_dict())
    assert head + [r["id"] for r in c.next(11)] == full
    (tmp_path / "shard_0000.jsonl").write_text(json.dumps({"id": "r0", "prompt": "changed", "answer": "a"}) + "\n")
    with pytest.raises(AssertionError):
        PromptStream(tmp_path, seed=5).load_state_dict(b.state_dict())


@pytest.mark.parametrize("save_every", [1, 2])
def test_kill_resume_identical(setup, save_every):
    root, base = setup
    cfg = dict(base, save_every=save_every)
    ref, ref_codes = run_to_completion(root, cfg, f"ref_{save_every}")
    got, codes = run_to_completion(root, cfg, f"killed_{save_every}", kill_at="2:train,4:rollout,6:after_ckpt")
    assert ref_codes == [0] and codes.count(-9) == 3 and codes[-1] == 0

    a, b = ledger(ref), ledger(got)
    assert a == b, "consumed samples differ after kill/resume"
    assert [x["step"] for x in b] == list(range(STEPS))
    ids = [i for x in b for i in x["ids"]]
    per_epoch = {}
    for x in b:
        per_epoch.setdefault(x["epoch"], []).extend(x["ids"])
    assert sorted(per_epoch[0]) == [f"q{i:02d}" for i in range(N_PROMPTS)], "epoch 0: duplicates or gaps"
    assert len(per_epoch[1]) == len(set(per_epoch[1])), "epoch 1: duplicates"
    assert len(ids) == STEPS * PROMPTS_PER_STEP

    for s in range(STEPS):  # rollouts (sampled tokens, rewards) are identical, not just the sample ids
        ra, rb = torch.load(ref / "rollouts" / f"step_{s:06d}.pt"), torch.load(got / "rollouts" / f"step_{s:06d}.pt")
        assert torch.equal(ra["ids"], rb["ids"]) and torch.equal(ra["rewards"], rb["rewards"])
    la, lb = torch.load(ref / "adapter.pt")["lora"], torch.load(got / "adapter.pt")["lora"]
    assert all(torch.equal(la[k], lb[k]) for k in la), "final LoRA differs after kill/resume"
    assert any(v.abs().sum() > 0 for k, v in la.items() if k.endswith("_B")), "LoRA never trained: test is vacuous"

    log = [json.loads(l) for l in open(got / "train_log.jsonl")]
    attempts = [json.loads(l) for l in open(got / "attempts.jsonl")]
    assert len(attempts) == 4  # start + 3 resumes, each recorded with its cursor
    # save_every=1: every kill lands before the step's log line or after its checkpoint -> no step is logged twice.
    # save_every=2: steps after the last checkpoint are replayed, and the raw log keeps both attempts.
    replayed = len(log) - STEPS
    assert replayed == 0 if save_every == 1 else replayed > 0


def test_kill_before_first_checkpoint_no_duplicates(setup):
    """save_every=2: a kill after step 0 (logged, never checkpointed) must not leave step 0 twice in the ledger."""
    root, base = setup
    cfg = dict(base, save_every=2, steps=3)
    ref, _ = run_to_completion(root, cfg, "ref_early")
    got, codes = run_to_completion(root, cfg, "killed_early", kill_at="0:after_ckpt")
    assert codes == [-9, 0]
    assert ledger(got) == ledger(ref) and [x["step"] for x in ledger(got)] == [0, 1, 2]


def test_reward_shuffle_control_is_a_permutation_and_resumes(setup):
    """Part 3 negative control: the trained-on rewards are a seeded permutation of the true ones (same values,
    broken pairing), the true reward is still logged, and kill/resume stays exact."""
    root, base = setup
    cfg = dict(base, reward_shuffle=True, steps=4)
    ref, _ = run_to_completion(root, cfg, "shuf_ref")
    got, codes = run_to_completion(root, cfg, "shuf_killed", kill_at="2:train")
    assert codes == [-9, 0]
    moved = 0
    for s in range(4):
        r = torch.load(ref / "rollouts" / f"step_{s:06d}.pt")
        assert torch.equal(r["train_rewards"].sort().values, r["rewards"].sort().values)
        moved += int(not torch.equal(r["train_rewards"], r["rewards"]))
        assert torch.equal(r["train_rewards"], torch.load(got / "rollouts" / f"step_{s:06d}.pt")["train_rewards"])
    assert moved > 0, "no step was actually shuffled: the control is vacuous"
    log = [json.loads(l) for l in open(ref / "train_log.jsonl")]
    assert all(x["reward_shuffled"] and "reward_true_mean" in x for x in log)
    la, lb = torch.load(ref / "adapter.pt")["lora"], torch.load(got / "adapter.pt")["lora"]
    assert all(torch.equal(la[k], lb[k]) for k in la)
