"""Part 3 verification script: the decision rule on synthetic arms (each rule must be able to fail), and the
real verify.py -> evaluate.py -> merge chain end to end on a tiny on-disk model (CPU, fp32)."""
import json
import subprocess
import sys

import pytest
import torch

from common import ROOT, local_tokenizer_dir, save_tiny_model_dir
from data_stream import write_jsonl_shards
from lora import LoRAParams, lora_shapes
from merge import merge
from verify import compare

N = 120


def arm(p_correct, seed, boxed=True, texts=None):
    g = torch.Generator().manual_seed(seed)
    r = (torch.rand(N, generator=g) < p_correct).float().tolist()
    items = {f"q{i}": [{"id": f"q{i}", "reward": r[i], "has_box": boxed, "text": (texts or {}).get(i, str(r[i]))}]
             for i in range(N)}
    s = {"data": "d", "n_items": N, "k": 1, "temperature": 0.0, "pass_rate": sum(r) / N}
    return s, items


def test_rule_trained():
    base, ad = arm(0.3, 0), arm(0.7, 1)
    rep = compare({"base": base, "adapter": ad, "merged": ad, "control": arm(0.3, 2)})
    assert rep["checks"] == {"R1_effect": True, "R2_not_generic": True, "R3_merge": True, "R4_not_format": True}
    assert rep["verdict"] == "TRAINED"
    assert rep["merged_adapter_text_identical"] == 1.0


def test_rule_fails_when_control_gains_too():
    ad = arm(0.7, 1)
    rep = compare({"base": arm(0.3, 0), "adapter": ad, "merged": ad, "control": arm(0.7, 3)})
    assert rep["checks"]["R2_not_generic"] is False and rep["verdict"].startswith("NOT SHOWN")


def test_rule_fails_when_merge_lost_the_adapter():
    rep = compare({"base": arm(0.3, 0), "adapter": arm(0.7, 1), "merged": arm(0.3, 0), "control": arm(0.3, 2)})
    assert rep["checks"]["R3_merge"] is False


def test_rule_no_effect():
    rep = compare({"base": arm(0.5, 0), "adapter": arm(0.5, 1), "merged": arm(0.5, 1), "control": arm(0.5, 2)})
    assert rep["checks"]["R1_effect"] is False


def test_format_only_gain_is_flagged():
    """The adapter only learned to box: among items both arms boxed it is no better than base."""
    base_s, base_items = arm(0.3, 0, boxed=False)
    for i in range(0, N, 2):  # base boxed half the items; adapter boxes all and gets those right at the same rate
        base_items[f"q{i}"][0]["has_box"] = True
    ad_s, ad_items = arm(0.3, 0)
    for i in range(1, N, 2):  # extra correct answers only where base had no box
        ad_items[f"q{i}"][0]["reward"] = 1.0
    rep = compare({"base": (base_s, base_items), "adapter": (ad_s, ad_items), "merged": (ad_s, ad_items),
                   "control": (base_s, base_items)})
    assert rep["checks"]["R1_effect"] is True and rep["checks"]["R4_not_format"] is False


def test_refuses_mismatched_arms():
    s, items = arm(0.5, 0)
    items.pop("q0")
    with pytest.raises(AssertionError, match="different items"):
        compare({"base": arm(0.5, 0), "adapter": (s, items)})
    s2, items2 = arm(0.5, 1)
    s2["temperature"] = 1.0
    with pytest.raises(AssertionError, match="temperature"):
        compare({"base": arm(0.5, 0), "adapter": (s2, items2)})


@pytest.mark.skipif(local_tokenizer_dir() is None, reason="needs a Qwen3 tokenizer in the HF cache")
def test_verify_end_to_end(tmp_path):
    L = 2
    cfg = save_tiny_model_dir(tmp_path / "model", L=L)

    def save_adapter(seed, path):
        torch.manual_seed(seed)
        lora = LoRAParams(lora_shapes(cfg), L, r=4, alpha=8)
        lora.randomize_B(std=0.05, seed=seed)
        ad = {"lora": lora.state_dict(), "r": lora.r, "alpha": lora.alpha, "targets": lora.targets}
        torch.save(ad, path)
        return ad

    ad = save_adapter(1, tmp_path / "adapter.pt")
    save_adapter(2, tmp_path / "control.pt")
    merge(tmp_path / "model", ad, tmp_path / "merged")
    write_jsonl_shards([{"id": f"h{i}", "prompt": f"What is {i} times 3?", "answer": str(3 * i)} for i in range(6)],
                       tmp_path / "heldout", per_shard=4)
    out = tmp_path / "verify"
    cmd = [sys.executable, str(ROOT / "verify.py"), "--model", str(tmp_path / "model"),
           "--adapter", str(tmp_path / "adapter.pt"), "--merged", str(tmp_path / "merged"),
           "--control", str(tmp_path / "control.pt"), "--data", str(tmp_path / "heldout"), "--out", str(out),
           "--n", "6", "--max-new", "6", "--reward", "toy_ascii", "--device", "cpu", "--dtype", "float32",
           "--resident", "1", "--batch", "4"]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-3000:]
    rep = json.load(open(out / "verify_report.json"))
    assert rep["n_items"] == 6 and set(rep["arms"]) == {"base", "adapter", "merged", "control"}
    assert rep["merged_adapter_text_identical"] == 1.0  # fp32 merge: same model, same greedy text
    assert rep["checks"]["R3_merge"] is True
    texts = {nm: [json.loads(l)["text"] for l in open(out / f"{nm}.jsonl")] for nm in ("base", "adapter")}
    assert texts["base"] != texts["adapter"], "adapter arm decoded like the base: adapter not applied"
    p2 = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)  # second call reuses the arms
    assert p2.returncode == 0 and p2.stdout.count("reusing") == 4
