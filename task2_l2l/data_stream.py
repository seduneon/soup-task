"""Prompts streamed from JSONL shards on disk, with deterministic resume; rollouts and checkpoints on disk.

PromptStream
  - one pass over the shards builds a byte-offset index (no record kept in memory); records are read by
    seek + readline when they are needed;
  - epoch e visits the records in randperm(N, seed, e) order; the cursor is (epoch, pos) and nothing else,
    so the order of every future sample is a pure function of (seed, cursor);
  - state_dict() carries a fingerprint of the dataset, so resuming against changed data fails loudly.

Commit protocol (train.py): a step is consumed only when its checkpoint is committed. The checkpoint is
written atomically (tmp + fsync + os.replace) and contains the cursor, LoRA, Adam and RNG states. The
consumed-samples ledger is appended before the checkpoint; on resume, ledger lines after the checkpoint's
step are dropped (they belong to steps that will be replayed identically).
"""
import hashlib
import json
import os
from pathlib import Path

import torch


def atomic_write_bytes(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class PromptStream:
    def __init__(self, data_dir, seed=0, shuffle=True):
        self.files = sorted(Path(data_dir).glob("*.jsonl"))
        assert self.files, f"no *.jsonl shards in {data_dir}"
        self.seed, self.shuffle = seed, shuffle
        self.index, ids, h = [], set(), hashlib.sha256()
        for fi, fn in enumerate(self.files):
            with open(fn, "rb") as f:
                while True:
                    off = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    rid = json.loads(line)["id"]
                    assert rid not in ids, f"duplicate id {rid!r} in {fn}"
                    ids.add(rid)
                    self.index.append((fi, off))
                    h.update(line)
        self.N, self.fingerprint = len(self.index), h.hexdigest()[:16]
        self.epoch, self.pos, self._order, self._handles = 0, 0, None, {}

    def order(self, epoch):
        if not self.shuffle:
            return list(range(self.N))
        return torch.randperm(self.N, generator=torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)).tolist()

    def read(self, k):
        fi, off = self.index[k]
        f = self._handles.get(fi) or self._handles.setdefault(fi, open(self.files[fi], "rb"))
        f.seek(off)
        return json.loads(f.readline())

    def next(self, n):
        """The next n records (crossing into the next epoch if needed), each tagged with its epoch."""
        out = []
        while len(out) < n:
            if self._order is None:
                self._order = self.order(self.epoch)
            rec = self.read(self._order[self.pos])
            rec["_epoch"] = self.epoch
            out.append(rec)
            self.pos += 1
            if self.pos == self.N:
                self.epoch, self.pos, self._order = self.epoch + 1, 0, None
        return out

    def state_dict(self):
        return {"epoch": self.epoch, "pos": self.pos, "seed": self.seed, "N": self.N, "fingerprint": self.fingerprint}

    def load_state_dict(self, s):
        assert (s["seed"], s["N"], s["fingerprint"]) == (self.seed, self.N, self.fingerprint), \
            "dataset or seed changed since the checkpoint"
        self.epoch, self.pos, self._order = s["epoch"], s["pos"], None


class RolloutStore:
    """Each step's rollout batch is written to disk after generation and read back by the training passes."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, step):
        return self.root / f"step_{step:06d}.pt"

    def save(self, step, data):
        atomic_torch_save(data, self.path(step))

    def load(self, step):
        return torch.load(self.path(step), weights_only=False)


def write_jsonl_shards(records, out_dir, per_shard):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in range(0, len(records), per_shard):
        with open(out_dir / f"shard_{s // per_shard:04d}.jsonl", "w") as f:
            for r in records[s:s + per_shard]:
                f.write(json.dumps(r) + "\n")
