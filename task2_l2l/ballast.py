#!/usr/bin/env python3
"""Make a larger card behave like an 8 GB one: a separate process holds (total - cap) of device memory.

    python ballast.py --cap-mib 8192 &      # then run train/evaluate/merge with cap_offset_mib = printed footprint

An allocator limit (set_per_process_memory_fraction) is not a cap: it ignores the CUDA context and any
non-PyTorch allocation. The ballast is physical: whatever the training process does, it cannot get more than
`cap` from the driver. Evidence stays the nvidia-smi logs: per-process used_memory for our PID (compute-apps
CSV) must stay <= cap. Prefer renting an 8 GB card; use this only on 12 GB+ cards.
"""
import argparse
import signal
import time

import torch

MiB = 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-mib", type=int, default=8192)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.cuda.init()
    torch.empty(1, device=dev)  # create this process's context first, so it is counted
    free, total = torch.cuda.mem_get_info(dev)
    want = total - args.cap_mib * MiB          # bytes this process must occupy in total
    already = total - free                     # our context + anything else already on the card
    hold = want - already
    assert hold > 0, f"card has {total / MiB:.0f} MiB; nothing to hold for a {args.cap_mib} MiB cap"
    t = torch.empty(hold, dtype=torch.uint8, device=dev)
    t.fill_(1)  # touch it
    free2, _ = torch.cuda.mem_get_info(dev)
    print(f"ballast holds {hold / MiB:.0f} MiB; device used {(total - free2) / MiB:.0f} MiB of {total / MiB:.0f}; "
          f"left for the job {free2 / MiB:.0f} MiB; set cap_offset_mib={(total - free2) / MiB:.0f}", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: exit(0))
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
