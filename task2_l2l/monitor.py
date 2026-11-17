"""Per-phase wall time and memory, and the nvidia-smi sampler that runs for the whole job.

Phase boundaries synchronise CUDA and reset the peak counters, so each phase reports its own
max_memory_allocated / max_memory_reserved (what the Part 1 ledger predicts per phase). nvidia-smi is
sampled every 200 ms into a raw CSV: that, not the allocator, is what the 8 GB cap is judged on.
"""
import shutil
import subprocess
import time
from contextlib import contextmanager

import psutil
import torch

MiB = 2**20


class Phases:
    """cap_mib: if set, every phase end checks device-wide used memory (total - free, what nvidia-smi shows,
    context included) and records over_cap; strict=True raises instead. The nvidia-smi CSV stays the evidence;
    this is the in-process alarm."""

    def __init__(self, device, cap_mib=None, strict=False, offset_mib=0):
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.cap_mib, self.strict, self.offset_mib = cap_mib, strict, offset_mib  # offset: ballast footprint
        self.rec = {}

    @contextmanager
    def __call__(self, name):
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.cuda:
                torch.cuda.synchronize(self.device)
            r = {"s": round(time.perf_counter() - t0, 4), "rss_mib": round(psutil.Process().memory_info().rss / MiB)}
            if self.cuda:
                free, total = torch.cuda.mem_get_info(self.device)
                r.update(alloc_peak_mib=round(torch.cuda.max_memory_allocated(self.device) / MiB),
                         reserved_peak_mib=round(torch.cuda.max_memory_reserved(self.device) / MiB),
                         device_used_mib=round((total - free) / MiB - self.offset_mib))
                if self.cap_mib is not None and r["device_used_mib"] > self.cap_mib:
                    r["over_cap"] = True
                    if self.strict:
                        raise RuntimeError(f"phase {name}: device used {r['device_used_mib']} MiB > cap {self.cap_mib} MiB")
            self.rec[name] = r

    def pop(self):
        rec, self.rec = self.rec, {}
        return rec


class NvSmi:
    QUERY = ("timestamp,index,memory.used,memory.total,utilization.gpu,clocks.sm,clocks.mem,"
             "pcie.link.gen.current,pcie.link.width.current,temperature.gpu,power.draw")

    def __init__(self, path, interval_ms=200):
        """Two raw CSVs: device-wide (path) and per process (path with .apps.csv: pid, used_memory), so the cap
        can be shown for our PID even when a ballast or another process shares the card."""
        self.path, self.procs = str(path), []
        if shutil.which("nvidia-smi"):
            self.procs.append(subprocess.Popen(["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv",
                                                f"-lms={interval_ms}", f"--filename={self.path}"]))
            self.procs.append(subprocess.Popen(["nvidia-smi", "--query-compute-apps=timestamp,pid,used_memory",
                                                "--format=csv", f"-lms={interval_ms}",
                                                f"--filename={self.path.replace('.csv', '')}.apps.csv"]))

    def stop(self):
        for p in self.procs:
            p.terminate()
            p.wait(timeout=10)
