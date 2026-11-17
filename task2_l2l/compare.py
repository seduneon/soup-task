"""Equivalence rule shared by the unit tests and the real-model script.

The streamed engine runs the same floating-point operations as the reference, partly in a different order
(chunked head, per-layer recompute, micro-batches). So there is no fixed tolerance to pick: we measure the
reference's own rounding error against a high-precision truth (fp64 on CPU), and require the streamed
result to be no worse than RATIO times that, per tensor and in aggregate. A streamed path that is wrong
(stale layer, wrong order, severed graph) is off by O(1) relative error, many orders above that floor.

    err(x) = ||x - truth|| / ||truth||
    pass   = err(stream) <= RATIO * err(ref) + ABS_FLOOR   for every tensor and for all tensors concatenated
"""
import torch

RATIO = 1.5
ABS_FLOOR = 1e-12  # covers tensors where the reference happens to be exact (err(ref) = 0)


def rel_err(x, truth):
    x, truth = x.detach().double().cpu(), truth.detach().double().cpu()
    return (x - truth).norm().item() / max(truth.norm().item(), 1e-300)


def compare(stream, ref, truth, ratio=RATIO):
    """stream, ref, truth: dicts name -> tensor (same keys). Returns a report dict with 'pass'."""
    rows, worst = [], 0.0
    for k in truth:
        es, er = rel_err(stream[k], truth[k]), rel_err(ref[k], truth[k])
        ok = es <= ratio * er + ABS_FLOOR
        worst = max(worst, es / (er + ABS_FLOOR))
        rows.append(dict(name=k, err_stream=es, err_ref=er, ok=ok,
                         bitexact=bool(torch.equal(stream[k].cpu(), ref[k].cpu())),
                         nonzero=bool(truth[k].abs().sum() > 0)))
    cat = lambda d: torch.cat([d[k].detach().double().cpu().reshape(-1) for k in truth])
    es, er = rel_err(cat(stream), cat(truth)), rel_err(cat(ref), cat(truth))
    agg_ok = es <= ratio * er + ABS_FLOOR
    return dict(pass_=agg_ok and all(r["ok"] for r in rows), agg_err_stream=es, agg_err_ref=er,
                bitexact=all(r["bitexact"] for r in rows), all_nonzero=all(r["nonzero"] for r in rows),
                worst_ratio=worst, n=len(rows), rows=rows)


def lora_grads(lora):
    return {k: (v.grad.clone() if v.grad is not None else torch.zeros_like(v)) for k, v in lora.p.items()}
