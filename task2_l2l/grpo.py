"""GRPO pieces: group advantages, completion-only targets, an additive per-token loss, diagnostics.

One optimizer update per rollout batch, so the policy that sampled is the policy being trained: the PPO
ratio is exactly 1 and its clip is inactive. The gradient is then sum_i A_i * c_i * grad log pi(o_i),
c_i = 1 / (|o_i| * B) ("seq" normalisation, original GRPO) or 1 / total completion tokens ("token", DAPO).
Optional KL to the reference (adapter off) with the k3 estimator, per token: exp(d) - d - 1, d = ref - logp.

The loss VALUE is not a progress signal (it is ~0 by construction); log rewards, KL and the diagnostics.
"""
import torch

from engine import IGNORE


def group_advantages(rewards, G, scale_std=True, eps=1e-6):
    """rewards [B] in prompt-major order (G consecutive rollouts per prompt) -> (adv [B], stats)."""
    r = rewards.view(-1, G).double()
    std = r.std(-1, unbiased=False, keepdim=True)
    adv = r - r.mean(-1, keepdim=True)
    if scale_std:
        adv = adv / (std + eps)
    adv = torch.where(std > 0, adv, torch.zeros_like(adv))
    return adv.view(-1).float(), {"zero_std_groups": float((std == 0).float().mean()),
                                  "reward_mean": float(r.mean()), "reward_std_in_group": float(std.mean())}


def completion_targets(ids, comp):
    """targets [B, S]: position c-1 predicts column c for every completion column (prompt and pads IGNORE)."""
    t = torch.full(ids.shape, IGNORE, dtype=torch.long)
    t[:, :-1] = torch.where(comp[:, 1:], ids[:, 1:], torch.full_like(ids[:, 1:], IGNORE))
    return t


def token_coef(targets, norm="seq"):
    """Per-position normalisation c [B, S] (0 outside the loss)."""
    m = (targets != IGNORE).float()
    if norm == "seq":
        n = m.sum(-1, keepdim=True).clamp(min=1)
        return m / (n * targets.shape[0])
    if norm == "token":
        return m / m.sum().clamp(min=1)
    raise ValueError(norm)


def make_loss(coef, adv, ref_logp=None, beta=0.0, stash=None):
    """loss_fn(logp, flat_idx) for engine.train_grads. stash [B*S] receives the training-time log-probs."""
    c = coef.reshape(-1)
    a = adv[:, None].expand_as(coef).reshape(-1)
    ref = ref_logp.reshape(-1) if ref_logp is not None else None
    assert beta == 0 or ref is not None, "beta > 0 needs reference log-probs"

    def loss_fn(logp, idx):
        ci, ai = c.to(logp.device)[idx], a.to(logp.device)[idx]
        loss = -(ci * ai * logp)
        if beta:
            d = ref.to(logp.device)[idx] - logp
            loss = loss + beta * ci * (torch.exp(d) - d - 1)
        if stash is not None:
            stash[idx] = logp.detach().to(stash.device)
        return loss.sum()
    return loss_fn


def diagnostics(targets, train_logp, rollout_logp_cols, ref_logp=None):
    """Silent-failure signals, all over completion tokens:
    rollout_vs_train: |logp at sampling - logp in the training pass|; should be numerical noise. A systematic
                      gap means rollout and training see different inputs (template, positions, adapter state).
    kl_ref:           mean k3 KL(policy || reference); stays 0 if the adapter never changes anything."""
    m = targets != IGNORE
    out = {"completion_tokens": int(m.sum())}
    if m.any():
        gap = (train_logp[m] - rollout_logp_cols[m]).abs()
        out.update(rollout_vs_train_max=float(gap.max()), rollout_vs_train_mean=float(gap.mean()),
                   train_logp_mean=float(train_logp[m].mean()))
        if ref_logp is not None:
            d = ref_logp[m] - train_logp[m]
            out["kl_ref"] = float((torch.exp(d) - d - 1).mean())
    return out


def rollout_logp_grid(ro, S):
    """Decode-time log-probs placed on the training grid: position c-1 holds the logp of column c."""
    g = torch.zeros(ro.ids.shape[0], S)
    C = ro.logp.shape[1]
    g[:, ro.P - 1:ro.P - 1 + C] = ro.logp
    return g
