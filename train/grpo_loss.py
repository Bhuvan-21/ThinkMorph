"""
GRPO loss computation: clipped surrogate objective with KL regularization.
"""

import torch


def compute_grpo_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    clip_epsilon: float = 0.2,
    kl_weight: float = 0.01,
) -> dict:
    """
    Compute the GRPO policy gradient loss.

    Args:
        current_log_probs: Per-token log π_θ(a|s) under the current (updated) policy.
            Shape: (num_tokens,)
        old_log_probs: Per-token log π_old(a|s) from the rollout policy (detached).
            Shape: (num_tokens,)
        ref_log_probs: Per-token log π_ref(a|s) from the frozen reference (detached).
            Shape: (num_tokens,)
        advantages: Per-token advantages (broadcast from per-completion group-relative
            advantages). Shape: (num_tokens,)
        loss_mask: Binary mask for valid tokens (1 = compute loss, 0 = ignore).
            Shape: (num_tokens,)
        clip_epsilon: PPO-style clipping range [1-ε, 1+ε].
        kl_weight: Coefficient β for the KL penalty term.

    Returns:
        Dict with 'loss' (scalar), 'policy_loss' (scalar), 'kl_loss' (scalar),
        'clip_fraction' (scalar, for monitoring).
    """
    # Importance sampling ratio
    log_ratio = current_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # Clipped surrogate
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2)

    # Approximate KL: E_π_θ[log π_θ - log π_ref]
    kl = current_log_probs - ref_log_probs
    kl_loss = kl_weight * kl

    # Combined per-token loss
    per_token_loss = policy_loss + kl_loss

    # Mask and average
    num_valid = loss_mask.sum().clamp(min=1)
    loss = (per_token_loss * loss_mask).sum() / num_valid

    # Monitoring stats (detached)
    with torch.no_grad():
        clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float()
        clip_fraction = (clip_fraction * loss_mask).sum() / num_valid
        mean_kl = (kl * loss_mask).sum() / num_valid
        mean_policy_loss = (policy_loss * loss_mask).sum() / num_valid

    return {
        "loss": loss,
        "policy_loss": mean_policy_loss,
        "kl_loss": mean_kl,
        "clip_fraction": clip_fraction,
    }


def compute_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute group-relative advantages from per-completion rewards.

    Args:
        rewards: Shape (G,) — one reward per completion in the group.
        eps: Small constant for numerical stability when std ≈ 0.

    Returns:
        Advantages: Shape (G,) — normalized within the group.
    """
    mean = rewards.mean()
    std = rewards.std()
    if std < eps:
        return torch.zeros_like(rewards)
    return (rewards - mean) / (std + eps)
