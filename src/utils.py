"""Utility functions: metrics, visualization helpers."""

import torch


def count_flops_per_forward(
    seq_len: int,
    hidden_dim: int,
    num_saccades: int,
    window_size: int,
    block_size: int,
    num_heads: int,
    num_saccadic_layers: int,
) -> dict:
    """Estimate FLOPs for saccadic attention vs full attention.

    Returns dict with flop counts for both approaches.
    """
    # Full attention: 2 * n^2 * d per layer (Q@K^T + attn@V)
    full_attn_flops = 2 * seq_len * seq_len * hidden_dim * num_saccadic_layers

    # Saccadic attention per layer:
    #   Peripheral encoder: O(n * d) — linear scan
    peripheral_flops = seq_len * hidden_dim
    #   Controller per saccade: O(n/B * d)
    controller_flops = (seq_len // block_size) * hidden_dim * num_saccades
    #   Foveal processor per saccade: O(k^2 * d)
    foveal_flops = window_size * window_size * hidden_dim * num_saccades
    saccadic_flops = (peripheral_flops + controller_flops + foveal_flops) * num_saccadic_layers

    return {
        'full_attention_flops': full_attn_flops,
        'saccadic_attention_flops': saccadic_flops,
        'speedup_ratio': full_attn_flops / max(saccadic_flops, 1),
        'peripheral_flops': peripheral_flops * num_saccadic_layers,
        'controller_flops': controller_flops * num_saccadic_layers,
        'foveal_flops': foveal_flops * num_saccadic_layers,
    }


def fixation_entropy(fixation_logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy of fixation distribution.

    Args:
        fixation_logits: (batch, num_blocks) or (num_saccades, batch, num_blocks)

    Returns:
        entropy: scalar
    """
    probs = torch.softmax(fixation_logits, dim=-1)
    return -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
