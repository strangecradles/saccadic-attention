"""Foveal Processor: O(k²) high-resolution attention over a local window."""

import torch
import torch.nn as nn


class FovealProcessor(nn.Module):
    """Applies multi-head self-attention over a small window centered at the fixation point.

    The current accumulated state is prepended as a CLS-like token. After attention,
    the output at the CLS position becomes the updated state.

    Input:  full embeddings x, fixation_point, current state, window_size
    Output: updated state vector (batch, hidden_dim)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, window_size: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.window_size = window_size

        # Multi-head self-attention over the foveal window + state token
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Layer norm and FFN for processing the updated state
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        fixation_point: torch.Tensor,
        state: torch.Tensor,
        accumulated_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_dim) — full token embeddings
            fixation_point: (batch,) — integer fixation positions per batch element
            state: (batch, hidden_dim) — current accumulated state
            accumulated_context: (batch, prev_tokens, hidden_dim) — tokens from all
                previous fixation windows, or None for the first saccade

        Returns:
            updated_state: (batch, hidden_dim)
            new_accumulated_context: (batch, prev_tokens + window_size, hidden_dim)
        """
        batch, seq_len, hidden_dim = x.shape  # (B, N, D)

        # Extract window for each batch element
        # window: (batch, window_size, hidden_dim)
        window = self._extract_windows(x, fixation_point)

        # Build full context: [CLS state] + [previous fixation windows] + [current window]
        state_token = state.unsqueeze(1)  # (batch, 1, D)
        if accumulated_context is not None:
            # full_context: (batch, 1 + prev_tokens + window_size, D)
            full_context = torch.cat([state_token, accumulated_context, window], dim=1)
        else:
            # full_context: (batch, 1 + window_size, D)
            full_context = torch.cat([state_token, window], dim=1)

        # Self-attention over the full accumulated context
        normed = self.norm1(full_context)
        attn_out, _ = self.attn(normed, normed, normed)
        full_context = full_context + attn_out

        # FFN on the CLS token output
        # cls_out: (batch, D)
        cls_out = full_context[:, 0, :]
        cls_out = cls_out + self.ffn(self.norm2(cls_out))

        # Update accumulated context (exclude CLS token, include new window)
        if accumulated_context is not None:
            new_accumulated = torch.cat([accumulated_context, window], dim=1)
        else:
            new_accumulated = window  # (batch, window_size, D)

        return cls_out, new_accumulated  # (batch, D), (batch, accumulated_tokens, D)

    def _extract_windows(
        self,
        x: torch.Tensor,
        fixation_point: torch.Tensor,
    ) -> torch.Tensor:
        """Extract a window of tokens centered at the fixation point for each batch element.

        Args:
            x: (batch, seq_len, hidden_dim)
            fixation_point: (batch,) — center positions

        Returns:
            windows: (batch, window_size, hidden_dim)
        """
        batch, seq_len, hidden_dim = x.shape
        half_w = self.window_size // 2

        windows = []
        for i in range(batch):
            center = fixation_point[i].item()
            # Clamp window to sequence boundaries
            start = max(0, center - half_w)
            end = min(seq_len, center + half_w)
            # Adjust if window is truncated at boundaries
            if end - start < self.window_size:
                if start == 0:
                    end = min(seq_len, self.window_size)
                else:
                    start = max(0, end - self.window_size)

            # window_i: (window_size, D) or smaller if seq_len < window_size
            window_i = x[i, start:end, :]

            # Pad if sequence is shorter than window_size
            actual_len = end - start
            if actual_len < self.window_size:
                pad = torch.zeros(
                    self.window_size - actual_len, hidden_dim,
                    device=x.device, dtype=x.dtype,
                )
                window_i = torch.cat([window_i, pad], dim=0)

            windows.append(window_i)

        # (batch, window_size, D)
        return torch.stack(windows, dim=0)
