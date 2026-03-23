"""Peripheral Encoder: O(n) global context compression via learned weighted block pooling."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PeripheralEncoder(nn.Module):
    """Compresses full token sequence into a low-resolution peripheral map.

    Divides the sequence into blocks of size B and produces a single vector
    per block using learned weighted pooling, then adds positional embeddings.

    Input:  (batch, seq_len, hidden_dim)
    Output: (batch, num_blocks, hidden_dim)
    """

    def __init__(self, hidden_dim: int, block_size: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.block_size = block_size

        # Learned weighted pooling: produces per-token importance score within a block
        self.weight_proj = nn.Linear(hidden_dim, 1)

        # Layer norm applied after pooling
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Positional embeddings for blocks (will be lazily expanded if needed)
        self.max_blocks = 1024  # supports up to 1024 * block_size = 32K tokens
        self.pos_embedding = nn.Embedding(self.max_blocks, hidden_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim) — token embeddings
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            peripheral_map: (batch, num_blocks, hidden_dim)
        """
        batch, seq_len, hidden_dim = x.shape  # (B_batch, N, D)

        # Pad sequence length to multiple of block_size
        remainder = seq_len % self.block_size
        if remainder != 0:
            pad_len = self.block_size - remainder
            # x: (batch, seq_len, D) -> (batch, seq_len + pad_len, D)
            x = F.pad(x, (0, 0, 0, pad_len))
            if attention_mask is not None:
                # attention_mask: (batch, seq_len) -> (batch, seq_len + pad_len)
                attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
            seq_len = seq_len + pad_len

        num_blocks = seq_len // self.block_size

        # Reshape into blocks: (batch, num_blocks, block_size, D)
        blocks = rearrange(x, 'b (nb bs) d -> b nb bs d', bs=self.block_size)

        # Compute per-token weights within each block
        # weight_logits: (batch, num_blocks, block_size, 1)
        weight_logits = self.weight_proj(blocks)
        # weight_logits: (batch, num_blocks, block_size)
        weight_logits = weight_logits.squeeze(-1)

        # Mask padded tokens so they don't contribute to pooling
        if attention_mask is not None:
            # block_mask: (batch, num_blocks, block_size)
            block_mask = rearrange(attention_mask, 'b (nb bs) -> b nb bs', bs=self.block_size)
            weight_logits = weight_logits.masked_fill(block_mask == 0, float('-inf'))

        # weights: (batch, num_blocks, block_size) — softmax over tokens within each block
        weights = F.softmax(weight_logits, dim=-1)

        # Handle fully-masked blocks (all -inf -> nan after softmax)
        weights = weights.nan_to_num(0.0)

        # Weighted sum: (batch, num_blocks, D)
        # weights: (batch, num_blocks, block_size) @ blocks: (batch, num_blocks, block_size, D)
        peripheral_map = torch.einsum('bnk,bnkd->bnd', weights, blocks)

        # Layer norm
        peripheral_map = self.layer_norm(peripheral_map)

        # Add positional embeddings: (num_blocks, D)
        positions = torch.arange(num_blocks, device=x.device)
        peripheral_map = peripheral_map + self.pos_embedding(positions)

        return peripheral_map  # (batch, num_blocks, hidden_dim)
