"""Tests for PeripheralEncoder."""

import torch
import pytest
from src.peripheral_encoder import PeripheralEncoder


@pytest.fixture
def encoder():
    return PeripheralEncoder(hidden_dim=64, block_size=8)


def test_output_shape(encoder):
    """Output should be (batch, num_blocks, hidden_dim)."""
    x = torch.randn(2, 32, 64)  # 32 tokens / 8 block_size = 4 blocks
    out = encoder(x)
    assert out.shape == (2, 4, 64)


def test_output_shape_longer_sequence(encoder):
    """Works with longer sequences."""
    x = torch.randn(1, 128, 64)  # 128 / 8 = 16 blocks
    out = encoder(x)
    assert out.shape == (1, 16, 64)


def test_padding_when_not_divisible(encoder):
    """Pads sequence to nearest multiple of block_size."""
    x = torch.randn(2, 30, 64)  # 30 not divisible by 8 -> pads to 32 -> 4 blocks
    out = encoder(x)
    assert out.shape == (2, 4, 64)


def test_attention_mask(encoder):
    """Masked tokens should not contribute to pooling."""
    x = torch.randn(1, 16, 64)
    # Mask out the second block entirely
    mask = torch.ones(1, 16)
    mask[:, 8:16] = 0
    out = encoder(x, attention_mask=mask)
    assert out.shape == (1, 2, 64)
    # The second block should still produce output (zeros from nan_to_num + pos embed)
    assert not torch.isnan(out).any()


def test_gradient_flow(encoder):
    """Gradients should flow back through the encoder."""
    x = torch.randn(2, 32, 64, requires_grad=True)
    out = encoder(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert (x.grad != 0).any()


def test_single_block(encoder):
    """Works when sequence is exactly one block."""
    x = torch.randn(1, 8, 64)
    out = encoder(x)
    assert out.shape == (1, 1, 64)


def test_batch_independence(encoder):
    """Each batch element should be processed independently."""
    x = torch.randn(2, 16, 64)
    out = encoder(x)
    # Process each element separately
    out0 = encoder(x[0:1])
    out1 = encoder(x[1:2])
    torch.testing.assert_close(out[0:1], out0)
    torch.testing.assert_close(out[1:2], out1)
