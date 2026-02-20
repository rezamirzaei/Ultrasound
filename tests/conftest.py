"""Shared test fixtures for ultrasound imaging toolkit."""

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_grayscale():
    """Create a sample 64x64 grayscale image (uint8)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(64, 64), dtype=np.uint8)


@pytest.fixture
def sample_float_image():
    """Create a sample 64x64 float image in [0, 1]."""
    rng = np.random.default_rng(42)
    return rng.random((64, 64)).astype(np.float64)


@pytest.fixture
def sample_rgb():
    """Create a sample 64x64x3 RGB image (uint8)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def binary_mask_pair():
    """Create a pair of binary masks for metric testing."""
    pred = np.zeros((64, 64), dtype=np.uint8)
    target = np.zeros((64, 64), dtype=np.uint8)
    # Overlapping circles
    y, x = np.ogrid[:64, :64]
    pred[((x - 30) ** 2 + (y - 30) ** 2) < 100] = 1
    target[((x - 32) ** 2 + (y - 32) ** 2) < 100] = 1
    return pred, target


@pytest.fixture
def torch_batch():
    """Create a dummy torch batch (B=2, C=3, H=64, W=64)."""
    return torch.randn(2, 3, 64, 64)


@pytest.fixture
def torch_seg_pair():
    """Create a logits/target pair for segmentation loss testing."""
    logits = torch.randn(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    return logits, target
